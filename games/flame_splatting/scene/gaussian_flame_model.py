#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use
#

import torch
import numpy as np

from torch import nn
from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid
from games.mesh_splatting.utils.general_utils import rot_to_quat_batch
from utils.sh_utils import RGB2SH
from games.mesh_splatting.utils.graphics_utils import MeshPointCloud


class GaussianFlameModel(GaussianModel):

    def __init__(self, sh_degree: int):

        super().__init__(sh_degree)
        self.point_cloud = None
        self.flame_params = None
        self._alpha = torch.empty(0)
        self.alpha = torch.empty(0)
        self.softmax = torch.nn.Softmax(dim=2)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.update_alpha_func = self.softmax

        self.vertices = None
        self.faces = None
        self._scales = torch.empty(0)
        self.faces = torch.empty(0)
        self._vertices_enlargement = torch.empty(0)

        self.use_2d_flying_splats = False
        self._xy = torch.empty(0)          # raw trainable 2D centers
        self.xy = torch.empty(0)           # final 2D centers
        self._scales_2d = torch.empty(0)   # e.g. 2D scale
        self._rotation_2d = torch.empty(0) # e.g. 2D orientation angle
        # ----------------------------------------------------

    @property
    def get_xyz(self):
        return self._xyz

    def create_from_pcd(self, pcd: MeshPointCloud, spatial_lr_scale: float):

        self.point_cloud = pcd
        self.spatial_lr_scale = spatial_lr_scale
        pcd_alpha_shape = pcd.alpha.shape

        print("Number of faces: ", pcd_alpha_shape[0])
        print("Number of points at initialisation in face: ", pcd_alpha_shape[1])

        alpha_point_cloud = pcd.alpha.float().cuda()
        scales = torch.ones((pcd.points.shape[0], 1)).float().cuda()

        print("Number of points at initialisation : ",
              alpha_point_cloud.shape[0] * alpha_point_cloud.shape[1])

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        opacities = inverse_sigmoid(0.1 * torch.ones((pcd.points.shape[0], 1), dtype=torch.float, device="cuda"))

        self.create_flame_params()

        self._alpha = nn.Parameter(alpha_point_cloud.requires_grad_(True))  # check update_alpha
        self.update_alpha()
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scales = nn.Parameter(scales.requires_grad_(True))
        self.prepare_scaling_rot()
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def create_2d_splats(self, points_2d, colors=None):
        """
        Initialize "flying" 2D splats, storing their 2D centers and
        hooking them into the same color/opacity pipeline as 3D.
        """
        self.use_2d_flying_splats = True  # <--- this triggers 2D logic

        if not isinstance(points_2d, torch.Tensor):
            points_2d = torch.tensor(points_2d, dtype=torch.float32)
        points_2d = points_2d.cuda()

        # Our 2D centers become a trainable parameter
        self._xy = nn.Parameter(points_2d, requires_grad=True)

        # If you want an opacity or "alpha" for each splat:
        self._alpha = nn.Parameter(
            inverse_sigmoid(0.1 * torch.ones((points_2d.shape[0], 1), device="cuda")),
            requires_grad=True
        )
        self.alpha = torch.sigmoid(self._alpha)

        # For color, re-use the same approach as 3D
        if colors is not None:
            if not isinstance(colors, torch.Tensor):
                colors = torch.tensor(colors, dtype=torch.float32).cuda()
            fused_color = RGB2SH(colors) if colors.ndim == 2 else colors
        else:
            # random color init
            fused_color = RGB2SH(torch.rand(points_2d.shape[0], 3, device="cuda"))

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2),
                               dtype=torch.float32, device="cuda")
        features[:, :3, 0] = fused_color  # set DC
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )

        # For 2D scale, let's store (scale_x, scale_y)
        self._scales_2d = nn.Parameter(0.05 * torch.ones(points_2d.shape[0], 2, device="cuda"), requires_grad=True)

        # If you want 2D rotation angles, e.g. for elliptical Gaussians
        self._rotation_2d = nn.Parameter(
            torch.zeros(points_2d.shape[0], device="cuda"),
            requires_grad=True
        )

        # Possibly store an explicit "opacity" parameter if you like
        self._opacity = nn.Parameter(
            inverse_sigmoid(0.1 * torch.ones((points_2d.shape[0], 1), device="cuda")),
            requires_grad=True
        )

        # Ensure we compute the 3D placeholders
        self.update_alpha()

    def prepare_timestep_index_db(self, train_cameras):
        if self.flame_params is None:
            self.flame_params = {}

        for camera in train_cameras:
            timestep_index, flame_params_path = camera.timestep_index, camera.flame_params
            flame_params = np.load(flame_params_path)

            params = self.flame_params.get(timestep_index, None)
            if params is None:
                params = {
                    "shape": nn.Parameter(torch.unsqueeze(torch.from_numpy(flame_params['shape']).to(device='cuda'), 0).requires_grad_(True)),
                    "expr": nn.Parameter(torch.from_numpy(flame_params['expr']).to(device='cuda').requires_grad_(True)),
                    "eyes": nn.Parameter(torch.from_numpy(flame_params['eyes_pose']).to(device='cuda').requires_grad_(True)),
                    "neck": nn.Parameter(torch.from_numpy(flame_params['neck_pose']).to(device='cuda').requires_grad_(True)),
                    "translation": nn.Parameter(torch.from_numpy(flame_params['translation']).to(device='cuda').requires_grad_(True)),
                    "rotation": nn.Parameter(torch.from_numpy(flame_params['rotation']).to(device='cuda').requires_grad_(True)),
                    "jaw": nn.Parameter(torch.from_numpy(flame_params['jaw_pose']).to(device='cuda').requires_grad_(True)),
                }
                self.flame_params[timestep_index] = params

    def create_flame_params(self):
        """
        Create manipulation parameters FLAME model.

        Each parameter is responsible for something different,
        respectively: shape, facial expression, etc.
        """
        self.flame_params = {
            "base": {
                "shape": nn.Parameter(self.point_cloud.flame_model_shape_init.requires_grad_(True)),
                "expr": nn.Parameter(self.point_cloud.flame_model_expression_init.requires_grad_(True)),
                "eyes": nn.Parameter(self.point_cloud.flame_model_eyes_pose_init.requires_grad_(True)),
                "neck": nn.Parameter(self.point_cloud.flame_model_neck_pose_init.requires_grad_(True)),
                "translation": nn.Parameter(self.point_cloud.flame_model_transl_init.requires_grad_(True)),
                "rotation": nn.Parameter(self.point_cloud.flame_model_rotation_init.requires_grad_(True)),
                "jaw": nn.Parameter(self.point_cloud.flame_model_jaw_pose_init.requires_grad_(True)),
            }
        }
        self.faces = self.point_cloud.faces

        vertices_enlargement = torch.ones_like(self.point_cloud.vertices_init).requires_grad_(True)
        self._vertices_enlargement = nn.Parameter(self.point_cloud.vertices_enlargement_init * vertices_enlargement)

    def _calc_xyz(self):
        """
        calculate the 3d Gaussian center in the coordinates xyz.

        The alphas that are taken into account are the distances
        to the vertices and the coordinates of
        the triangles forming the mesh.

        """
        if not self.use_2d_flying_splats:
            _xyz = torch.matmul(
                self.alpha,
                self.vertices[self.faces]
            )
            self._xyz = _xyz.reshape(
                _xyz.shape[0] * _xyz.shape[1], 3
            )
        else:
            self.xy = torch.sigmoid(self._xy) if self.update_alpha_func is None else self._xy
            z_zeros = torch.zeros_like(self.xy[:, :1])
            self._xyz = torch.cat((self.xy, z_zeros), dim=-1)
    def prepare_scaling_rot(self, eps=1e-8):
        """
        approximate covariance matrix and calculate scaling/rotation tensors

        covariance matrix is [v0, v1, v2], where
        v0 is a normal vector to each face
        v1 is a vector from centroid of each face and 1st vertex
        v2 is obtained by orthogonal projection of a vector from centroid
        to 2nd vertex onto subspace spanned by v0 and v1
        """
        if not self.use_2d_flying_splats:
            def dot(v, u):
                return (v * u).sum(dim=-1, keepdim=True)

            def proj(v, u):
                """
                projection of vector v onto subspace spanned by u

                vector u is assumed to be already normalized
                """
                coef = dot(v, u)
                return coef * u

            triangles = self.vertices[self.faces]
            normals = torch.linalg.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
                dim=1
            )
            v0 = normals / (torch.linalg.vector_norm(normals, dim=-1, keepdim=True) + eps)
            means = torch.mean(triangles, dim=1)
            v1 = triangles[:, 1] - means
            v1_norm = torch.linalg.vector_norm(v1, dim=-1, keepdim=True) + eps
            v1 = v1 / v1_norm
            v2_init = triangles[:, 2] - means
            v2 = v2_init - proj(v2_init, v0) - proj(v2_init, v1)  # Gram-Schmidt
            v2 = v2 / (torch.linalg.vector_norm(v2, dim=-1, keepdim=True) + eps)

            s1 = v1_norm / 2.
            s2 = dot(v2_init, v2) / 2.
            s0 = eps * torch.ones_like(s1)
            scales = torch.concat((s0, s1, s2), dim=1).unsqueeze(dim=1)
            scales = scales.broadcast_to((*self.alpha.shape[:2], 3))

            self._scaling = torch.log(
                torch.nn.functional.relu(self._scales * scales.flatten(start_dim=0, end_dim=1)) + eps
            )

            rotation = torch.stack((v0, v1, v2), dim=1).unsqueeze(dim=1)
            rotation = rotation.broadcast_to((*self.alpha.shape[:2], 3, 3)).flatten(start_dim=0, end_dim=1)
            rotation = rotation.transpose(-2, -1)
            self._rotation = rot_to_quat_batch(rotation)

        else:
            self._scaling_2d = torch.log(torch.relu(self._scales_2d) + eps)
            pass

    def update_alpha(self, timestep_index=-1, flame_params=None):
        """
        Function to control the alpha value.

        Alpha is the distance of the center of the gauss
         from the vertex of the triangle of the mesh.
        Thus, for each center of the gauss, 3 alphas
        are determined: alpha1+ alpha2+ alpha3.
        For a point to be in the center of the vertex,
        the alphas must meet the assumptions:
        alpha1 + alpha2 + alpha3 = 1
        and alpha1 + alpha2 +alpha3 >= 0

        #TODO
        check:
        # self.alpha = torch.relu(self._alpha)
        # self.alpha = self.alpha / self.alpha.sum(dim=-1, keepdim=True)

        """
        if not self.use_2d_flying_splats:
            self.alpha = self.update_alpha_func(self._alpha)

            if flame_params is None:
                params = self.flame_params["base"]
                print("Flame params set to base")
            else:
                params = self.flame_params.get(timestep_index, None)
                flame_params = np.load(flame_params)
                if params is None:
                    params = {
                        "shape": nn.Parameter(torch.unsqueeze(torch.from_numpy(flame_params['shape']).to(device='cuda'), 0).requires_grad_(True)),
                        "expr": nn.Parameter(torch.from_numpy(flame_params['expr']).to(device='cuda').requires_grad_(True)),
                        "eyes": nn.Parameter(torch.from_numpy(flame_params['eyes_pose']).to(device='cuda').requires_grad_(True)),
                        "neck": nn.Parameter(torch.from_numpy(flame_params['neck_pose']).to(device='cuda').requires_grad_(True)),
                        "translation": nn.Parameter(torch.from_numpy(flame_params['translation']).to(device='cuda').requires_grad_(True)),
                        "rotation": nn.Parameter(torch.from_numpy(flame_params['rotation']).to(device='cuda').requires_grad_(True)),
                        "jaw": nn.Parameter(torch.from_numpy(flame_params['jaw_pose']).to(device='cuda').requires_grad_(True)),
                    }
                    self.flame_params[timestep_index] = params

            vertices, _ = self.point_cloud.flame_model(
                shape=params["shape"],
                expr=params["expr"],
                eyes=params["eyes"],
                neck=params["neck"],
                translation=params["translation"],
                rotation=params["rotation"],
                jaw=params["jaw"],
            )

            self.vertices = self.point_cloud.transform_vertices_function(
                vertices,
                self._vertices_enlargement
            )
            self._calc_xyz()

        else:
            self.alpha = torch.sigmoid(self._alpha)
            self._calc_xyz()

    def training_setup(self, training_args):
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        shape_params, expr_params, eyes_params, neck_params = [], [], [], []
        translation_params, rotation_params, jaw_params = [], [], []
        for flame_dict in self.flame_params.values():
            shape_params.append(flame_dict['shape'])
            expr_params.append(flame_dict['expr'])
            eyes_params.append(flame_dict['eyes'])
            neck_params.append(flame_dict['neck'])
            translation_params.append(flame_dict['translation'])
            rotation_params.append(flame_dict['rotation'])
            jaw_params.append(flame_dict['jaw'])

        lr_params = [
            {'params': shape_params, 'lr': training_args.flame_shape_lr, "name": "shape"},
            {'params': expr_params, 'lr': training_args.flame_expr_lr, "name": "expression"},
            {'params': eyes_params, 'lr': training_args.flame_eyes_lr, "name": "eyes"},
            {'params': neck_params, 'lr': training_args.flame_neck_lr, "name": "neck"},
            {'params': translation_params, 'lr': training_args.flame_translation_lr, "name": "translation"},
            {'params': rotation_params, 'lr': training_args.flame_rotation_lr, "name": "rotation"},
            {'params': jaw_params, 'lr': training_args.flame_jaw_lr, "name": "jaw"},
            {'params': [self._vertices_enlargement], 'lr': training_args.vertices_enlargement_lr, "name": "vertices_enlargement"},
            {'params': [self._alpha], 'lr': training_args.alpha_lr, "name": "alpha"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scales], 'lr': training_args.scaling_lr, "name": "scaling"},  # for 3D
        ]

        if self.use_2d_flying_splats:
            lr_params += [
                {'params': [self._xy], 'lr': training_args.alpha_lr, "name": "xy_2d"},
                {'params': [self._scales_2d], 'lr': training_args.scaling_lr, "name": "scaling_2d"},
                {'params': [self._rotation_2d], 'lr': training_args.scaling_lr, "name": "rotation_2d"},
            ]

        self.optimizer = torch.optim.Adam(lr_params, lr=0.0, eps=1e-15)

    def save_ply(self, path):
        self._save_ply(path)

        attrs = self.__dict__
        flame_additional_attrs = [
            'flame_params',
            '_vertices_enlargement', '_alpha', 'faces', '_scales', '_opacity',
            'alpha', 'point_cloud',
        ]

        if self.use_2d_flying_splats:
            flame_additional_attrs += ['_xy', '_scales_2d', '_rotation_2d']

        save_dict = {}
        for attr_name in flame_additional_attrs:
            save_dict[attr_name] = attrs[attr_name]

        path_flame = path.replace('point_cloud.ply', 'flame_params.pt')
        torch.save(save_dict, path_flame)

    def load_ply(self, path):
        self._load_ply(path)
        path_flame = path.replace('point_cloud.ply', 'flame_params.pt')
        params = torch.load(path_flame)
        self.flame_params = params['flame_params']
        self._vertices_enlargement = params['_vertices_enlargement']
        self.faces = params['faces']
        self.alpha = params['alpha']
        self._alpha = params['_alpha']
        self._scales = params['_scales']
        self._opacity = params['_opacity']
        self.point_cloud = params['point_cloud']

        if '_xy' in params:
            self.use_2d_flying_splats = True
            self._xy = params['_xy']
            self._scales_2d = params['_scales_2d']
            self._rotation_2d = params['_rotation_2d']

        self.update_alpha()
