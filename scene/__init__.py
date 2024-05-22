#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json

from utils.system_utils import searchForMaxIteration
from games.scenes import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

import imageio
import numpy as np
import torch


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            if args.gs_type == "gs_multi_mesh":
                scene_info = sceneLoadTypeCallbacks["Colmap_Mesh"](
                    args.source_path, args.images, args.eval, args.num_splats, args.meshes
                )
            else:
                scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            if args.gs_type == "gs_mesh":
                print("Found transforms_train.json file, assuming Blender_Mesh data set!")
                scene_info = sceneLoadTypeCallbacks["Blender_Mesh"](
                    args.source_path, args.white_background, args.eval, args.num_splats[0]
                )
            elif args.gs_type == "gs_flame":
                print("Found transforms_train.json file, assuming Flame Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender_FLAME"](args.source_path, args.white_background, args.eval)
            else:
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            if args.gs_type == "gs_multi_mesh":
                for i, ply_path in enumerate(scene_info.ply_path):
                    with open(ply_path, 'rb') as src_file, open(os.path.join(self.model_path, f"input_{i}.ply") , 'wb') as dest_file:
                        dest_file.write(src_file.read())
            else:
                with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.point_cloud = scene_info.point_cloud
            if args.gs_type == "gs_mesh":
                self.gaussians.triangles = scene_info.point_cloud.triangles
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def load_flame_data(self, basedir, expressions=True, load_frontal_faces=False, load_bbox=True):
        print("Starting flame data loading")
        splits = ["train", "val", "test"]
        metas = {}
        for s in splits:
            with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
                metas[s] = json.load(fp)

        all_imgs = []
        all_poses = []
        all_expressions = []
        all_bboxs = []
        counts = [0]
        for s in splits:
            meta = metas[s]
            imgs = []
            poses = []
            expressions = []
            bboxs = []
            
            for frame in meta["frames"][::1]:
                fname = os.path.join(basedir, frame["file_path"] + ".png")
                imgs.append(imageio.imread(fname))
                poses.append(np.array(frame["transform_matrix"]))
                expressions.append(np.array(frame["expression"]))
                if load_bbox:
                    if "bbox" not in frame.keys():
                        bboxs.append(np.array([0.0,1.0,0.0,1.0]))
                    else:
                        bboxs.append(np.array(frame["bbox"]))

            imgs = (np.array(imgs) / 255.0).astype(np.float32)
            poses = np.array(poses).astype(np.float32)
            expressions = np.array(expressions).astype(np.float32)
            bboxs = np.array(bboxs).astype(np.float32)

            counts.append(counts[-1] + imgs.shape[0])
            all_imgs.append(imgs)
            all_poses.append(poses)
            all_expressions.append(expressions)
            all_bboxs.append(bboxs)

        i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

        imgs = np.concatenate(all_imgs, 0)
        poses = np.concatenate(all_poses, 0)
        expressions = np.concatenate(all_expressions, 0)
        bboxs = np.concatenate(all_bboxs, 0)

        H, W = imgs[0].shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        intrinsics = meta["intrinsics"] if meta["intrinsics"] else None
        if meta["intrinsics"]:
            intrinsics = np.array(meta["intrinsics"])
        else:
            intrinsics = np.array([focal, focal, 0.5, 0.5]) # fx fy cx cy

        render_poses = torch.stack(
            [
                torch.from_numpy(self.pose_spherical(angle, -30.0, 4.0))
                for angle in np.linspace(-180, 180, 40 + 1)[:-1]
            ],
            0,
        )

        imgs = [
            torch.from_numpy(imgs[i]) for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)

        poses = torch.from_numpy(poses)
        expressions = torch.from_numpy(expressions)
        bboxs[:,0:2] *= H
        bboxs[:,2:4] *= W
        bboxs = np.floor(bboxs)
        bboxs = torch.from_numpy(bboxs).int()
        print("Done with flame data loading")

        return imgs, poses, render_poses, [H, W, intrinsics], i_split, expressions, bboxs

    def pose_spherical(self, theta, phi, radius):
        c2w = self.translate_by_t_along_z(radius)
        c2w = self.rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
        c2w = self.rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w
    
    def translate_by_t_along_z(self, t):
        tform = np.eye(4).astype(np.float32)
        tform[2][3] = t
        return tform

    def rotate_by_phi_along_x(self, phi):
        tform = np.eye(4).astype(np.float32)
        tform[1, 1] = tform[2, 2] = np.cos(phi)
        tform[1, 2] = -np.sin(phi)
        tform[2, 1] = -tform[1, 2]
        return tform

    def rotate_by_theta_along_y(self, theta):
        tform = np.eye(4).astype(np.float32)
        tform[0, 0] = tform[2, 2] = np.cos(theta)
        tform[0, 2] = -np.sin(theta)
        tform[2, 0] = -tform[0, 2]
        return tform
