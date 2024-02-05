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

from arguments import ParamGroup


class OptimizationParamsMesh(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.vertices_lr = 0.00016
        self.alpha_lr = 0.001
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.random_background = False
        self.use_mesh = True
        super().__init__(parser, "Optimization Parameters")


class OptimizationParamsFlame(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.alpha_lr = 0.001
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.flame_shape_lr = 0.01
        self.flame_exp_lr = 0.001
        self.flame_pose_lr = 0.001
        self.flame_neck_pose_lr = 0.001
        self.flame_trans_lr = 0.001
        self.vertices_enlargement_lr = 0.0002
        self.random_background = False
        self.use_mesh = True
        super().__init__(parser, "Optimization Parameters")
