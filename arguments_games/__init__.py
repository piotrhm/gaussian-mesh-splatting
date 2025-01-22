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

from arguments import ParamGroup


class OptimizationParamsMesh(ParamGroup):
    def __init__(self, parser):
        self.iterations = 600_000
        self.vertices_lr = 0.0  # 0.00016
        self.alpha_lr = 0.001
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.017
        self.rotation_lr = 0.001
        self.random_background = False
        self.use_mesh = True
        self.lambda_dssim = 0.2
        super().__init__(parser, "Optimization Parameters")


class OptimizationParamsFlame(ParamGroup):
    def __init__(self, parser):
        self.iterations = 600_000
        self.alpha_lr = 0.001
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.017
        self.rotation_lr = 0.001
        self.flame_shape_lr = 0.01
        self.flame_expr_lr = 1e-3
        self.flame_eyes_lr = 1e-5
        self.flame_neck_lr = 1e-5
        self.flame_translation_lr = 1e-6
        self.flame_rotation_lr = 1e-5
        self.flame_jaw_lr = 1e-5
        self.vertices_enlargement_lr = 0.0002
        self.random_background = False
        self.use_mesh = True
        self.lambda_dssim = 0.2
        super().__init__(parser, "Optimization Parameters")
