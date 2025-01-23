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

import torch
from scene import Scene
import numpy as np
from tqdm import tqdm
from renderer.flame_gaussian_renderer import flame_render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from games.flame_splatting.scene.gaussian_flame_model import GaussianFlameModel

from utils.loss_utils import ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr



def render_and_evaluate(name, views, gaussians, pipeline, background):
    ssims = []
    psnrs = []
    lpipss = []
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gaussians.update_alpha(view.timestep_index, view.flame_params)
        gaussians.prepare_scaling_rot()
        
        rendering = flame_render(view, gaussians, pipeline, background, recalc=True)["render"]        
        image = torch.clamp(rendering, 0.0, 1.0)
        gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
        
        ssims.append(ssim(image, gt_image))
        psnrs.append(psnr(image, gt_image))
        lpipss.append(lpips(image, gt_image, net_type='vgg'))

    return {
        "SSIM": torch.tensor(ssims).mean(),
        "PSNR": torch.tensor(psnrs).mean(),
        "LPIPS": torch.tensor(lpipss).mean()
    }


def evaluate_model(
    gs_type: str,
    dataset : ModelParams,
    iteration : int,
    pipeline : PipelineParams,
    skip_train : bool,
    skip_test : bool,
    skip_val: bool,
):
    with torch.no_grad():
        gaussians = GaussianFlameModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
            out_train = render_and_evaluate("train", scene.getTrainCameras()[:10], scene.gaussians, pipeline, background)
            print("Train:", out_train)

        if not skip_test:
            out_test = render_and_evaluate("test", scene.getTestCameras()[:10], scene.gaussians, pipeline, background)
            print("Test:", out_test)
            
        if not skip_val:
            out_val = render_and_evaluate("val", scene.getValCameras()[:10], scene.gaussians, pipeline, background)
            print("Val:", out_val)
            
if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--gs_type', type=str, default="gs_flame")
    parser.add_argument("--num_splats", nargs="+", type=int, default=10)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    model.gs_type = args.gs_type
    model.num_splats = args.num_splats
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    evaluate_model(args.gs_type, model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_val)
