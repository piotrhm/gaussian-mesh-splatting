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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir, batch_size=10):
    ssims = []
    psnrs = []
    lpipss = []

    render_images = []
    gt_images = []

    # Preload images
    for fname in tqdm(renders_dir.iterdir(), desc="Loading Images"):
        render = Image.open(renders_dir / fname)
        render = tf.to_tensor(render)[:, :3, :, :].cuda()
        render_images.append(render)

        gt = Image.open(gt_dir / fname)
        gt = tf.to_tensor(gt)[:, :3, :, :].cuda()
        gt_images.append(gt)

        # Process in batches
        if len(render_images) == batch_size:
            render_images = torch.stack(render_images)
            gt_images = torch.stack(gt_images)

            for render, gt in zip(render_images, gt_images):
                ssims.append(ssim(render.unsqueeze(0), gt.unsqueeze(0)))
                psnrs.append(psnr(render.unsqueeze(0), gt.unsqueeze(0)))
                lpipss.append(lpips(render.unsqueeze(0), gt.unsqueeze(0), net_type='vgg'))

            render_images = []
            gt_images = []

    # Process remaining images
    if render_images:
        render_images = torch.stack(render_images)
        gt_images = torch.stack(gt_images)

        for render, gt in zip(render_images, gt_images):
            ssims.append(ssim(render.unsqueeze(0), gt.unsqueeze(0)))
            psnrs.append(psnr(render.unsqueeze(0), gt.unsqueeze(0)))
            lpipss.append(lpips(render.unsqueeze(0), gt.unsqueeze(0), net_type='vgg'))

    return ssims, psnrs, lpipss

def evaluate(gs_type, model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        #try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / f"renders_{gs_type}"
                ssims, psnrs, lpipss = readImages(renders_dir, gt_dir)

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                # per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                #                                             "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                #                                             "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + f"/results_{gs_type}.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            # with open(scene_dir + f"/per_view_{gs_type}.json", 'w') as fp:
            #     json.dump(per_view_dict[scene_dir], fp, indent=True)
        #except:
        #    print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Metrics script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--gs_type', type=str, default="gs_flat")
    args = parser.parse_args()
    evaluate(args.gs_type, args.model_paths)
