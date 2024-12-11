#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# sparse gradio demo functions
# --------------------------------------------------------
import math
import os
from glob import glob
import numpy as np
import trimesh
import copy
from scipy.spatial.transform import Rotation
from pathlib import Path
import argparse
import pickle
import math
import sys
import cv2

sys.path.insert(0, "./thirdparty/mast3r")

from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.utils.schedules import cosine_schedule
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
import matplotlib.pyplot as pl
from mast3r.model import AsymmetricMASt3R
import torch

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12


def get_sparse_optim_args(config):
    optim_args = {
        "lr1": 0.2,
        "niter1": 500,
        "lr2": 0.02,
        "niter2": 500,
        "opt_pp": True,
        "opt_depth": True,
        "schedule": "cosine",
        "depth_mode": "add",
        "exp_depth": False,
        "lora_depth": False,
        "shared_intrinsics": True,
        "device": "cuda",
        "dtype": torch.float32,
        "matching_conf_thr": 5.0,
        "loss_dust3r_w": 0.01,
    }

    # update with config
    for key, value in config.items():
        if key in optim_args:
            optim_args[key] = value
    optim_args["opt_depth"] = "depth" in config["optim_level"]
    optim_args["schedule"] = cosine_schedule

    return optim_args


def set_scenegraph_options(inputfiles, win_cyclic, refid, scenegraph_type, winsize=10):
    num_files = len(inputfiles) if inputfiles is not None else 1
    max_winsize, min_winsize = 1, 1
    if scenegraph_type == "swin":
        if win_cyclic:
            max_winsize = max(1, math.ceil((num_files - 1) / 2))
        else:
            max_winsize = num_files - 1
    elif scenegraph_type == "logwin":
        if win_cyclic:
            half_size = math.ceil((num_files - 1) / 2)
            max_winsize = max(1, math.ceil(math.log(half_size, 2)))
        else:
            max_winsize = max(1, math.ceil(math.log(num_files, 2)))
    winsize = min(max_winsize, max(min_winsize, winsize))

    return winsize, win_cyclic


def get_geometries_from_scene(scene, clean_depth, mask_sky, min_conf_thr):

    # post processes - clean depth, mask sky
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg, focals = scene.imgs, scene.get_focals().cpu()

    print("obtained focal length : ", focals)
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d, depths, confs = to_numpy(
        scene.get_dense_pts3d(clean_depth=clean_depth)
    )  # ref mast3r/cloud_opt SparseGA() class
    msk = to_numpy([c > min_conf_thr for c in confs])

    # get normalized depthmaps
    depths_max = max([d.max() for d in depths])  # type: ignore
    depths = [d / depths_max for d in depths]

    return rgbimg, pts3d, msk, focals, cams2world, depths, depths_max


def points_to_pct(pts, color, extrinsic, save_path=None):
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", [np.deg2rad(180)]).as_matrix()

    pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=color.reshape(-1, 3))
    pct.apply_transform(np.linalg.inv(extrinsic))

    if save_path is not None:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        pct.export(save_path)
    return pct


def save_each_geometry(
    outdir,
    imgs,
    pts3d,
    mask,
    focals,
    cams2world,
    imgname=None,
    depths=None,
    depths_max=None,
    filter_pct=True,
):
    print(
        f"len(pts3d) : {len(pts3d)}, len(mask) : {len(mask)}, len(imgs) : {len(imgs)}, len(cams2world) : {len(cams2world)}"
    )
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world)
    extrinsic = cams2world[0].detach().numpy()
    pts3d, imgs, focals, cams2world = (
        to_numpy(pts3d),
        to_numpy(imgs),
        to_numpy(focals),
        to_numpy(cams2world),
    )

    if imgname is not None:
        outdir = os.path.join(outdir, imgname["data"])

    # original : dict_keys(['focal', 'cam2worlds', 'pct2worlds', 'pointcloud_paths', 'max_depths', 'depths'])
    global_dict = {
        "focals": [],
        "cam2worlds": [],
        "pointcloud_paths": [],
        "max_depths": [],
        "depths": [],
        "masks": [],
    }

    args_iter = (
        zip(pts3d, imgs, mask, focals, cams2world)
        if depths is None
        else zip(pts3d, imgs, mask, focals, cams2world, depths)
    )
    for i, arguments in enumerate(args_iter):
        if depths is None:
            points, img, point_mask, focal, cam2world = arguments
            depth, depths_max = None, None
        else:
            points, img, point_mask, focal, cam2world, depth = arguments

        if filter_pct:
            pts = points[point_mask.ravel()].reshape(-1, 3)
            col = img[point_mask].reshape(-1, 3)
            valid_msk = np.isfinite(pts.sum(axis=1))
            pts, col = pts[valid_msk], col[valid_msk]
        else:
            pts, col = points.reshape(-1, 3), img.reshape(-1, 3)

        filename = (
            f"pointcloud_{i:04d}.ply"
            if imgname is None
            else f"{imgname['img_nums'][i]:04d}_pointcloud_{i:04d}.ply"
        )
        points_to_pct(
            pts,
            col,
            np.eye(extrinsic.shape[0]),
            save_path=os.path.join(outdir, filename),
        )
        cam_params = {
            "focal": focal,
            "cam2world": cam2world,
            "c2w_original": cam2world,
            "depth": depth,
            "depth_max": depths_max,
            "base_extrinsic": extrinsic,
            "imgname": imgname,
        }
        maskdir = os.path.join(outdir, "masks")
        os.makedirs(maskdir, exist_ok=True)

        # save mask to png
        re_mask = point_mask.reshape(depth.shape)
        mask = np.zeros_like(re_mask, dtype=np.uint8)
        mask[re_mask == 0] = 255
        cv2.imwrite(os.path.join(maskdir, f"{i:04d}.png"), mask)

        with open(os.path.join(outdir, filename.replace(".ply", ".pkl")), "wb") as f:
            pickle.dump(cam_params, f)

        global_dict["focals"].append(focal)
        global_dict["cam2worlds"].append(cam2world)
        global_dict["pointcloud_paths"].append(os.path.join(outdir, filename))
        global_dict["max_depths"].append(depths_max)
        global_dict["depths"].append(depth)
        global_dict["masks"].append(point_mask)

    return global_dict


def get_reconstructed_scene(
    outdir,
    cache_dir,
    model,
    device,
    image_size,
    filelist,
    optim_level,
    lr1,
    niter1,
    lr2,
    niter2,
    min_conf_thr,
    matching_conf_thr,
    mask_sky,
    clean_depth,
    scenegraph_type,
    winsize,
    win_cyclic,
    refid,
    TSDF_thresh,
    shared_intrinsics,
    filter_pct,
    optim_args,
    **kw,
):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=True)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1
        filelist = [filelist[0], filelist[0] + "_2"]

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append("noncyclic")
    scene_graph = "-".join(scene_graph_params)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    if optim_level == "coarse":
        niter2 = 0

    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(filelist, pairs, cache_dir, model, **optim_args)

    rgbimg, pts3d, msk, focals, cams2world, depths, depths_max = (
        get_geometries_from_scene(scene, clean_depth, mask_sky, min_conf_thr)
    )
    global_dict = save_each_geometry(
        os.path.join(outdir, "op_results"),
        rgbimg,
        pts3d,
        msk,
        focals,
        cams2world,
        imgname=None,
        depths=depths,
        depths_max=depths_max,
        filter_pct=filter_pct,
    )

    with open(os.path.join(outdir, "global_params.pkl"), "wb") as f:
        pickle.dump(global_dict, f)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data", help="input directory")
    parser.add_argument(
        "--output_dir", type=str, default="output", help="output directory"
    )
    parser.add_argument("--exp_name", type=str, default="exp", help="experiment name")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        help="mast3r ckpt",
    )
    parser.add_argument(
        "--cache_dir", type=str, default="optim_cache", help="cache directory"
    )

    args = parser.parse_args()

    device = "cuda"
    model = AsymmetricMASt3R.from_pretrained(args.ckpt).to(device)

    filelist = sorted(glob(os.path.join(args.input_dir, "*.png")))

    cache_dir = os.path.join(
        args.cache_dir,
        f"{os.path.basename(os.path.dirname(args.input_dir))}_{np.random.randint(1e6):05d}",
    )
    outdir = os.path.join(args.output_dir, args.exp_name + "_000")

    data = {"cache_dir": "optim_cache"}
    optimization = {
        "device": "cuda",
        "image_size": 512,
        "shared_intrinsics": True,
        "win_cyclic": False,
        "lr1": 0.07,
        "niter1": 500,
        "lr2": 0.014,
        "niter2": 200,
        "optim_level": "refine+depth",
        "scenegraph_type": "swin",
        "winsize": 10,
        "refid": 0,
        "TSDF_thresh": 0,
        "schedule": "cosine",
        "min_conf_thr": 1.5,
        "matching_conf_thr": 5,
        "mask_sky": False,
        "clean_depth": True,
        "filter_pct": True,
    }
    arg_dict = dict()

    arg_dict.update(data)
    arg_dict.update(optimization)
    arg_dict.update({"filelist": filelist, "model": model, "schedule": cosine_schedule})
    arg_dict.update(
        {"outdir": outdir, "input_dir": args.input_dir, "cache_dir": cache_dir}
    )
    winsize, win_cyclic = set_scenegraph_options(filelist, False, 0, "swin", 10)
    arg_dict.update({"winsize": winsize, "win_cyclic": win_cyclic})
    arg_dict.update({"optim_args": get_sparse_optim_args(optimization)})

    get_reconstructed_scene(**arg_dict)


if __name__ == "__main__":
    main()
