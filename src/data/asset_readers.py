# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
from pathlib import Path
from PIL import Image
import json
import pickle

import numpy as np
import torch
from torchvision.transforms import ToTensor

from src.utils.graphic_utils import focal2fov, quaternion_to_matrix
from src.utils.point_utils import uniform_sample, merge_pcds
from .utils import fetchPly


class GTCameraReader:
    # To read full poses (for evaluation)

    def __init__(self, dirpath, fname, **kwargs):

        poses = []

        with open(os.path.join(dirpath, fname)) as json_file:
            contents = json.load(json_file)
            fovx = contents["camera_angle_x"]
            for frame in contents["frames"]:
                c2w = np.array(frame["transform_matrix"], dtype=np.float32)
                poses.append(torch.from_numpy(c2w))
        self._poses = np.array(torch.stack(poses))
        self._fovx = np.deg2rad(fovx)

    def get_poses(self, idx=None):
        if idx is None:
            return self._poses
        else:
            return self._poses[idx]

    def get_fovx(self, idx):
        return self._fovx


class DepthAnythingReader:
    prefix: str = "depth_anything"

    def __init__(self, **kwargs):
        pass

    def __call__(self, dirpath, basename):
        asset_path = Path(dirpath).joinpath(self.prefix)
        base_name_wo_ext = os.path.splitext(basename)[0]
        base_name_tiff = base_name_wo_ext + ".npy"
        file_path = asset_path.joinpath(base_name_tiff)
        depth = -torch.from_numpy(np.load(file_path.as_posix())).unsqueeze(0)
        return (depth - depth.min()) / (depth.max() - depth.min())


class TAMMaskReader:
    prefix: str = "tam_mask"
    to_tensor = ToTensor()

    def __init__(self, split, resolution=1):
        assert split in ["train", "val", "test"]
        self.prefix = f"tam_mask"
        self.resolution = resolution

    def __call__(self, dirpath, basename):
        asset_path = Path(dirpath).joinpath(self.prefix)
        base_name_wo_ext = os.path.splitext(basename)[0]
        rgb_idx = base_name_wo_ext.split("_")[-1]
        rgb_idx = rgb_idx.zfill(6)
        base_name_png = f"{rgb_idx}.jpg"
        file_path = asset_path.joinpath(base_name_png)
        if not file_path.exists():
            base_name_png = f"{rgb_idx}.png"
            file_path = asset_path.joinpath(base_name_png)
        if self.resolution != 1:
            img = Image.open(file_path)
            w, h = img.size
            new_size = (w // self.resolution, h // self.resolution)
            return self.to_tensor(img.resize(new_size, Image.NEAREST)) > 0
        else:
            return self.to_tensor(Image.open(file_path)) > 0


class Test_MASt3RFovCameraReader:
    # To read full poses and trained fov (for evaluation)

    def __init__(self, dirpath, fname, mast3r_expname, mast3r_img_res, **kwargs):

        self.dirname = "mast3r_opt"
        poses = []

        # read gt test poses from test_transforms.json
        with open(os.path.join(dirpath, fname)) as json_file:
            contents = json.load(json_file)
            for frame in contents["frames"]:
                c2w = frame["transform_matrix"]
                poses.append(c2w)
        self._poses = np.array(poses, dtype=np.float32)

        # read fov from dust3r init
        pkl_path = Path(dirpath, self.dirname, mast3r_expname, "global_params.pkl")
        with open(pkl_path.as_posix(), "rb") as pkl_file:
            data = pickle.load(pkl_file)

        self._fovx = focal2fov(data["focals"][0], mast3r_img_res)

    def get_poses(self, idx=None):
        if idx is None:
            return self._poses
        else:
            return self._poses[idx]

    def get_fovx(self, idx):
        return self._fovx


class MASt3RCameraReader:

    dirname = "mast3r_opt"

    def __init__(self, dirpath, mast3r_expname, mast3r_img_res, **kwargs):

        pkl_path = Path(dirpath, self.dirname, mast3r_expname, "global_params.pkl")
        with open(pkl_path.as_posix(), "rb") as pkl_file:
            data = pickle.load(pkl_file)

        self._fovx = focal2fov(data["focals"][0], mast3r_img_res)
        self._poses = data["cam2worlds"]

    def get_poses(self, idx):
        return self._poses[idx]

    def get_fovx(self, idx):
        return self._fovx


class MASt3R_CKPTCameraReader:

    dirname = "mast3r_opt"

    def __init__(self, dirpath, ckpt_path, mast3r_expname, mast3r_img_res, **kwargs):

        # read poses from ckpt
        ckpt = torch.load(ckpt_path)
        c2w_rot = quaternion_to_matrix(ckpt[0]["camera"]["R_c2ws_quat"])
        c2w_trans = ckpt[0]["camera"]["T_c2ws"].unsqueeze(-1)
        c2w = torch.cat((c2w_rot, c2w_trans), dim=-1)
        bottom_row = (
            torch.tensor([[0, 0, 0, 1]], dtype=c2w_rot.dtype)
            .expand(c2w.shape[0], -1, -1)
            .cuda()
        )
        self._poses = torch.cat((c2w, bottom_row), dim=1).cpu().numpy()

        # read fovx from dust3r init
        pkl_path = Path(dirpath, self.dirname, mast3r_expname, "global_params.pkl")
        with open(pkl_path.as_posix(), "rb") as pkl_file:
            data = pickle.load(pkl_file)
        self._fovx = focal2fov(data["focals"][0], mast3r_img_res)

    def get_poses(self, idx):
        return self._poses[idx]

    def get_fovx(self, idx):
        return self._fovx


class MASt3RPCDReader:

    path = "./op_results"
    dirname = "mast3r_opt"
    dynamic_path = "./dynamic"
    static_path = "./static"

    skip_dynamic = False

    def __init__(
        self,
        dirpath,
        mast3r_expname,
        mode=None,
        downsample_ratio=0.1,
        num_limit_points=None,
        **kwargs,
    ):

        if not Path(dirpath, self.dirname, mast3r_expname, self.dynamic_path).exists():
            static_pcd_paths = Path(
                dirpath, self.dirname, mast3r_expname, self.static_path
            )
            static_pcd_file = [
                pth.as_posix() for pth in sorted(static_pcd_paths.glob("*.ply"))
            ][0]
            static_pcd = fetchPly(static_pcd_file)
            self.pcd = static_pcd
            self.skip_dynamic = True
            return

        if mode == "dynamic":
            pcd_paths = Path(dirpath, self.dirname, mast3r_expname, self.dynamic_path)
            pcd_files = [pth.as_posix() for pth in sorted(pcd_paths.glob("*.ply"))]
        elif mode == "static":
            pcd_paths = Path(dirpath, self.dirname, mast3r_expname, self.static_path)
            pcd_files = [pth.as_posix() for pth in sorted(pcd_paths.glob("*.ply"))]
        else:
            pcd_paths = Path(dirpath, self.dirname, mast3r_expname, self.path)
            pcd_files = [pth.as_posix() for pth in sorted(pcd_paths.glob("*.ply"))]
        pcds = [fetchPly(pcd_file) for pcd_file in pcd_files]

        json_file = Path(dirpath, "train_transforms.json")
        with open(json_file) as json_file:
            contents = json.load(json_file)
            times = [frame["time"] for frame in contents["frames"]]

        for idx, pcd in enumerate(pcds):
            time = times[idx]
            num_points = len(pcd.points)
            pcd.time = np.ones(num_points) * time

        merged_pcds = merge_pcds(pcds)

        if num_limit_points is not None:
            print(f"override downsample_ratio with num_vertices: {num_limit_points}")
            downsample_ratio = min(num_limit_points / len(merged_pcds.points), 1.0)

        self.pcd = uniform_sample(merged_pcds, downsample_ratio)

    def __call__(self):
        return self.pcd, self.skip_dynamic
