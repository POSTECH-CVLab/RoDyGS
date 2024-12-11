# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import json
from PIL import Image
from typing import Iterable, Dict, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from plyfile import PlyData

from src.utils.configs import instantiate_from_config
from .utils import CameraInterface, PILtoTorch, FixedCameraTorch
from src.utils.graphic_utils import focal2fov, fov2focal, getWorld2View2
from src.utils.point_utils import BasicPointCloud
from src.utils.graphic_utils import matrix_to_quaternion, quaternion_to_matrix


class GSDataModule:
    # Read train, test cameras from datadir
    def __init__(
        self,
        dirpath: str,
        train_dset_config: DictConfig,
        test_dset_config: DictConfig,
        train_dloader_config: DictConfig,
        test_dloader_config: DictConfig,
        train_pcd_reader_config: DictConfig,
        train_pose_reader_config: Union[DictConfig] = None,
        normalize_cams: bool = False,
        train_transform_fname: str = "train_transforms.json",
        test_transform_fname: str = "test_transforms.json",
        ckpt_path: str = None,
    ):
        # instantiate dataset from dataset configs
        self.train_dset = instantiate_from_config(
            train_dset_config,
            dirpath=dirpath,
            fname=train_transform_fname,
            ckpt_path=ckpt_path,
        ).cuda()
        test_dset = instantiate_from_config(
            test_dset_config,
            dirpath=dirpath,
            fname=test_transform_fname,
            ckpt_path=ckpt_path,
        ).cuda()

        # nerf_normalization is used to allow adaptable learning rate
        self._nerf_normalization = self.train_dset.getNerfppNorm()

        # instantiate dataloaders from dataloader configs
        self._train_dloader = instantiate_from_config(
            train_dloader_config, dataset=self.train_dset
        )
        self._test_dloader = instantiate_from_config(
            test_dloader_config, dataset=test_dset
        )

        self._pcd, self.skip_dynamic = instantiate_from_config(
            train_pcd_reader_config,
            dirpath=dirpath,
            nerf_normalization=self._nerf_normalization,
        )()

        if train_pose_reader_config:
            gt_train_dset = instantiate_from_config(
                train_pose_reader_config, dirpath=dirpath, fname="train_transforms.json"
            ).cuda()
            self._gt_train_dset = gt_train_dset

        if normalize_cams:
            self.train_dset.normalize(self._nerf_normalization)
            test_dset.normalize(self._nerf_normalization)
            self._pcd = self.normalize_pcd(self._pcd, self._nerf_normalization)
            self._nerf_normalization = self.train_dset.getNerfppNorm()

    def get_train_dset(self):
        return self.train_dset

    def register_frame(self) -> None:
        if self.train_dloader.is_incremental:
            self.train_dloader.register_frame()

    def get_init_pcd(self) -> PlyData:
        return self._pcd

    def get_normalization(self) -> Dict:
        return self._nerf_normalization

    def get_train_dloader(self) -> Iterable:
        return self._train_dloader

    def get_test_dloader(self) -> Iterable:
        return self._test_dloader

    def get_gt_train_poses(self) -> torch.Tensor:
        return self._gt_train_dset.get_poses()

    def get_train_poses(self) -> torch.Tensor:
        return self._train_dloader.dataset.get_poses()

    def normalize_pcd(self, pcd, nerf_normalization):
        translate, radius = (
            nerf_normalization["translate"],
            nerf_normalization["radius"],
        )

        if isinstance(translate, torch.Tensor):
            translate = translate.detach().cpu().numpy()

        pcd_points_norm = (pcd.points + translate[None, :]) / radius
        return BasicPointCloud(pcd_points_norm, pcd.colors, pcd.normals, pcd.time)


class DataReader(nn.Module):
    # Converts PIL images to image tensors
    def __init__(
        self,
        dirpath: str,
        fname: str,
        camera_config: DictConfig,
        pose_reader: Union[DictConfig] = None,
        depth_reader: Union[DictConfig] = None,
        normal_reader: Union[DictConfig] = None,
        motion_mask_reader: Union[DictConfig] = None,
        ckpt_path: str = None,
    ):

        super(DataReader, self).__init__()
        self.camera_config = camera_config

        cam_infos = []

        pose_reader_object = instantiate_from_config(
            pose_reader, dirpath=dirpath, fname=fname, ckpt_path=ckpt_path
        )

        with open(os.path.join(dirpath, fname)) as json_file:
            contents = json.load(json_file)

            frames = contents["frames"]
            for idx, frame in enumerate(frames):
                cam_name = os.path.join(dirpath, frame["file_path"])
                base_name = os.path.basename(frame["file_path"])
                time = frame["time"]

                c2w = pose_reader_object.get_poses(idx)
                fovx = pose_reader_object.get_fovx(idx)

                if depth_reader is None:
                    depth = None
                else:
                    depth = instantiate_from_config(depth_reader)(dirpath, base_name)

                if normal_reader is None:
                    normal = None
                else:
                    normal = instantiate_from_config(normal_reader)(dirpath, base_name)

                if motion_mask_reader is None:
                    motion_mask = None
                else:
                    motion_mask = instantiate_from_config(motion_mask_reader)(
                        dirpath, base_name
                    )

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = w2c[:3, :3].astype(np.float32)
                T = w2c[:3, 3].astype(np.float32)

                image_name = Path(cam_name).stem
                image = Image.open(cam_name)

                im_data = np.array(image.convert("RGBA"))

                bg = np.array([0, 0, 0])

                norm_data = im_data / 255.0
                arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                    1 - norm_data[:, :, 3:4]
                )
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
                W, H = image.size[0], image.size[1]
                image = PILtoTorch(image)

                fovy = focal2fov(fov2focal(fovx, W), H)

                loaded_mask = None
                if image.shape[0] == 4:
                    loaded_mask = image[3:4, ...]

                image = image.clamp(0.0, 1.0)
                if loaded_mask is not None:
                    image = image * loaded_mask

                cam_infos.append(
                    instantiate_from_config(
                        camera_config,
                        R=R,
                        T=T,
                        FoVx=fovx,
                        FoVy=fovy,
                        image=image,
                        image_name=image_name,
                        time=time,
                        depth=depth,
                        normal=normal,
                        motion_mask=motion_mask,
                        cam_idx=idx,
                    )
                )

        self.cam_infos = nn.ModuleList(cam_infos)

    def __getitem__(self, index) -> CameraInterface:
        return self.cam_infos[index]

    def __len__(self):
        return len(self.cam_infos)

    def get_poses(self):
        poses = []
        for cam in self.cam_infos:
            poses.append(torch.inverse(cam.world_view_transform).cuda())
        return torch.stack(poses)

    def get_times(self):
        return [cam.time for cam in self.cam_infos]

    def get_w2cs(self):
        w2cs = []
        for cam in self.cam_infos:
            w2cs.append(cam.world_view_transform.cuda())
        return torch.stack(w2cs)

    def getNerfppNorm(self):
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        cam_centers = []

        for cam in self.cam_infos:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1

        translate = -center

        return {"translate": translate, "radius": radius}

    def normalize(self, nerf_normalization):

        translate, radius = (
            nerf_normalization["translate"],
            nerf_normalization["radius"],
        )

        if isinstance(translate, torch.Tensor):
            translate = translate.detach().cpu().numpy()

        cam_infos = []

        for cam in self.cam_infos:

            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = cam.R
            w2c[:3, 3] = cam.T
            c2w = np.linalg.inv(w2c)
            c2w[:3, 3] = (c2w[:3, 3] + translate) / radius
            w2c = np.linalg.inv(c2w)

            cam_depth = cam.depth
            if cam_depth is not None:
                cam_depth = cam_depth / radius

            cam_infos.append(
                instantiate_from_config(
                    self.camera_config,
                    R=cam.R,
                    T=w2c[:3, 3],
                    FoVx=cam.FoVx,
                    FoVy=cam.FoVy,
                    image=cam.original_image,
                    image_name=cam.image_name,
                    time=cam.time,
                    depth=cam_depth,
                    normal=cam.normal,
                    motion_mask=cam.motion_mask,
                    cam_idx=cam.cam_idx,
                ).cuda()
            )

        self.cam_infos = nn.ModuleList(cam_infos)


class LazyDataReader(DataReader):
    # Converts PIL images to image tensors
    def __init__(
        self,
        dirpath: str,
        fname: str,
        pose_reader: Union[DictConfig] = None,
        depth_reader: Union[DictConfig] = None,
        normal_reader: Union[DictConfig] = None,
        motion_mask_reader: Union[DictConfig] = None,
        max_depth_reader: Union[DictConfig] = None,
        ckpt_path: str = None,
        **kwargs,
    ):
        super(DataReader, self).__init__()
        cam_infos, R_c2ws_quat, T_c2ws = [], [], []

        pose_reader_object = instantiate_from_config(
            pose_reader, dirpath=dirpath, fname=fname, ckpt_path=ckpt_path
        )

        R_c2ws_quat, T_c2ws = [], []

        with open(os.path.join(dirpath, fname)) as json_file:
            contents = json.load(json_file)

            frames = contents["frames"]
            for idx, frame in enumerate(frames):
                cam_name = os.path.join(dirpath, frame["file_path"])
                base_name = os.path.basename(frame["file_path"])
                time = frame["time"]

                c2w = pose_reader_object.get_poses(idx)
                fovx = pose_reader_object.get_fovx(idx)

                if depth_reader is None:
                    depth, max_depth = None, None
                else:
                    depth = instantiate_from_config(depth_reader)(dirpath, base_name)
                    if type(depth) == tuple:
                        depth, max_depth = depth
                    else:
                        max_depth = None

                if normal_reader is None:
                    normal = None
                else:
                    normal = instantiate_from_config(normal_reader)(dirpath, base_name)

                if motion_mask_reader is None:
                    motion_mask = None
                else:
                    motion_mask = instantiate_from_config(motion_mask_reader)(
                        dirpath, base_name
                    )

                # get the world-to-camera transform and set R, T
                c2w = torch.from_numpy(c2w).float()

                image_name = Path(cam_name).stem
                image = Image.open(cam_name)

                im_data = np.array(image.convert("RGBA"))

                bg = np.array([0, 0, 0])

                norm_data = im_data / 255.0
                arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (
                    1 - norm_data[:, :, 3:4]
                )
                image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
                W, H = image.size[0], image.size[1]
                image = PILtoTorch(image)

                fovy = focal2fov(fov2focal(fovx, W), H)

                loaded_mask = None
                if image.shape[0] == 4:
                    loaded_mask = image[3:4, ...]

                image = image.clamp(0.0, 1.0)
                if loaded_mask is not None:
                    image = image * loaded_mask

                cam_infos.append(
                    {
                        "FoVx": fovx,
                        "FoVy": fovy,
                        "image": image,
                        "image_name": image_name,
                        "time": time,
                        "depth": depth,
                        "max_depth": max_depth,
                        "normal": normal,
                        "motion_mask": motion_mask,
                        "cam_idx": idx,
                    }
                )
                R_c2ws_quat.append(matrix_to_quaternion(c2w[:3, :3]))
                T_c2ws.append(c2w[:3, 3])

        self.cam_infos = cam_infos
        self.register_parameter(
            "R_c2ws_quat", nn.Parameter(torch.stack(R_c2ws_quat), requires_grad=True)
        )
        self.register_parameter(
            "T_c2ws", nn.Parameter(torch.stack(T_c2ws), requires_grad=True)
        )

    def __getitem__(self, index) -> CameraInterface:
        cam_info = self.cam_infos[index]

        return FixedCameraTorch(
            R_quat_c2w=self.R_c2ws_quat[index],
            T_c2w=self.T_c2ws[index],
            **cam_info,
        )

    def __len__(self):
        return len(self.cam_infos)

    def get_times(self):
        return [cam_info["time"] for cam_info in self.cam_infos]

    def get_poses(self):
        R_c2w = quaternion_to_matrix(self.R_c2ws_quat)
        T_c2w = self.T_c2ws
        ret = torch.zeros(R_c2w.shape[0], 4, 4).type_as(R_c2w)
        ret[:, :3, :3] = R_c2w
        ret[:, :3, 3] = T_c2w
        ret[:, 3, 3] = 1
        return ret

    def get_w2cs(self):
        R_c2w = quaternion_to_matrix(self.R_c2ws_quat)
        T_c2w = self.T_c2ws
        R_w2c = R_c2w.transpose(-1, -2)
        T_w2c = -torch.einsum("bij, bj -> bi", R_w2c, T_c2w)
        ret = torch.zeros(R_c2w.shape[0], 4, 4).type_as(R_c2w)
        ret[:, :3, :3] = R_w2c
        ret[:, :3, 3] = T_w2c
        ret[:, 3, 3] = 1

        return ret

    @torch.no_grad()
    def getNerfppNorm(self):
        def get_center_and_diag(cam_centers):
            avg_cam_center = torch.mean(cam_centers, dim=0, keepdims=True)
            center = avg_cam_center
            dist = torch.norm(cam_centers - center, dim=1)
            diagonal = torch.max(dist)
            return center.flatten(), diagonal

        cam_centers = self.T_c2ws
        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1

        translate = -center

        return {"translate": translate, "radius": radius.item()}

    @torch.no_grad()
    def normalize(self, nerf_normalization):
        translate, radius = (
            nerf_normalization["translate"],
            nerf_normalization["radius"],
        )
        T_c2ws = self.T_c2ws.clone().detach()
        del self.T_c2ws
        self.register_parameter(
            "T_c2ws", nn.Parameter((T_c2ws + translate) / radius, requires_grad=True)
        )
