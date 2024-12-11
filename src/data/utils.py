# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import math

import numpy as np
import torch
import torch.nn as nn
from plyfile import PlyData

from src.utils.graphic_utils import (
    getProjectionMatrix,
    getWorld2View2,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from src.utils.point_utils import BasicPointCloud


class CameraInterface(nn.Module):

    znear = 0.01
    zfar = 100.0
    trans = 0.0
    scale = 1.0

    def __init__(
        self, R, T, image, image_name, time, depth, normal, motion_mask, cam_idx
    ):
        super(CameraInterface, self).__init__()

        self.R = R
        self.T = T
        self.image_name = image_name
        self.image_width = image.shape[2]
        self.image_height = image.shape[1]
        self.cam_idx = cam_idx
        self.register_buffer("time", torch.tensor(time))
        self.register_buffer("original_image", image)

        if depth is not None:
            self.register_buffer("depth", depth)
        else:
            self.depth = None

        if normal is not None:
            self.register_buffer("normal", normal)
        else:
            self.normal = None

        if motion_mask is not None:
            self.register_buffer("motion_mask", motion_mask)
        else:
            self.motion_mask = None


class FixedCamera(CameraInterface):
    def __init__(
        self,
        R,
        T,
        FoVx,
        FoVy,
        image,
        image_name,
        time,
        depth,
        normal,
        motion_mask,
        cam_idx,
    ):

        super(FixedCamera, self).__init__(
            R=R,
            T=T,
            image=image,
            image_name=image_name,
            time=time,
            depth=depth,
            normal=normal,
            motion_mask=motion_mask,
            cam_idx=cam_idx,
        )
        self.FoVx = FoVx
        self.FoVy = FoVy

        world_view_transform = torch.from_numpy(
            getWorld2View2(R, T, self.trans, self.scale)
        )
        projection_matrix = getProjectionMatrix(
            self.znear, self.zfar, self.FoVx, self.FoVy
        )

        self.register_buffer("world_view_transform", world_view_transform)
        self.register_buffer("projection_matrix", projection_matrix)


class FixedCameraTorch:

    znear = 0.01
    zfar = 100.0
    trans = 0.0
    scale = 1.0

    def __init__(
        self,
        R_quat_c2w,
        T_c2w,
        FoVx,
        FoVy,
        image,
        image_name,
        time,
        depth,
        max_depth,
        normal,
        motion_mask,
        cam_idx,
    ):

        self.image_name = image_name
        self.image_width = image.shape[2]
        self.image_height = image.shape[1]
        self.time = torch.tensor(time)
        self.original_image = image

        self.depth = depth
        self.max_depth = max_depth
        self.normal = normal
        self.motion_mask = motion_mask

        self.FoVx = FoVx
        self.FoVy = FoVy
        self.R_quat_c2w = R_quat_c2w
        self.T_c2w = T_c2w
        projection_matrix = getProjectionMatrix(
            self.znear, self.zfar, self.FoVx, self.FoVy
        )
        self.projection_matrix = projection_matrix
        self.cam_idx = cam_idx

    def cuda(self):
        self.original_image = self.original_image.cuda()
        if self.depth is not None:
            self.depth = self.depth.cuda()
        if self.normal is not None:
            self.normal = self.normal.cuda()
        if self.motion_mask is not None:
            self.motion_mask = self.motion_mask.cuda()
        self.time = self.time.cuda()
        self.projection_matrix = self.projection_matrix.cuda()
        return self

    @property
    def world_view_transform(self):
        R_c2w = quaternion_to_matrix(self.R_quat_c2w)
        T_c2w = (self.T_c2w + self.trans) * self.scale
        R_w2c = R_c2w.transpose(0, 1)
        T_w2c = -torch.einsum("ij, j -> i", R_w2c, T_c2w)
        ret = torch.eye(4).type_as(R_c2w)
        ret[:3, :3] = R_w2c
        ret[:3, 3] = T_w2c
        return ret


class LearnableCamera(CameraInterface):

    def __init__(
        self,
        R,
        T,
        FoVx,
        FoVy,
        image,
        image_name,
        time,
        depth,
        normal,
        motion_mask,
        cam_idx,
    ):
        super(LearnableCamera, self).__init__(
            R=R,
            T=T,
            image=image,
            image_name=image_name,
            time=time,
            depth=depth,
            normal=normal,
            motion_mask=motion_mask,
            cam_idx=cam_idx,
        )

        self.image_name = image_name
        self.image_width = image.shape[2]
        self.image_height = image.shape[1]

        R_tr = np.transpose(R, (1, 0))
        R_c2w = torch.from_numpy(R_tr)
        T_c2w = torch.from_numpy(-np.einsum("ij, j", R_tr, T))

        R_c2w_quat = matrix_to_quaternion(R_c2w)

        self.register_parameter(
            "R_c2w_quat", nn.Parameter(R_c2w_quat, requires_grad=True)
        )
        self.register_parameter("T_c2w", nn.Parameter(T_c2w, requires_grad=True))
        self.register_parameter(
            "FoVx",
            nn.Parameter(torch.tensor(FoVx).clone().detach(), requires_grad=False),
        )
        self.register_parameter(
            "FoVy",
            nn.Parameter(torch.tensor(FoVy).clone().detach(), requires_grad=False),
        )

    @property
    def world_view_transform(self):
        R_c2w = quaternion_to_matrix(self.R_c2w_quat)
        T_c2w = (self.T_c2w + self.trans) * self.scale
        ret = torch.eye(4).type_as(R_c2w)
        rot_transpose = R_c2w.transpose(0, 1)
        ret[:3, :3] = rot_transpose
        ret[:3, 3] = -torch.einsum("ij, j -> i", rot_transpose, T_c2w)
        return ret

    @property
    def projection_matrix(self):
        tanHalfFovY = math.tan((self.FoVy / 2))
        tanHalfFovX = math.tan((self.FoVx / 2))

        top = tanHalfFovY * self.znear
        bottom = -top
        right = tanHalfFovX * self.znear
        left = -right

        P = torch.zeros(4, 4).type_as(self.R_c2w_quat)

        z_sign = 1.0

        P[0, 0] = 2.0 * self.znear / (right - left)
        P[1, 1] = 2.0 * self.znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * self.zfar / (self.zfar - self.znear)
        P[2, 3] = -(self.zfar * self.znear) / (self.zfar - self.znear)

        return P


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    if "nx" in vertices:
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    else:
        normals = np.zeros_like(positions)
    if "time" in vertices:
        timestamp = vertices["time"][:, None]
    else:
        timestamp = None
    return BasicPointCloud(
        points=positions, colors=colors, normals=normals, time=timestamp
    )


def PILtoTorch(pil_image):
    image_tensor = torch.from_numpy(np.array(pil_image)) / 255.0
    if len(image_tensor.shape) == 3:
        return image_tensor.permute(2, 0, 1)
    else:
        return image_tensor.unsqueeze(dim=-1).permute(2, 0, 1)
