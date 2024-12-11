# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from simple_knn._C import distCUDA2

from diff_gauss_pose import GaussianRasterizationSettings, GaussianRasterizer
from src.utils.general_utils import inverse_sigmoid
from src.utils.sh_utils import RGB2SH
from src.utils.point_utils import BasicPointCloud
from src.utils.general_utils import strip_symmetric, build_scaling_rotation


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm


class StaticRoDyGS:

    def __init__(self, sh_degree: int = 0, isotropic: bool = False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0
        self.isotropic = isotropic

        self.timetokey = lambda time: int(torch.trunc(torch.tensor(time) * 1000).item())
        self.tokeylist = lambda time_list: [self.timetokey(time) for time in time_list]
        self.real_times = None
        self.unique_times = None

        print("------------------------------------")
        print("GaussianSplatting initialised to isotropic : ", isotropic)
        print("Max SH degree : ", sh_degree)
        print("------------------------------------")

    def sync_gaussian_to_time_ind(
        self,
    ):
        # sample unique time steps from self._timestep
        if self.real_times is None:
            self.real_times, _ = torch.sort(torch.unique(self.gaussian_to_time))
            self.unique_times = list(
                set(self.tokeylist(self.gaussian_to_time.tolist()))
            )

            # sorting time steps and inverse mapping
            self.unique_times = sorted(self.unique_times)
            self.time_to_ind = {time: ind for ind, time in enumerate(self.unique_times)}
        # get gaussian to time index
        # each gaussian has a special time step(index)
        # tensor

        self.gaussian_to_time_ind = torch.tensor(
            [self.time_to_ind[self.timetokey(time)] for time in self.gaussian_to_time]
        ).cuda()  # [num_gaussians], gaussian -> time ind

    @property
    def get_scaling(self):
        scaling = torch.exp(self._scaling)
        if self.isotropic:
            scaling = scaling.repeat(1, 3)
        return scaling

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return build_covariance_from_scaling_rotation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(not self.isotropic))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        if pcd.time is not None:
            self.gaussian_to_time = torch.from_numpy(pcd.time).float().cuda()
        else:
            # if there is no pcd time info
            # set all time info of all gaussians to "1"
            self.gaussian_to_time = (
                torch.from_numpy(np.full(len(pcd.points), 1)).float().cuda()
            )
        self.sync_gaussian_to_time_ind()

        return max_radii2D, xyz_gradient_accum, denom

    def create_from_state_dict(self, state_dict, spatial_lr_scale: float):
        self.active_sh_degree = state_dict["active_sh_degree"]
        self.spatial_lr_scale = spatial_lr_scale
        self._xyz = nn.Parameter(state_dict["model"]["_xyz"].cuda())
        self._features_dc = nn.Parameter(state_dict["model"]["_features_dc"].cuda())
        self._features_rest = nn.Parameter(state_dict["model"]["_features_rest"].cuda())
        self._scaling = nn.Parameter(state_dict["model"]["_scaling"].cuda())
        self._rotation = nn.Parameter(state_dict["model"]["_rotation"].cuda())
        self._opacity = nn.Parameter(state_dict["model"]["_opacity"].cuda())

    def render(
        self,
        viewpoint_camera,
        bg_color: torch.Tensor,
        scaling_modifier=1.0,
        override_color=None,
        enable_sh_grad=False,
        enable_cov_grad=False,
        translation: Union[torch.Tensor, float] = 0.0,
        rotation: Union[torch.Tensor, float] = 0.0,
        **kwargs,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.get_xyz,
                dtype=self.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            projmatrix=viewpoint_camera.projection_matrix.transpose(
                0, 1
            ),  # glm storage
            sh_degree=self.active_sh_degree,
            prefiltered=False,
            debug=False,
            enable_cov_grad=enable_sh_grad,
            enable_sh_grad=enable_cov_grad,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.get_xyz + translation
        means2D = screenspace_points
        opacity = self.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        cov3D_precomp = None
        scales = self.get_scaling
        rotations = (
            self.get_rotation + rotation if not self.isotropic else self.get_rotation
        )

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            shs = self.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        (
            rendered_image,
            rendered_depth,
            rendered_normal,
            rendered_alpha,
            radii,
            extra,
        ) = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3Ds_precomp=cov3D_precomp,
            viewmatrix=viewpoint_camera.world_view_transform.transpose(
                0, 1
            ),  # glm storage
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "rendered_image": rendered_image,
            "rendered_depth": rendered_depth,
            "rendered_normal": rendered_normal,
            "rendered_alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "extra": extra,
            "translation": translation,
            "rotation": rotation,
        }
