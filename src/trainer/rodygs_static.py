# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from tqdm import tqdm
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig

from src.model.rodygs_static import StaticRoDyGS
from src.utils.general_utils import get_expon_lr_func, inverse_sigmoid, build_rotation
from src.utils.configs import instantiate_from_config
from src.trainer.utils import (
    replace_tensor_to_optimizer,
    cat_tensors_to_optimizer,
    prune_optimizer,
)


class ThreeDGSTrainer:

    def __init__(
        self,
        datamodule,
        logdir: Path,
        model: StaticRoDyGS,
        loss_config: DictConfig,
        # optimization params
        num_iterations: int,
        position_lr_init: float,
        position_lr_final: float,
        position_lr_delay_mult: float,
        position_lr_max_steps: int,
        feature_lr: float,
        opacity_lr: float,
        scaling_lr: float,
        rotation_lr: float,
        percent_dense: float,
        opacity_reset_interval: int,
        densify_grad_threshold: float,
        densify_from_iter: int,
        densify_until_iter: int,
        densification_interval: int,
        deform_warmup_steps: int = -1,
        # logging option
        log_freq: int = 50,
        # camera optim
        camera_opt_config: Optional[DictConfig] = None,
    ):
        self.datamodule = datamodule
        self.model = model
        self.logdir = logdir
        self.validation_dir = logdir.joinpath("validation")
        self.validation_dir.mkdir(exist_ok=True)

        self.num_iterations = num_iterations
        self.opacity_reset_interval = opacity_reset_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densification_interval = densification_interval
        self.percent_dense = percent_dense
        self.log_freq = log_freq
        self.is_optimizable_cam = not camera_opt_config is None
        self.deform_warmup_steps = deform_warmup_steps

        # create gaussians from loaded pcd.
        self.spatial_lr_scale = datamodule.get_normalization()["radius"]
        self.max_radii2D, self.xyz_gradient_accum, self.denom = (
            self.model.create_from_pcd(
                self.datamodule.get_init_pcd(), self.spatial_lr_scale
            )
        )

        # optimizer setup
        self.optim_setup(
            position_lr_init=position_lr_init,
            position_lr_final=position_lr_final,
            position_lr_delay_mult=position_lr_delay_mult,
            position_lr_max_steps=position_lr_max_steps,
            feature_lr=feature_lr,
            opacity_lr=opacity_lr,
            scaling_lr=scaling_lr,
            rotation_lr=rotation_lr,
        )

        if self.is_optimizable_cam:
            self.cam_optimizer = instantiate_from_config(
                camera_opt_config,
                dataset=datamodule.get_train_dloader().dataset,
                spatial_lr_scale=self.spatial_lr_scale,
            )

        self.bg_color = torch.rand((3), device="cuda")

        self.loss = instantiate_from_config(loss_config)

    def optim_setup(
        self,
        position_lr_init: float,
        position_lr_final: float,
        position_lr_delay_mult: float,
        position_lr_max_steps: int,
        feature_lr: float,
        opacity_lr: float,
        scaling_lr: float,
        rotation_lr: float,
    ):

        l = [
            {
                "params": [self.model._xyz],
                "lr": position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {"params": [self.model._features_dc], "lr": feature_lr, "name": "f_dc"},
            {
                "params": [self.model._features_rest],
                "lr": feature_lr / 20.0,
                "name": "f_rest",
            },
            {"params": [self.model._opacity], "lr": opacity_lr, "name": "opacity"},
            {"params": [self.model._scaling], "lr": scaling_lr, "name": "scaling"},
            {"params": [self.model._rotation], "lr": rotation_lr, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_fn = get_expon_lr_func(
            lr_init=position_lr_init * self.spatial_lr_scale,
            lr_final=position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=position_lr_delay_mult,
            max_steps=position_lr_max_steps,
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_fn(iteration)
                param_group["lr"] = lr
                return lr

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(
                self.model.get_opacity, torch.ones_like(self.model.get_opacity) * 0.01
            )
        )
        optimizable_tensors = replace_tensor_to_optimizer(
            self.optimizer, opacities_new, "opacity"
        )
        self.model._opacity = optimizable_tensors["opacity"]

    def set_attributes_from_opt_tensors(self, optimizable_tensors):
        self.model._xyz = optimizable_tensors["xyz"]
        self.model._features_dc = optimizable_tensors["f_dc"]
        self.model._features_rest = optimizable_tensors["f_rest"]
        self.model._opacity = optimizable_tensors["opacity"]
        self.model._scaling = optimizable_tensors["scaling"]
        self.model._rotation = optimizable_tensors["rotation"]

    def densification_postfix(self, updated_attributes):
        optimizable_tensors = cat_tensors_to_optimizer(
            self.optimizer, updated_attributes
        )
        self.set_attributes_from_opt_tensors(optimizable_tensors)

        self.xyz_gradient_accum = torch.zeros(
            (self.model.get_xyz.shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.model.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.model.get_xyz.shape[0]), device="cuda")

    def densify_and_split_update_attributes(
        self, grads, grad_threshold, scene_extent, N=2
    ):
        n_init_points = self.model.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.model.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )
        stds = self.model.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.model._rotation[selected_pts_mask]).repeat(N, 1, 1)

        scale_attributes = torch.log(
            self.model.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        if self.model.isotropic:
            scale_attributes = scale_attributes[:, [0]]

        return {
            "xyz": torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            + self.model.get_xyz[selected_pts_mask].repeat(N, 1),
            "f_dc": self.model._features_dc[selected_pts_mask].repeat(N, 1, 1),
            "f_rest": self.model._features_rest[selected_pts_mask].repeat(N, 1, 1),
            "opacity": self.model._opacity[selected_pts_mask].repeat(N, 1),
            "scaling": scale_attributes,
            "rotation": self.model._rotation[selected_pts_mask].repeat(N, 1),
        }, selected_pts_mask

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        # Extract points that satisfy the gradient condition
        updated_attributes, selected_pts_mask = (
            self.densify_and_split_update_attributes(
                grads, grad_threshold, scene_extent, N
            )
        )

        self.densification_postfix(updated_attributes)

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.model.gaussian_to_time = torch.cat(
            (
                self.model.gaussian_to_time,
                *[self.model.gaussian_to_time[selected_pts_mask] for _ in range(N)],
            )
        )
        self.model.gaussian_to_time_ind = torch.cat(
            (
                self.model.gaussian_to_time_ind,
                *[self.model.gaussian_to_time_ind[selected_pts_mask] for _ in range(N)],
            )
        )
        assert self.model.gaussian_to_time.shape[0] == self.model.get_xyz.shape[0]
        self.prune_points(prune_filter)

    def densify_and_clone_update_attributes(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        max_values = torch.max(self.model.get_scaling, dim=1).values
        selected_pts_mask = torch.logical_and(
            selected_pts_mask, max_values <= self.percent_dense * scene_extent
        )
        return {
            "xyz": self.model._xyz[selected_pts_mask],
            "f_dc": self.model._features_dc[selected_pts_mask],
            "f_rest": self.model._features_rest[selected_pts_mask],
            "opacity": self.model._opacity[selected_pts_mask],
            "scaling": self.model._scaling[selected_pts_mask],
            "rotation": self.model._rotation[selected_pts_mask],
        }, selected_pts_mask

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        updated_attributes, selected_pts_mask = (
            self.densify_and_clone_update_attributes(
                grads, grad_threshold, scene_extent
            )
        )
        self.model.gaussian_to_time = torch.cat(
            (
                self.model.gaussian_to_time,
                self.model.gaussian_to_time[selected_pts_mask],
            )
        )
        self.model.gaussian_to_time_ind = torch.cat(
            (
                self.model.gaussian_to_time_ind,
                self.model.gaussian_to_time_ind[selected_pts_mask],
            )
        )
        self.densification_postfix(updated_attributes)
        assert self.model.gaussian_to_time.shape[0] == self.model.get_xyz.shape[0]

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.model.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.model.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )

        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def prune_points(self, mask):

        valid_points_mask = ~mask
        optimizable_tensors = prune_optimizer(self.optimizer, valid_points_mask)
        self.set_attributes_from_opt_tensors(optimizable_tensors)

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.model.gaussian_to_time = self.model.gaussian_to_time[valid_points_mask]
        self.model.gaussian_to_time_ind = self.model.gaussian_to_time_ind[
            valid_points_mask
        ]

    def add_densification_stats(self, update_filter, grad_densification):
        self.xyz_gradient_accum[update_filter] += grad_densification[update_filter]
        self.denom[update_filter] += 1

    def state_dict(self, iteration):
        state_dict = {
            "iteration": iteration,
            "active_sh_degree": self.model.active_sh_degree,
            "model": {
                "_xyz": self.model._xyz,
                "_features_dc": self.model._features_dc,
                "_features_rest": self.model._features_rest,
                "_scaling": self.model._scaling,
                "_rotation": self.model._rotation,
                "_opacity": self.model._opacity,
            },
            "optim": {
                "max_radii2D": self.max_radii2D,
                "xyz_gradient_accum": self.xyz_gradient_accum,
                "denom": self.denom,
                "optimizer": self.optimizer.state_dict(),
            },
            "spatial_lr_scale": self.spatial_lr_scale,
        }

        if self.is_optimizable_cam:
            state_dict["camera"] = (
                self.datamodule.get_train_dloader().dataset.state_dict()
            )

        return state_dict

    def train(self):
        pbar = tqdm(total=self.num_iterations)
        train_dloader_iter = iter(self.datamodule.get_train_dloader())

        for iteration in range(1, self.num_iterations + 1):
            self.train_iteration(train_dloader_iter, iteration)

            if iteration % self.log_freq == 0:
                pbar.update(self.log_freq)

        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save(
            (self.state_dict(iteration), iteration),
            self.logdir.as_posix() + "/last.ckpt",
        )

    def train_iteration(self, train_dloader_iter, iteration):

        camera = next(train_dloader_iter).cuda()
        self.update_learning_rate(iteration)
        if self.is_optimizable_cam:
            self.cam_optimizer.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            self.model.oneupSHdegree()

        render_pkg = self.model.render(
            viewpoint_camera=camera,
            bg_color=torch.zeros(3).cuda(),
            enable_sh_grad=self.is_optimizable_cam,
            enable_cov_grad=self.is_optimizable_cam,
            use_deform=iteration > self.deform_warmup_steps,
        )

        image = render_pkg["rendered_image"]
        depth = render_pkg["rendered_depth"]
        normal = render_pkg["rendered_normal"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        pred_translation = render_pkg["translation"]
        pred_rotation = render_pkg["rotation"]

        gt_image = camera.original_image
        gt_depth = camera.depth
        gt_normal = camera.normal

        # rodynrf motion mask
        gt_motion_mask = camera.motion_mask

        loss, loss_dict = self.loss(
            iteration=iteration,
            pred_img=image,
            gt_img=gt_image,
            pred_depth=depth,
            gt_depth=gt_depth,
            pred_normal=normal,
            gt_normal=gt_normal,
            model=self.model,
            pred_translation=pred_translation,
            pred_rotation=pred_rotation,
            datamodule=self.datamodule,
            motion_mask=gt_motion_mask,
            camera=camera,
        )
        loss.backward()

        with torch.no_grad():
            # Densification
            if iteration < self.densify_until_iter:

                max_radii_update = torch.max(
                    self.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                self.max_radii2D[visibility_filter] = max_radii_update
                grad_densification = torch.norm(
                    viewspace_point_tensor.grad[:, :2], dim=-1, keepdim=True
                )
                self.add_densification_stats(visibility_filter, grad_densification)

                if (
                    self.densification_interval != 0
                    and iteration > self.densify_from_iter
                    and iteration % self.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > self.opacity_reset_interval else None
                    )
                    self.densify_and_prune(
                        self.densify_grad_threshold,
                        0.005,
                        self.spatial_lr_scale,
                        size_threshold,
                    )

                if (
                    self.opacity_reset_interval != 0
                    and iteration % self.opacity_reset_interval == 0
                ):
                    self.reset_opacity()

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if self.is_optimizable_cam:
                self.cam_optimizer.step()
                self.cam_optimizer.zero_grad(set_to_none=True)
