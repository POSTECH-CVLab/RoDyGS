# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from pathlib import Path
from typing import Optional

from omegaconf import DictConfig
import torch
from tqdm import tqdm
from src.model.rodygs_dynamic import DynRoDyGS
from src.trainer.rodygs_static import ThreeDGSTrainer
from src.trainer.utils import prune_optimizer
from src.utils.general_utils import get_expon_lr_func


class DynTrainer(ThreeDGSTrainer):

    def __init__(
        self,
        datamodule,
        logdir: Path,
        model: DynRoDyGS,
        loss_config: DictConfig,
        # GS params
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
        # DyNMF params
        deform_lr_init: float,
        deform_lr_final: float,
        deform_lr_delay_mult: float,
        deform_lr_max_steps: int,
        motion_coeff_lr: float,
        deform_warmup_steps: int,
        # logging option
        log_freq: int = 50,
        # camera optim
        camera_opt_config: Optional[DictConfig] = None,
    ):
        super(DynTrainer, self).__init__(
            datamodule=datamodule,
            logdir=logdir,
            model=model,
            loss_config=loss_config,
            num_iterations=num_iterations,
            position_lr_init=position_lr_init,
            position_lr_final=position_lr_final,
            position_lr_delay_mult=position_lr_delay_mult,
            position_lr_max_steps=position_lr_max_steps,
            feature_lr=feature_lr,
            opacity_lr=opacity_lr,
            scaling_lr=scaling_lr,
            rotation_lr=rotation_lr,
            percent_dense=percent_dense,
            opacity_reset_interval=opacity_reset_interval,
            densify_grad_threshold=densify_grad_threshold,
            densify_from_iter=densify_from_iter,
            densify_until_iter=densify_until_iter,
            densification_interval=densification_interval,
            log_freq=log_freq,
            camera_opt_config=camera_opt_config,
            deform_warmup_steps=deform_warmup_steps,
        )

        self.append_motion_optim(
            deform_lr_init=deform_lr_init,
            deform_lr_final=deform_lr_final,
            deform_lr_delay_mult=deform_lr_delay_mult,
            deform_lr_max_steps=deform_lr_max_steps,
            motion_coeff_lr=motion_coeff_lr,
        )

    def append_motion_optim(
        self,
        deform_lr_init: float,
        deform_lr_final: float,
        deform_lr_delay_mult: float,
        deform_lr_max_steps: int,
        motion_coeff_lr: float,
    ):
        # Additional parameters and learning rates
        additional_params = [
            {
                "params": list(self.model._deform_network.parameters()),
                "lr": deform_lr_init,
                "name": "deform_network",
            },
            {
                "params": [self.model._motion_coeff],
                "lr": motion_coeff_lr,
                "name": "motion_coeff",
            },
        ]

        # Append additional parameters and learning rates to the optimizer
        for additional_param in additional_params:
            self.optimizer.add_param_group(additional_param)

        self.deform_scheduler_fn = get_expon_lr_func(
            lr_init=deform_lr_init,
            lr_final=deform_lr_final,
            lr_delay_mult=deform_lr_delay_mult,
            max_steps=deform_lr_max_steps,
        )

    def train(self):
        pbar = tqdm(total=self.num_iterations)
        dloader = self.datamodule.get_train_dloader()

        times = dloader.dataset.get_times()
        times = torch.tensor(times).cuda().view(-1, 1)
        self.model._time_embeddings = self.model._deform_network.batch_embedding(times)

        print("pre-encoding time embeddings")
        print("shape : ", self.model._time_embeddings.shape)

        train_dloader_iter = iter(dloader)

        for iteration in range(1, self.num_iterations + 1):
            self.train_iteration(train_dloader_iter, iteration)

            if iteration % self.log_freq == 0:
                pbar.update(self.log_freq)

        print("\n[ITER {}] Saving Checkpoint".format(iteration))
        torch.save(
            (self.state_dict(iteration), iteration),
            self.logdir.as_posix() + "/last.ckpt",
        )

    def prune_points(self, mask):
        # the same as the original 3DGS pruner, but also prunes the motion coefficients
        valid_points_mask = ~mask
        optimizable_tensors = prune_optimizer(self.optimizer, valid_points_mask)

        self.model._xyz = optimizable_tensors["xyz"]
        self.model._features_dc = optimizable_tensors["f_dc"]
        self.model._features_rest = optimizable_tensors["f_rest"]
        self.model._opacity = optimizable_tensors["opacity"]
        self.model._scaling = optimizable_tensors["scaling"]
        self.model._rotation = optimizable_tensors["rotation"]
        self.model._motion_coeff = optimizable_tensors["motion_coeff"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.model.gaussian_to_time = self.model.gaussian_to_time[valid_points_mask]
        self.model.gaussian_to_time_ind = self.model.gaussian_to_time_ind[
            valid_points_mask
        ]

    def set_attributes_from_opt_tensors(self, optimizable_tensors):
        super(DynTrainer, self).set_attributes_from_opt_tensors(optimizable_tensors)
        self.model._motion_coeff = optimizable_tensors["motion_coeff"]

    def densify_and_clone_update_attributes(self, grads, grad_threshold, scene_extent):
        (
            updated_attributes,
            selected_pts_mask,
        ) = super().densify_and_clone_update_attributes(
            grads, grad_threshold, scene_extent
        )
        updated_attributes["motion_coeff"] = self.model._motion_coeff[selected_pts_mask]
        return updated_attributes, selected_pts_mask

    def densify_and_split_update_attributes(
        self, grads, grad_threshold, scene_extent, N
    ):
        (
            updated_attributes,
            selected_pts_mask,
        ) = super().densify_and_split_update_attributes(
            grads, grad_threshold, scene_extent, N
        )
        updated_attributes["motion_coeff"] = self.model._motion_coeff[
            selected_pts_mask
        ].repeat(N, 1, 1)
        return updated_attributes, selected_pts_mask

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""

        xyz_found = False
        deform_found = False

        for param_group in self.optimizer.param_groups:
            if xyz_found and deform_found:
                break
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_fn(iteration)
                param_group["lr"] = lr
                xyz_found = True
            elif param_group["name"] == "deform":
                lr = self.deform_scheduler_fn(iteration)
                param_group["lr"] = lr
                deform_found = True

    def state_dict(self, iteration):
        state_dict = super(DynTrainer, self).state_dict(iteration)
        state_dict["model"]["_motion_coeff"] = self.model._motion_coeff
        state_dict["model"]["_deform_network"] = self.model._deform_network.state_dict()
        state_dict["model"]["_timestep"] = self.model.gaussian_to_time
        return state_dict
