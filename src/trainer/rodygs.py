# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch

from pathlib import Path
from tqdm import tqdm

from src.model.rodygs_static import StaticRoDyGS
from src.model.rodygs_dynamic import DynRoDyGS

from src.data.datamodule import GSDataModule
from src.utils.configs import instantiate_from_config
from src.trainer.renderer import render


class RoDyGSTrainer:

    def __init__(
        self,
        static,
        dynamic,
        static_datamodule: GSDataModule,
        dynamic_datamodule: GSDataModule,
        logdir: Path,
        static_model: StaticRoDyGS,
        dynamic_model: DynRoDyGS,
        # SH degree params
        sh_up_start_iteration: int = 0,
        sh_up_period: int = 1000,
        # logging option
        log_freq: int = 50,
    ):
        self.sh_up_start_iteration = sh_up_start_iteration
        self.sh_up_period = sh_up_period

        self.log_freq = log_freq

        self.skip_dynamic = False

        print("Init Static GS trainer")
        self.threeDGSTrainer = instantiate_from_config(
            static, datamodule=static_datamodule, model=static_model, logdir=logdir
        )
        self.static_gs = self.threeDGSTrainer.model

        if static_datamodule.skip_dynamic:
            print("Skip init Dynamic GS trainer!")
            self.skip_dynamic = True

        if not self.skip_dynamic:
            print("Init Dynamic GS trainer")
            self.dynMFTrainer = instantiate_from_config(
                dynamic,
                datamodule=dynamic_datamodule,
                model=dynamic_model,
                logdir=logdir,
            )
            self.dynamic_gs = self.dynMFTrainer.model

    def get_GS_properties(self, viewpoint_camera, use_deform=True):
        dyn_gs_translation, dyn_gs_rotation = (
            self.dynamic_gs.get_gaussian_deformation(viewpoint_camera.time)
            if use_deform
            else (0, 0)
        )

        # rotation for anisotropic GS
        if self.dynamic_gs.isotropic:
            dynamic_gs_rotation = self.dynamic_gs.get_rotation
        else:
            dynamic_gs_rotation = self.dynamic_gs.get_rotation + dyn_gs_rotation

        # assert self.static_gs.active_sh_degree != self.dynamic_gs.active_sh_degree, "Static Gaussian SH degree != Dynamic Gaussian SH degree"
        assert (
            self.static_gs.isotropic == self.dynamic_gs.isotropic
        ), "Both static and dynamic Gaussians must be isotropic or anisotropic"

        # convert get_* to variable property name
        xyz = torch.cat(
            (self.static_gs.get_xyz, self.dynamic_gs.get_xyz + dyn_gs_translation),
            dim=0,
        )
        opacity = torch.cat(
            (self.static_gs.get_opacity, self.dynamic_gs.get_opacity), dim=0
        )
        scaling = torch.cat(
            (self.static_gs.get_scaling, self.dynamic_gs.get_scaling), dim=0
        )
        rotation = torch.cat((self.static_gs.get_rotation, dynamic_gs_rotation), dim=0)
        features = torch.cat(
            (self.static_gs.get_features, self.dynamic_gs.get_features), dim=0
        )

        active_sh_degree = self.static_gs.active_sh_degree

        return (
            xyz,
            opacity,
            scaling,
            rotation,
            features,
            active_sh_degree,
            dyn_gs_translation,
            dyn_gs_rotation,
        )

    def get_static_GS_properties(self):
        xyz = self.static_gs.get_xyz
        opacity = self.static_gs.get_opacity
        scaling = self.static_gs.get_scaling
        rotation = self.static_gs.get_rotation
        features = self.static_gs.get_features
        active_sh_degree = self.static_gs.active_sh_degree
        dyn_gs_translation = None
        dyn_gs_rotation = None

        return (
            xyz,
            opacity,
            scaling,
            rotation,
            features,
            active_sh_degree,
            dyn_gs_translation,
            dyn_gs_rotation,
        )

    def train(self):
        pbar = tqdm(total=self.threeDGSTrainer.num_iterations)
        static_dloader = self.threeDGSTrainer.datamodule.get_train_dloader()
        static_train_dloader_iter = iter(static_dloader)

        if not self.skip_dynamic:
            dynamic_dloader = self.dynMFTrainer.datamodule.get_train_dloader()
            times = dynamic_dloader.dataset.get_times()
            times = torch.tensor(times).cuda().view(-1, 1)
            self.dynamic_gs._time_embeddings = (
                self.dynamic_gs._deform_network.batch_embedding(times)
            )

            print("(Dynamic) pre-encoding time embeddings")
            print("(Dynamic) shape : ", self.dynamic_gs._time_embeddings.shape)
            dynamic_train_dloader_iter = iter(dynamic_dloader)

        # For each training step:
        # 1. Optimize static GS & camera
        # 2. Update cameras in dynamic trainer to refined camera in (1.)
        # 3. Optimize dynamic GS
        for iteration in range(1, self.threeDGSTrainer.num_iterations + 1):
            self.train_iteration(
                train_dloader_iter=static_train_dloader_iter,
                iteration=iteration,
                learn_static=True,
            )

            # update dynamic camera
            if not self.skip_dynamic:
                static_dset = self.threeDGSTrainer.datamodule.get_train_dset()
                dynamic_dloader.dataset.cam_infos = static_dset.cam_infos
                with torch.no_grad():
                    dynamic_dloader.dataset.R_c2ws_quat.data = (
                        static_dset.R_c2ws_quat.data.clone().detach()
                    )
                    dynamic_dloader.dataset.T_c2ws.data = (
                        static_dset.T_c2ws.data.clone().detach()
                    )
                self.train_iteration(
                    train_dloader_iter=dynamic_train_dloader_iter,
                    iteration=iteration,
                    learn_dynamic=True,
                )

            if iteration % self.log_freq == 0:
                pbar.update(self.log_freq)

        if self.threeDGSTrainer.num_iterations < self.log_freq:
            pbar.update(self.threeDGSTrainer.num_iterations)
        print("\n[ITER {}] Saving Static Part Checkpoint".format(iteration))
        torch.save(
            (self.threeDGSTrainer.state_dict(iteration), iteration),
            self.threeDGSTrainer.logdir.as_posix() + "/static_last.ckpt",
        )
        if not self.skip_dynamic:
            print("\n[ITER {}] Saving Dynamic Part Checkpoint".format(iteration))
            torch.save(
                (self.dynMFTrainer.state_dict(iteration), iteration),
                self.dynMFTrainer.logdir.as_posix() + "/dynamic_last.ckpt",
            )

    def train_iteration(
        self,
        train_dloader_iter,
        iteration,
        learn_static=False,
        learn_dynamic=False,
        iteration_off=0,
    ):
        current_gs = self.threeDGSTrainer if learn_static else self.dynMFTrainer

        camera = next(train_dloader_iter).cuda()
        current_gs.update_learning_rate(iteration)
        if current_gs.is_optimizable_cam:
            current_gs.cam_optimizer.update_learning_rate(iteration)

        # training order: static -> dynamic
        # if SH of dynamic != static, set SH of dynamic to static's
        if (
            learn_static
            and iteration > self.sh_up_start_iteration
            and iteration % self.sh_up_period == 0
        ):
            current_gs.model.oneupSHdegree()

        if not self.skip_dynamic:
            if (
                learn_dynamic
                and self.dynamic_gs.active_sh_degree != self.static_gs.active_sh_degree
            ):
                print(
                    f"Dyn GS SH = {self.dynamic_gs.active_sh_degree} and Static GS SH = {self.static_gs.active_sh_degree}"
                )
                print("Set SH of Dyn GS to Static's")
                self.dynamic_gs.active_sh_degree = self.static_gs.active_sh_degree

            # concat static and dynamic GS
            (
                xyz,
                opacity,
                scaling,
                rotation,
                features,
                active_sh_degree,
                dyn_gs_translation,
                dyn_gs_rotation,
            ) = self.get_GS_properties(
                viewpoint_camera=camera,
                use_deform=iteration > current_gs.deform_warmup_steps,
            )
        else:
            (
                xyz,
                opacity,
                scaling,
                rotation,
                features,
                active_sh_degree,
                dyn_gs_translation,
                dyn_gs_rotation,
            ) = self.get_static_GS_properties()

        render_pkg = render(
            xyz,
            active_sh_degree,
            opacity,
            scaling,
            rotation,
            features,
            viewpoint_camera=camera,
            bg_color=torch.zeros(3).cuda(),
            enable_sh_grad=current_gs.is_optimizable_cam,
            enable_cov_grad=current_gs.is_optimizable_cam,
        )

        image = render_pkg["rendered_image"]
        depth = render_pkg["rendered_depth"]
        normal = render_pkg["rendered_normal"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = camera.original_image
        gt_depth = camera.depth
        gt_normal = camera.normal

        max_depth = camera.max_depth

        if not self.skip_dynamic:
            if (
                image.shape[2] != camera.motion_mask.shape[2]
                or image.shape[1] != camera.motion_mask.shape[1]
            ):
                assert (
                    False
                ), f"train img size ({image.shape[2]},{image.shape[1]}) != motion mask size ({camera.motion_mask.shape[2]},{camera.motion_mask.shape[1]})"

        loss, loss_dict = current_gs.loss(
            iteration=iteration,
            pred_img=image,
            gt_img=gt_image,
            pred_depth=depth,
            gt_depth=gt_depth,
            pred_normal=normal,
            gt_normal=gt_normal,
            model=current_gs.model,
            pred_translation=dyn_gs_translation,
            pred_rotation=dyn_gs_rotation,
            datamodule=current_gs.datamodule,
            motion_mask=camera.motion_mask,
            camera=camera,
            gt_max_depth=max_depth,
        )
        loss.backward(retain_graph=True)

        if not self.skip_dynamic:
            # clean total motion table (motion for every timesteps)
            self.dynamic_gs.clean_motion_table()

        with torch.no_grad():
            # Densification
            # TODO : change to support simulatanous learning
            if iteration < current_gs.densify_until_iter:
                if learn_static:
                    radii = radii[: len(self.static_gs.get_xyz)]
                    grad_densification = torch.norm(
                        viewspace_point_tensor.grad[:, :2], dim=-1, keepdim=True
                    )[: len(self.static_gs.get_xyz)]
                    visibility_filter = visibility_filter[: len(self.static_gs.get_xyz)]

                if learn_dynamic:
                    radii = radii[len(self.static_gs.get_xyz) :]
                    grad_densification = torch.norm(
                        viewspace_point_tensor.grad[:, :2], dim=-1, keepdim=True
                    )[len(self.static_gs.get_xyz) :]
                    visibility_filter = visibility_filter[len(self.static_gs.get_xyz) :]

                max_radii_update = torch.max(
                    current_gs.max_radii2D[[visibility_filter]],
                    radii[visibility_filter],
                )
                current_gs.max_radii2D[visibility_filter] = max_radii_update
                current_gs.add_densification_stats(
                    visibility_filter, grad_densification
                )

                if (
                    current_gs.densification_interval != 0
                    and iteration > current_gs.densify_from_iter
                    and iteration % current_gs.densification_interval == 0
                ):
                    size_threshold = (
                        20 if iteration > current_gs.opacity_reset_interval else None
                    )
                    current_gs.densify_and_prune(
                        current_gs.densify_grad_threshold,
                        0.005,
                        current_gs.spatial_lr_scale,
                        size_threshold,
                    )

                if (
                    current_gs.opacity_reset_interval != 0
                    and iteration % current_gs.opacity_reset_interval == 0
                ):
                    current_gs.reset_opacity()

            current_gs.optimizer.step()
            current_gs.optimizer.zero_grad(set_to_none=True)

            if current_gs.is_optimizable_cam:
                current_gs.cam_optimizer.step()
                current_gs.cam_optimizer.zero_grad(set_to_none=True)
