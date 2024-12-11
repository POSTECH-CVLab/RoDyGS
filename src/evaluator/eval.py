# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import yaml

from src.model.rodygs_static import StaticRoDyGS
from src.model.rodygs_dynamic import DynRoDyGS
from src.utils.eval_utils import VizScoreEvaluator, PoseEvaluator
from src.utils.store_utils import AssetStorer
from src.data.asset_readers import GTCameraReader
from src.data.utils import LearnableCamera, FixedCamera
from src.utils.loss_utils import l2_loss
from .utils import search_nearest_two
from diff_gauss_pose import GaussianRasterizationSettings, GaussianRasterizer
import math
import glob
import imageio
import os


class RoDyGSEvaluator:

    def __init__(
        self,
        dirpath: str,
        static_datamodule,
        dynamic_datamodule,
        static_model: StaticRoDyGS,
        dynamic_model: DynRoDyGS,
        out_path: Path,
        static_ckpt_path: Path,
        dynamic_ckpt_path: Path,
        camera_lr: float = -1,  # no optimization for -1
        num_opts: int = -1,  # no optimization for -1
    ):

        self.dirpath = dirpath
        self.static_datamodule = static_datamodule
        self.static_model = static_model
        static_state_dict = torch.load(static_ckpt_path.as_posix())
        self.static_model.create_from_state_dict(
            static_state_dict[0], static_datamodule.get_normalization()["radius"]
        )

        if not static_datamodule.skip_dynamic:
            self.dynamic_datamodule = dynamic_datamodule
            self.dynamic_model = dynamic_model
            dynamic_state_dict = torch.load(dynamic_ckpt_path.as_posix())
            self.dynamic_model.create_from_state_dict(
                dynamic_state_dict[0], dynamic_datamodule.get_normalization()["radius"]
            )
            self.skip_dynamic = False
        else:
            self.skip_dynamic = True

        self.viz_evaluator = VizScoreEvaluator("cuda")
        self.out_path = out_path
        self.gt_asset_storer = AssetStorer(out_path.joinpath("gt"))
        self.pred_asset_storer = AssetStorer(out_path.joinpath("pred"))

        self.pose_evaluator = PoseEvaluator()

        self.is_optimizable_cam = camera_lr != -1
        if self.is_optimizable_cam:
            self.static_datamodule.get_train_dloader().dataset.load_state_dict(
                static_state_dict[0]["camera"]
            )
            self.pose_optimizer = PoseOptimizer(
                dirpath, self.static_datamodule, static_model, camera_lr, num_opts
            )
        self.test_dloader = self.static_datamodule.get_test_dloader()

    def render(
        self,
        viewpoint_camera,
        bg_color: torch.Tensor,
        scaling_modifier=1,
        override_color=None,
        enable_sh_grad=False,
        enable_cov_grad=False,
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

        means3D = self.get_xyz
        means2D = screenspace_points
        opacity = self.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        cov3D_precomp = None
        scales = self.get_scaling
        rotations = self.get_rotation

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
            "translation": self.dynGS_translation,
            "rotation": self.dynGS_rotation,
        }

    def get_motion_dynGS(self, viewpoint_camera, use_deform=True):

        if use_deform:
            time = viewpoint_camera.time
            translation, rotation = self.dynamic_model._deform_network(
                self.dynamic_model._motion_coeff, time
            )
            if self.dynamic_model.inverse_motion:
                delta_gaussians = self.dynamic_model._deform_network.batch_inference(
                    self.dynamic_model._time_batch_embeddings
                ).squeeze()  # [timesteps, num_basis, 7]
                # get the inverse motion
                # motion coeff : [num_gaussians, num_basis]
                delta_gaussians = (
                    self.dynamic_model._motion_coeff
                    @ delta_gaussians[self.dynamic_model.gaussian_to_time_ind]
                ).squeeze()

                # sample unique time steps from self._timestep
                translation = translation - delta_gaussians[..., :3]
                rotation = rotation - delta_gaussians[..., 3:]
            scaled_translation = translation * self.dynamic_model.spatial_lr_scale
        else:
            scaled_translation, rotation = 0, 0

        return scaled_translation, rotation

    def set_static_GS(self):
        self.dynGS_translation, self.dynGS_rotation = None, None
        self.get_xyz = self.static_model.get_xyz
        self.active_sh_degree = (
            self.static_model.active_sh_degree
        )  # assume SH degree of dynamic GS is same as static's
        self.get_opacity = self.static_model.get_opacity
        self.get_scaling = self.static_model.get_scaling
        self.get_rotation = self.static_model.get_rotation
        self.get_features = self.static_model.get_features

    def concat_GS(self, viewpoint_camera, use_deform=True):

        self.dynGS_translation, self.dynGS_rotation = self.get_motion_dynGS(
            viewpoint_camera=viewpoint_camera, use_deform=use_deform
        )

        assert (
            self.static_model.isotropic == self.dynamic_model.isotropic
        ), "Both static and dynamic Gaussians must be isotropic or anisotropic"

        self.get_xyz = torch.cat(
            (
                self.static_model.get_xyz,
                self.dynamic_model.get_xyz + self.dynGS_translation,
            ),
            dim=0,
        )
        self.active_sh_degree = (
            self.static_model.active_sh_degree
        )  # assume SH degree of dynamic GS is same as static's
        self.get_opacity = torch.cat(
            (self.static_model.get_opacity, self.dynamic_model.get_opacity), dim=0
        )
        self.get_scaling = torch.cat(
            (self.static_model.get_scaling, self.dynamic_model.get_scaling), dim=0
        )
        self.get_rotation = torch.cat(
            (
                self.static_model.get_rotation,
                (
                    self.dynamic_model.get_rotation + self.dynGS_rotation
                    if not self.dynamic_model.isotropic
                    else self.dynamic_model.get_rotation
                ),
            ),
            dim=0,
        )
        self.get_features = torch.cat(
            (self.static_model.get_features, self.dynamic_model.get_features), dim=0
        )

        return

    def eval(self):

        torch.cuda.empty_cache()
        viz_scores = defaultdict(list)
        scores_total = [viz_scores]
        scores_name = ["viz"]

        if self.skip_dynamic:
            self.set_static_GS()

        for idx, camera in enumerate(self.test_dloader):
            torch.cuda.empty_cache()

            camera = camera.cuda()

            if self.is_optimizable_cam:
                gt_pose = torch.inverse(camera.world_view_transform)
                camera = self.pose_optimizer(camera, gt_pose, camera.original_image)

            if not self.skip_dynamic:
                self.concat_GS(
                    viewpoint_camera=camera,
                    use_deform=True,
                )

            render_pkg = self.render(
                viewpoint_camera=camera,
                bg_color=torch.zeros(3).cuda(),
            )

            pred_image = render_pkg["rendered_image"]
            gt_image = camera.original_image

            viz_score = self.viz_evaluator.get_score(gt_image, pred_image)
            scores_frame = [viz_score]

            for score_frame, score_total in zip(scores_frame, scores_total):
                for key, value in score_frame.items():
                    score_total[key].append(value)

            image_name = f"{str(idx).zfill(5)}_" + camera.image_name + ".png"
            self.gt_asset_storer(image_name, gt_image)
            self.pred_asset_storer(image_name, pred_image)

        result_dict = {
            name: {
                k: torch.stack(v).mean().item()
                for k, v in score_dict.items()
                if v != []
            }
            for name, score_dict in zip(scores_name, scores_total)
        }

        # Evaluate train pose
        calibrated_poses = self.static_datamodule.get_train_poses().cuda()
        gt_poses = torch.tensor(
            GTCameraReader(self.dirpath, "train_transforms.json").get_poses()
        ).cuda()
        pose_scores = self.pose_evaluator.get_score(gt_poses, calibrated_poses)
        result_dict["pose"] = {}
        result_dict["pose"]["ATE"] = float(pose_scores["ATE"])
        result_dict["pose"]["RPE_trans"] = float(pose_scores["RPE_trans"])
        result_dict["pose"]["RPE_rot"] = float(pose_scores["RPE_rot"])

        with open(self.out_path.joinpath("result.yaml").as_posix(), "w") as fp:
            yaml.dump(result_dict, fp)

        eval_images_path = sorted(
            glob.glob(os.path.join(str(self.out_path), "pred", "viz", "*.png"))
        )
        eval_images = []
        for image_path in eval_images_path:
            image = imageio.imread(str(image_path))
            eval_images.append(image)

        # Define the output video path
        video_path = self.out_path.joinpath("video.mp4")

        with imageio.get_writer(str(video_path), fps=30, codec="libx264") as writer:
            for image in eval_images:
                writer.append_data(image)


class PoseOptimizer:

    def __init__(
        self, dirpath, datamodule, model, camera_lr, num_opts, is_optimizable_cam=True
    ):
        self.calibrated_poses = datamodule.get_train_poses()
        self.uncalibrated_poses = torch.tensor(
            GTCameraReader(dirpath, "train_transforms.json").get_poses()
        ).cuda()
        self.model = model
        self.camera_lr = camera_lr
        self.num_opts = num_opts
        self.is_optimizable_cam = is_optimizable_cam

    def __call__(self, camera: FixedCamera, gt_pose, rgb):

        nearest_indices = search_nearest_two(gt_pose, self.uncalibrated_poses)

        nearest_poses = self.calibrated_poses[nearest_indices]
        nearest_t = nearest_poses[:, :3, 3]
        nearest_R = nearest_poses[:, :3, :3]

        init_t = nearest_t[0]
        init_R = nearest_R[0]

        init_pose = torch.eye(4).cuda()
        init_pose[:3, :3] = init_R.detach().clone()
        init_pose[:3, 3] = init_t.detach().clone()

        init_w2c = torch.inverse(init_pose).cpu().numpy()
        camera_optim = LearnableCamera(
            init_w2c[:3, :3],
            init_w2c[:3, 3],
            camera.FoVx,
            camera.FoVy,
            camera.original_image,
            camera.image_name,
            camera.time,
            camera.depth,
            camera.normal,
            camera.motion_mask,
            camera.cam_idx,
        ).cuda()

        optimizer = torch.optim.Adam(
            camera_optim.parameters(), lr=self.camera_lr, eps=1e-15
        )

        for _ in tqdm(range(self.num_opts), desc=f"optimizing {camera.image_name}"):

            render_pkg = self.model.render(
                viewpoint_camera=camera_optim,
                bg_color=torch.zeros(3).cuda(),
                enable_sh_grad=self.is_optimizable_cam,
                enable_cov_grad=self.is_optimizable_cam,
            )

            loss = l2_loss(render_pkg["rendered_image"], rgb)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        return camera_optim
