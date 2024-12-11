# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import math
import random
from typing import List
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch3d.ops as torch3d

from src.utils.configs import instantiate_from_config
from src.utils.loss_utils import ssim, l1_loss, pearson_depth_loss, CharbonnierLoss
from src.utils.graphic_utils import quaternion_to_matrix


def read_loss_config(loss_config: DictConfig, **loss_model_kwargs):
    assert "name" in loss_config.keys(), "the name of loss is not given"
    assert "weight" in loss_config.keys(), "the weight is not given"

    name = loss_config.pop("name")
    weight = loss_config.pop("weight")
    freq = loss_config.pop("freq", 1)
    start = loss_config.pop("start", 0)
    model = instantiate_from_config(loss_config, **loss_model_kwargs)

    return name, weight, model, freq, start


class MultiLoss(nn.Module):

    def __init__(self, loss_configs: List[DictConfig], **loss_model_kwargs):
        super(MultiLoss, self).__init__()
        self.names, self.weights, self.loss_models, self.freqs, self.starts = (
            [],
            [],
            [],
            [],
            [],
        )
        for loss_config in loss_configs:
            name, weight, model, freq, start = read_loss_config(
                loss_config, **loss_model_kwargs
            )
            self.names.append(name)
            self.weights.append(weight)
            self.loss_models.append(model)
            self.freqs.append(freq)
            self.starts.append(start)

        self.loss_models = nn.ModuleList(self.loss_models)

    def forward(self, iteration, **kwargs):
        loss_total = 0.0
        loss_dict = {}
        # insert iteration to kwargs
        kwargs = {**kwargs, "iteration": iteration}

        for name, weight, loss_model, freq, start in zip(
            self.names, self.weights, self.loss_models, self.freqs, self.starts
        ):
            if iteration % freq == 0 and iteration > start:
                loss_out = loss_model(**kwargs)
                loss_dict[name] = loss_out
                loss_total += weight * loss_out

        return loss_total, loss_dict


class SSIMLoss(nn.Module):
    def __init__(self, mode=None):
        super(SSIMLoss, self).__init__()
        self.mode = mode

    def forward(self, pred_img, gt_img, iteration, motion_mask=None, **kwargs):
        if motion_mask is not None:
            assert self.mode != None, "Set mode(static or dynamic) of SSIMLoss"

            if self.mode == "static":
                return 1 - ssim(pred_img * ~motion_mask, gt_img * ~motion_mask)
            elif self.mode == "dynamic":
                return 1 - ssim(pred_img * motion_mask, gt_img * motion_mask)

        return 1 - ssim(pred_img, gt_img)


class L1Loss(nn.Module):
    def __init__(self, mode=None):
        super(L1Loss, self).__init__()
        self.mode = mode

    def forward(self, pred_img, gt_img, iteration, motion_mask=None, **kwargs):
        if motion_mask is not None:
            if self.mode == "static":
                return l1_loss(pred_img * ~motion_mask, gt_img * ~motion_mask)
            elif self.mode == "dynamic":
                return l1_loss(pred_img * motion_mask, gt_img * motion_mask)

        return l1_loss(pred_img, gt_img)


class GlobalPearsonDepthLoss(nn.Module):
    eps = 1e-6

    def __init__(self, mode=None):
        super(GlobalPearsonDepthLoss, self).__init__()
        self.mode = mode

    def forward(self, pred_depth, gt_depth, motion_mask=None, **kwargs):
        # motion_mask=Nones
        err_mask = None
        if self.mode == "static":
            err_mask = ~motion_mask
        elif self.mode == "dynamic":
            err_mask = motion_mask
        return pearson_depth_loss(
            pred_depth,
            gt_depth,
            self.eps,
            err_mask if motion_mask is not None else None,
        )


class LocalPearsonDepthLoss(nn.Module):
    eps = 1e-6

    def __init__(self, box_p: int, p_corr: float, mode=None):
        super(LocalPearsonDepthLoss, self).__init__()
        self.box_p = box_p
        self.p_corr = p_corr
        self.mode = mode

    def forward(self, pred_depth, gt_depth, motion_mask=None, **kwargs):

        num_box_h = math.floor(pred_depth.shape[1] / self.box_p)
        num_box_w = math.floor(pred_depth.shape[2] / self.box_p)
        max_h = pred_depth.shape[1] - self.box_p
        max_w = pred_depth.shape[2] - self.box_p

        n_corr = int(self.p_corr * num_box_h * num_box_w)
        x_0 = torch.randint(0, max_h, size=(n_corr,), device="cuda")
        y_0 = torch.randint(0, max_w, size=(n_corr,), device="cuda")
        x_1 = x_0 + self.box_p
        y_1 = y_0 + self.box_p

        _loss = torch.tensor(0.0).type_as(pred_depth)
        for i in range(len(x_0)):
            if motion_mask is not None:
                with torch.no_grad():
                    if self.mode == "static":
                        err_mask = ~motion_mask
                        mask = err_mask[:, x_0[i] : x_1[i], y_0[i] : y_1[i]].reshape(-1)
                        if mask.sum() == 0:
                            continue
                    elif self.mode == "dynamic":
                        err_mask = motion_mask
                        mask = err_mask[:, x_0[i] : x_1[i], y_0[i] : y_1[i]].reshape(-1)
                        if mask.sum() == 0:
                            continue
                    else:
                        mask = None
            else:
                mask = None

            depth_loss = pearson_depth_loss(
                pred_depth[:, x_0[i] : x_1[i], y_0[i] : y_1[i]].reshape(-1),
                gt_depth[:, x_0[i] : x_1[i], y_0[i] : y_1[i]].reshape(-1),
                self.eps,
                mask,
            )

            _loss += depth_loss

        return _loss / n_corr


class RigidityLoss(nn.Module):

    def __init__(
        self,
        scale: float = 2,
        K: int = 8,
        sim_metric: str = "l2",
        dist_weight_lambda: float = 0.1,
        color_sim: bool = True,
        dist_preserving_ratio=4,
        mode=["coeff"],
    ):
        super(RigidityLoss, self).__init__()
        self.scale = scale
        self.K = K
        self.sim_metric = sim_metric
        self.dist_weight_lambda = dist_weight_lambda
        self.color_sim = color_sim

        self.trunc_loss = CharbonnierLoss(out_norm="bc")
        # use rigidity loss for coefficients
        self.mode = mode
        self.dist_preserving_ratio = dist_preserving_ratio

        for m in self.mode:
            assert m in [
                "coeff",
                "surface",
                "distance_preserving",
            ], f"Invalid mode: {m}"

    def forward(self, model, pred_translation, **kwargs):
        canon_gaussian_points, coefficients = model._xyz, model._motion_coeff

        gaussian_points = canon_gaussian_points + pred_translation
        gaussian_colors = model._features_dc

        # compute the laplacian of the coefficients
        # random sample gaussians
        scale = 1 / self.scale if self.scale > 1 else self.scale
        indice = torch.tensor(
            random.sample(
                range(len(gaussian_points)), int(len(gaussian_points) * scale)
            )
        )
        target_points, target_coeffs = gaussian_points[indice], coefficients[indice]
        target_colors = gaussian_colors[indice]

        # knn point of random sampled query for all gaussians
        # knn_points : func(query, target, K=topk) , out : dist, idx [N, num_query_point, k-nn]
        results = torch3d.knn_points(target_points[None], target_points[None], K=self.K)
        deform_dists, nn_indice = results.dists, results.idx

        sim_loss = torch.tensor(0.0, dtype=torch.float32).to(target_points.device)

        if (
            "surface" in self.mode
        ):  # surface smoothing(oridinary laplacian regularization)
            # compute the mean of the coordinates of nn points
            nn_points = torch3d.knn_gather(target_points.view(1, -1, 3), nn_indice)
            nn_points = nn_points.reshape(-1, self.K, 3)

            # compute the l1 loss between the mean of nn points and the query points
            # mean shape : [N, 3]
            mean_nn_points = nn_points.mean(dim=1)
            sim_loss += F.pairwise_distance(target_points, mean_nn_points, p=2).mean()

        if "coeff" in self.mode:  # coeff rigidity
            coeff_shape = target_coeffs.shape
            coeff_nn = torch3d.knn_gather(
                target_coeffs.view(1, coeff_shape[0], -1), nn_indice
            )
            coeff_nn = coeff_nn.reshape(
                -1, self.K, *coeff_shape[1:]
            )  # get [1, query, K, dim]

            color_nn = torch3d.knn_gather(target_colors.view(1, -1, 3), nn_indice)
            color_nn = color_nn.reshape(-1, self.K, 3)
            color_dists = F.pairwise_distance(target_colors[None], color_nn[None], p=2)

            # compute the similarity between sampled coefficients between nn points
            sampled_coeff = coefficients[indice][:, None]
            dist_weights = torch.exp(
                -1 * self.dist_weight_lambda * (deform_dists[0] ** 2)
            )
            color_weights = torch.exp(
                -1 * self.dist_weight_lambda * (color_dists[0] ** 2)
            )

            if self.sim_metric == "cosine":
                sim = F.cosine_similarity(sampled_coeff, coeff_nn, dim=2)
            elif self.sim_metric == "l2":
                sim = F.pairwise_distance(sampled_coeff, coeff_nn, p=2)
            elif self.sim_metric == "l1":
                sim = F.pairwise_distance(sampled_coeff, coeff_nn, p=1)
            else:
                raise ValueError("Invalid similarity metric")

            # distance preserving loss between canonical space and deformed space

            if self.color_sim:
                sim = color_weights * dist_weights * sim.squeeze()
            else:
                sim = dist_weights * sim.squeeze()

            sim_loss += sim.mean()

        if "distance_preserving" in self.mode:  # distance preserving rigidity
            # random sample timesteps and embeddings and compute the motion of the gaussians for each timesteps
            # time_embeddings = torch.tensor(model._time_embeddings).cuda()
            total_timesteps = model.unique_times
            time_indices = torch.randint(
                0,
                len(total_timesteps) - 1,
                (len(total_timesteps) // self.dist_preserving_ratio,),
            )
            transl_gaussians = model.get_motion_for_times(
                timesteps=None, time_indices=time_indices
            )[..., :3]

            # matmul : refer to https://pytorch.org/docs/stable/generated/torch.matmul.html
            # output shape : [N_gaussian, timeframes, 1, 3]
            transl_for_times = target_coeffs[:, None] @ transl_gaussians
            transl_for_times = transl_for_times.squeeze()
            try:
                nn_transl_for_times = torch3d.knn_gather(
                    transl_for_times[None].view(1, len(indice), -1), nn_indice
                ).view(1, len(indice), self.K, transl_for_times.shape[1], 3)
            except:
                print(f"----- error occured in distance preserving rigidity loss -----")
                print(f"transl_for_times shape : {transl_for_times.shape}")
                print(f"nn_indice shape : {nn_indice.shape}")
                print(f"indice shape : {indice.shape}")
                print(f"target_points shape : {target_points.shape}")
                print(f"target_coeffs shape : {target_coeffs.shape}")
                print(f"transl_gaussians shape : {transl_gaussians.shape}")
                print(f"transl_for_times shape : {transl_for_times.shape}")
                print(f"-------------------------------------------------------------")

                return sim_loss
            nn_transl_for_times = nn_transl_for_times.squeeze().permute(2, 0, 1, 3)

            # sample the points in canonical space
            canon_target_points = canon_gaussian_points[indice]
            nn_points_canon = torch3d.knn_gather(
                canon_target_points.view(1, -1, 3), nn_indice
            )
            nn_points_canon = nn_points_canon.reshape(-1, self.K, 3)

            # compute the location of the gaussians in times
            # dim : [timeframes, N_gaussian, NN, 3] = [timeframes, N_gaussian, NN, 3] + [None, N_gaussian, NN, 3]
            gs_loc_in_times = nn_transl_for_times + nn_points_canon[None]

            # get gs location of the corresponded target points in various times
            # dim : [timeframes, N_gaussian, 3]
            target_gs_loc_in_times = (
                transl_for_times.transpose(0, 1)[None, :]
                + canon_target_points[None, None]
            )

            # compute distance between
            diff = gs_loc_in_times[None] - target_gs_loc_in_times[:, :, :, None]
            dists_between_nns = torch.norm(diff, dim=-1)

            # compare dists_between_nns with deform_dists
            # dists dim : [1, N, K], nn_dists dim : [timesteps, N, K]
            # dist_loss = (dists_between_nns - deform_dists[None]).mean()

            # dists dim : [N*K, 1, 1], nn_dists dim : [N*K, timesteps, 1]
            dist_loss = self.trunc_loss(
                dists_between_nns.view(-1, len(time_indices), 1),
                deform_dists[None].view(-1, 1, 1),
            )
            sim_loss += dist_loss

        return sim_loss


class MotionL1Loss(nn.Module):

    def forward(self, model, **kwargs):
        return model._motion_coeff.abs().mean()


class MotionSparsityLoss(nn.Module):

    def forward(self, model, **kwargs):
        coeffs = model._motion_coeff
        abs_tensor = torch.abs(coeffs)
        max_coeff, max_coeff_ind = torch.max(abs_tensor, dim=2)
        normalized_coeffs = abs_tensor / (max_coeff[..., None] + 1e-7)

        return normalized_coeffs.mean()


class MotionBasisRegularizaiton(nn.Module):
    def __init__(self, transl_degree=0, rot_degree=0, freq_div_mode="vanilla"):
        super(MotionBasisRegularizaiton, self).__init__()
        # 0 degree : velocity reg, 1 degree : acceleration reg, 2 degree : jerk reg
        self.degree = {"transl": transl_degree, "rot": rot_degree}
        # assume 16 basis
        self.coeff_bank = {
            "gaussian": torch.tensor(
                [
                    2.368737348178644,
                    2.3218332060968687,
                    2.186620166400238,
                    1.9785357455909518,
                    1.7200563444604107,
                    1.4367118264767467,
                    1.1529882480025957,
                    0.8890134170352768,
                    0.6585973377702478,
                    0.4687700396753248,
                    0.3205737399288996,
                    0.2106319563365025,
                    0.13296850925636292,
                    0.08064947764026723,
                    0.04699834214974086,
                    0.026314295000921823,
                ]
            ),
            "sigmoid": torch.tensor(
                [
                    0.0,
                    0.006057306357564347,
                    0.019407599012746118,
                    0.04848852855754725,
                    0.11024831053568876,
                    0.23462085565239668,
                    0.4602813915432914,
                    0.8016437593070956,
                    1.1983562406929047,
                    1.539718608456709,
                    1.7653791443476032,
                    1.889751689464311,
                    1.9515114714424528,
                    1.9805924009872535,
                    1.9939426936424351,
                    2.0,
                ]
            ),
            "laplacian": torch.tensor(
                [
                    3.0235547043507864,
                    2.475477220065594,
                    2.0267493286116927,
                    1.6593620041145454,
                    1.3585707032576908,
                    1.112303614987853,
                    0.910677176350366,
                    0.7455994104042655,
                    0.6104451667747834,
                    0.49979023110633275,
                    0.40919363229470634,
                    0.3350194107233597,
                    0.274290694437278,
                    0.22457022681891523,
                    0.18386255092234366,
                    0.15053392477948924,
                ]
            ),
            "cum_exponential": torch.tensor(
                [
                    0.24858106424723717,
                    0.45210202617930384,
                    0.6187308966091,
                    0.7551550771806206,
                    0.8668497492779882,
                    0.9582976122790642,
                    1.0331687900213073,
                    1.0944681257580495,
                    1.1446557770689725,
                    1.1857459506219796,
                    1.219387739359138,
                    1.246931306386802,
                    1.2694820717618154,
                    1.2879450768797849,
                    1.3030613069641026,
                    1.3154374294047362,
                ]
            ),
            "vanilla": torch.ones(16),
        }
        assert (
            freq_div_mode in self.coeff_bank.keys()
        ), f"Invalid freq_div_mode : {freq_div_mode}"
        self.reg_coeff = self.coeff_bank[freq_div_mode]
        reg_coeff_max = self.reg_coeff.max()
        self.reg_coeff = (
            self.reg_coeff / reg_coeff_max * 1.3
            if freq_div_mode != "vanilla"
            else self.reg_coeff
        )
        self.reg_coeff = self.reg_coeff.cuda()

    def first_derivate_motion(self, temporal_motion, is_rot=False):
        # if matrix, get diff by matrix multiplication
        if is_rot:
            return temporal_motion[1:] @ temporal_motion[:-1].transpose(-1, -2)

        return temporal_motion[1:] - temporal_motion[:-1]

    def derivate_motion(self, temporal_motion, degree):
        # get derivation recursively
        for _ in range(degree):
            temporal_motion = self.first_derivate_motion(temporal_motion)

        return temporal_motion

    def forward(self, model, **kwargs):
        temporal_motion = model.get_total_motion_table()
        transl_motion, rot_motion = temporal_motion[..., :3], temporal_motion[..., 3:]
        orig_shape = temporal_motion.shape[:-1]
        rot_motion = quaternion_to_matrix(rot_motion.reshape(-1, 4)).reshape(
            *orig_shape, 3, 3
        )

        # transl : [timesteps, N, 3]
        transl_deriv, rot_deriv = self.derivate_motion(
            transl_motion, self.degree["transl"] + 1
        ), self.derivate_motion(rot_motion, self.degree["rot"] + 1)

        # get l2 norm of translational motion (shape : [N, T, B, 3])
        transl_norm = torch.norm(transl_deriv, dim=-1) * self.reg_coeff[None]
        transl_norm = transl_norm.mean()

        rot_norm = (
            torch.norm(
                torch.eye(3)[None, None].to(rot_deriv.device) - rot_deriv, dim=(-1, -2)
            )
            * self.reg_coeff[None]
        )
        rot_norm = rot_norm.mean()

        if self.degree["transl"] < 0:
            transl_norm = 0
        if self.degree["rot"] < 0:
            rot_norm = 0

        return transl_norm + rot_norm
