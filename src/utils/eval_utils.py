# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from itertools import chain
from typing import Sequence
from collections import OrderedDict

import numpy as np
import scipy
from piqa import PSNR, SSIM, LPIPS, MS_SSIM
import torch
import torch.nn as nn
from torchvision import models

from src.utils.general_utils import batchify, reduce
from src.utils.pose_estim_utils import align_ate_c2b_use_a2b, compute_ATE, compute_rpe


class VizScoreEvaluator:

    def __init__(self, device):
        self.psnr_module = PSNR().to(device)
        self.ssim_module = SSIM().to(device)
        self.msssim_module = MS_SSIM().to(device)

    @torch.inference_mode()
    def get_score(self, gt_image, pred_image):

        bf_gt_image = batchify(gt_image).clip(0, 1).contiguous()
        bf_pred_image = batchify(pred_image).clip(0, 1).contiguous()

        psnr_score = reduce(self.psnr_module(bf_gt_image, bf_pred_image))
        ssim_score = reduce(self.ssim_module(bf_gt_image, bf_pred_image))
        lpipsa_score = reduce(lpips(bf_gt_image, bf_pred_image, net_type="alex"))
        lpipsv_score = reduce(lpips(bf_gt_image, bf_pred_image, net_type="vgg"))
        msssim_score = reduce(self.msssim_module(bf_gt_image, bf_pred_image))
        dssim_score = (1 - msssim_score) / 2

        return {
            "psnr": psnr_score,
            "ssim": ssim_score,
            "lpipsa": lpipsa_score,
            "lpipsv": lpipsv_score,
            "msssim": msssim_score,
            "dssim": dssim_score,
        }


class PoseEvaluator:
    def __init__(self):
        pass

    def normalize_pose(self, pose1, pose2):
        mtx1 = np.array(pose1.cpu().numpy(), dtype=np.double, copy=True)
        mtx2 = np.array(pose2.cpu().numpy(), dtype=np.double, copy=True)

        if mtx1.ndim != 2 or mtx2.ndim != 2:
            raise ValueError("Input matrices must be two-dimensional")
        if mtx1.shape != mtx2.shape:
            raise ValueError("Input matrices must be of same shape")
        if mtx1.size == 0:
            raise ValueError("Input matrices must be >0 rows and >0 cols")

        # translate all the data to the origin
        mtx1 -= np.mean(mtx1, 0)
        mtx2 -= np.mean(mtx2, 0)

        norm1 = np.linalg.norm(mtx1)
        norm2 = np.linalg.norm(mtx2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("Input matrices must contain >1 unique points")

        # change scaling of data (in rows) such that trace(mtx*mtx') = 1
        mtx1 /= norm1
        mtx2 /= norm2

        # transform mtx2 to minimize disparity
        R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
        mtx2 = mtx2 * s

        return mtx1, mtx2, R

    def algin_pose(self, gt, estim):
        gt_ret, estim_ret = gt.clone(), estim.clone()
        gt_transl, estim_transl, _ = self.normalize_pose(
            gt[:, :3, -1], estim_ret[:, :3, -1]
        )
        gt_ret[:, :3, -1], estim_ret[:, :3, -1] = torch.from_numpy(gt_transl).type_as(
            gt_ret
        ), torch.from_numpy(estim_transl).type_as(estim_ret)
        c2ws_est_aligned = align_ate_c2b_use_a2b(estim_ret, gt_ret)
        return gt_ret, c2ws_est_aligned

    @torch.inference_mode()
    def get_score(self, gt, estim):
        gt, c2ws_est_aligned = self.algin_pose(gt, estim)
        ate = compute_ATE(gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
        rpe_trans, rpe_rot = compute_rpe(
            gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy()
        )
        rpe_trans *= 100
        rpe_rot *= 180 / np.pi

        return {
            "ATE": ate,
            "RPE_trans": rpe_trans,
            "RPE_rot": rpe_rot,
            "aligned": c2ws_est_aligned,
        }


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """

    def __init__(self, net_type: str = "alex", version: str = "0.1"):

        assert version in ["0.1"], "v0.1 is only supported now"

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0), 0, True)


def get_network(net_type: str):
    if net_type == "alex":
        return AlexNet()
    elif net_type == "squeeze":
        return SqueezeNet()
    elif net_type == "vgg":
        return VGG16()
    else:
        raise NotImplementedError("choose net_type from [alex, squeeze, vgg].")


class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__(
            [
                nn.Sequential(nn.Identity(), nn.Conv2d(nc, 1, 1, 1, 0, bias=False))
                for nc in n_channels_list
            ]
        )

        for param in self.parameters():
            param.requires_grad = False


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer(
            "mean", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class SqueezeNet(BaseNet):
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.layers = models.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        self.set_requires_grad(False)


class AlexNet(BaseNet):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layers = models.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)


class VGG16(BaseNet):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(False)


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = "alex", version: str = "0.1"):
    # build url
    url = (
        "https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
        + f"master/lpips/weights/v{version}/{net_type}.pth"
    )

    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url,
        progress=True,
        map_location=None if torch.cuda.is_available() else torch.device("cpu"),
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace("lin", "")
        new_key = new_key.replace("model.", "")
        new_state_dict[new_key] = val

    return new_state_dict


def lpips(
    x: torch.Tensor, y: torch.Tensor, net_type: str = "alex", version: str = "0.1"
):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y)
