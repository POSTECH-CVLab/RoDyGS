#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from math import exp

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def smooth_loss(output):
    grad_output_x = output[:, :, :-1] - output[:, :, 1:]
    grad_output_y = output[:, :-1, :] - output[:, 1:, :]

    return torch.abs(grad_output_x).mean() + torch.abs(grad_output_y).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def logl1(pred, gt):
    return torch.log(1 + torch.abs(pred - gt))


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def pearson_depth_loss(input_depth, target_depth, eps, mask=None):

    if mask is not None:
        pred_depth = input_depth * mask
        gt_depth = target_depth * mask
    else:
        pred_depth = input_depth
        gt_depth = target_depth

    centered_pred_depth = pred_depth - pred_depth.mean()
    centered_gt_depth = gt_depth - gt_depth.mean()

    normalized_pred_depth = centered_pred_depth / (centered_pred_depth.std() + eps)
    normalized_gt_depth = centered_gt_depth / (centered_gt_depth.std() + eps)

    covariance = (normalized_pred_depth * normalized_gt_depth).mean()

    return 1 - covariance


def compute_fundamental_matrix(K, w2c1, w2c2):

    # Compute relative rotation and translation
    R_rel = w2c2[:, :3, :3] @ w2c1[:, :3, :3].transpose(1, 2)
    t_rel = w2c2[:, :3, 3] - torch.einsum("nij, nj -> ni", R_rel, w2c1[:, :3, 3])

    # Skew-symmetric matrix [t]_x for cross product
    def skew_symmetric(t):
        B = t.shape[0]
        zero = torch.zeros(B, device=t.device)
        tx = torch.stack(
            [zero, -t[:, 2], t[:, 1], t[:, 2], zero, -t[:, 0], -t[:, 1], t[:, 0], zero],
            dim=1,
        )
        return tx.view(B, 3, 3)

    t_rel_skew = skew_symmetric(t_rel)  # (B, 3, 3)

    # Compute Essential matrix E = [t]_x * R_rel
    E = t_rel_skew @ R_rel

    # Compute Fundamental matrix: F = K2^-T * E * K1^-1
    inverse_K = invert_intrinsics(K)
    K1_inv = inverse_K
    K2_inv_T = inverse_K.transpose(1, 2)
    F = K2_inv_T.bmm(E).bmm(K1_inv)

    return F


def invert_intrinsics(K):
    """
    Efficiently compute the inverse of a batch of intrinsic matrices.

    Args:
        K: (B, 3, 3) tensor representing a batch of intrinsic matrices.

    Returns:
        K_inv: (B, 3, 3) tensor representing the batch of inverse intrinsic matrices.
    """
    B = K.shape[0]

    # Extract intrinsic parameters from the matrix
    fx = K[:, 0, 0]  # focal length in x
    fy = K[:, 1, 1]  # focal length in y
    cx = K[:, 0, 2]  # principal point x
    cy = K[:, 1, 2]  # principal point y

    # Construct the inverse intrinsic matrices manually
    K_inv = torch.zeros_like(K)
    K_inv[:, 0, 0] = 1.0 / fx
    K_inv[:, 1, 1] = 1.0 / fy
    K_inv[:, 0, 2] = -cx / fx
    K_inv[:, 1, 2] = -cy / fy
    K_inv[:, 2, 2] = 1.0

    return K_inv


def construct_intrinsics(focal, image_width, image_height, batch_size):
    K = torch.zeros(batch_size, 3, 3)
    K[:, 0, 0] = focal
    K[:, 1, 1] = focal
    K[:, 2, 2] = 1.0
    K[:, 0, 2] = image_width / 2
    K[:, 1, 2] = image_height / 2
    return K


def compute_sampson_error(x1, x2, F):
    """
    :param x1 (*, N, 2)
    :param x2 (*, N, 2)
    :param F (*, 3, 3)
    """
    h1 = torch.cat([x1, torch.ones_like(x1[..., :1])], dim=-1)
    h2 = torch.cat([x2, torch.ones_like(x2[..., :1])], dim=-1)
    d1 = torch.matmul(h1, F.transpose(-1, -2))  # (B, N, 3)
    d2 = torch.matmul(h2, F)  # (B, N, 3)
    z = (h2 * d1).sum(dim=-1)  # (B, N)
    err = (z**2) / (
        d1[..., 0] ** 2 + d1[..., 1] ** 2 + d2[..., 0] ** 2 + d2[..., 1] ** 2
    )
    return err


def get_outnorm(x: torch.Tensor, out_norm: str = "") -> torch.Tensor:
    """Common function to get a loss normalization value. Can
    normalize by either the batch size ('b'), the number of
    channels ('c'), the image size ('i') or combinations
    ('bi', 'bci', etc)
    """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if "b" in out_norm:
        # normalize by batch size
        # norm /= b
        norm /= img_shape[0]
    if "c" in out_norm:
        # normalize by the number of channels
        # norm /= c
        norm /= img_shape[1]
    if "i" in out_norm:
        # normalize by image/map size
        # norm /= h*w
        norm /= img_shape[-1] * img_shape[-2]

    return norm


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6, out_norm: str = "bci"):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.out_norm = out_norm

    def forward(self, x, y, weight=None):
        norm = get_outnorm(x, self.out_norm)
        if weight is None:
            loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        else:
            loss = torch.sum(weight * torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss * norm
