import torch
import numpy as np
import trimesh

from functools import lru_cache


@lru_cache
def mask110(device, dtype):
    return torch.tensor((1, 1, 0), device=device, dtype=dtype)


def proj3d(inv_K, pixels, z):
    if pixels.shape[-1] == 2:
        pixels = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1)
    return z.unsqueeze(-1) * (
        pixels * inv_K.diag() + inv_K[:, 2] * mask110(z.device, z.dtype)
    )


# from dust3r point transformation
def geotrf(Trf, pts, ncol=None, norm=False):
    """Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (
        isinstance(Trf, torch.Tensor)
        and isinstance(pts, torch.Tensor)
        and Trf.ndim == 3
        and pts.ndim == 4
    ):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = (
                torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts)
                + Trf[:, None, None, :d, d]
            )
        else:
            raise ValueError(f"bad shape, not ending with 3 or 4, for {pts.shape=}")
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], "batch size does not match"
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def unproject_depth(focal, extrinsic, image, depth, export_path=None, mask=None):
    h, w, _ = image.shape

    # extract color
    pixels = torch.tensor(np.mgrid[:w, :h].T.reshape(-1, 2))
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
    points = proj3d(
        torch.tensor(np.linalg.inv(K)), pixels, torch.tensor(depth.reshape(-1))
    )

    color = image.reshape(-1, 3)

    if mask is not None:
        mask = mask.ravel()
        color = color[mask]
        points = points[mask.ravel()]

    # convert to world coordinates
    # points = np.stack([x, y, z], axis=-1)
    points = geotrf(torch.tensor(extrinsic), torch.tensor(points))
    points = np.asarray(points).reshape(-1, 3)

    # save to ply with img color
    if export_path is not None:
        mesh = trimesh.PointCloud(vertices=points, colors=color)
        mesh.export(export_path)

    return points
