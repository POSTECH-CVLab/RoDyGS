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

from typing import Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class BasicPointCloud:
    points: np.array
    colors: np.array
    normals: np.array
    time: Optional[np.array]


def uniform_sample(pcd: BasicPointCloud, ratio: float):
    assert ratio <= 1.0
    num_points = len(pcd.points)
    num_sample = int(num_points * ratio)
    point_idx = np.random.choice(num_points, num_sample, replace=False)

    return BasicPointCloud(
        points=pcd.points[point_idx],
        colors=pcd.colors[point_idx],
        normals=pcd.normals[point_idx],
        time=pcd.time[point_idx] if not pcd.time is None else None,
    )


def merge_pcds(pcds):

    points, colors, normals, time = [], [], [], []

    for pcd in pcds:
        points.append(pcd.points)
        colors.append(pcd.colors)
        normals.append(pcd.normals)
        time.append(pcd.time)

    return BasicPointCloud(
        points=np.concatenate(points),
        colors=np.concatenate(colors),
        normals=np.concatenate(normals),
        time=np.concatenate(time) if not time[0] is None else None,
    )
