# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from pathlib import Path
from typing import Union
from abc import ABC, abstractmethod

import numpy as np
import torch
import cv2


def load_depth_from_pgm(pgm_file_path, min_depth_threshold=None):
    """
    Load the depth map from PGM file
    :param pgm_file_path: pgm file path
    :return: depth map with 3D ND-array
    """
    raw_img = None
    with open(pgm_file_path, "rb") as f:
        line = str(f.readline(), encoding="ascii")
        if line != "P5\n":
            print("Error loading pgm, format error\n")

        line = str(f.readline(), encoding="ascii")
        max_depth = float(line.split(" ")[-1].strip())

        line = str(f.readline(), encoding="ascii")
        dims = line.split(" ")
        cols = int(dims[0].strip())
        rows = int(dims[1].strip())

        line = str(f.readline(), encoding="ascii")
        max_factor = float(line.strip())

        raw_img = (
            np.frombuffer(
                f.read(cols * rows * np.dtype(np.uint16).itemsize), dtype=np.uint16
            )
            .reshape((rows, cols))
            .astype(np.float32)
        )
        raw_img *= max_depth / max_factor

    if min_depth_threshold is not None:
        raw_img[raw_img < min_depth_threshold] = min_depth_threshold

    return np.expand_dims(raw_img, axis=0)


def save_depth_to_pgm(depth, pgm_file_path):
    """
    Save the depth map to PGM file
    :param depth: depth map with 3D ND-array, 1XHXW
    :param pgm_file_path: output file path
    """
    depth_flatten = depth[0]
    max_depth = np.max(depth_flatten)
    depth_copy = np.copy(depth_flatten)
    depth_copy = 65535.0 * (depth_copy / max_depth)
    depth_copy = depth_copy.astype(np.uint16)

    with open(pgm_file_path, "wb") as f:
        f.write(bytes("P5\n", encoding="ascii"))
        f.write(bytes("# %f\n" % max_depth, encoding="ascii"))
        f.write(
            bytes(
                "%d %d\n" % (depth_flatten.shape[1], depth_flatten.shape[0]),
                encoding="ascii",
            )
        )
        f.write(bytes("65535\n", encoding="ascii"))
        f.write(depth_copy.tobytes())


class Storer(ABC):
    """
    Abstract base class for storing tensor data with validation. Subclasses
    must implement the `sanity_check` method to validate tensors and the `store`
    method to define how tensors are stored.
    """

    sanity_check = None

    def __init__(self, path: Path):
        self.path = path
        self.path.mkdir(exist_ok=True)

    def to_cv2(self, tensor: torch.Tensor):
        np_arr = np.transpose(tensor.clamp(0, 1).cpu().detach().numpy(), (1, 2, 0))
        np_arr = np_arr[..., ::-1]
        # cv2_format = (np_arr * 65536).astype(np.uint16)
        cv2_format = (np_arr * 65535).astype(np.uint16)
        return cv2_format

    @abstractmethod
    def sanity_check(self, tensor: torch.Tensor) -> bool:
        """Abstract method for sanity checking the tensor.
        Subclasses should implement this method to define thir specific checks
        """
        raise NotImplementedError

    @abstractmethod
    def store(self, image_name, tensor: torch.Tensor) -> None:
        """Abstract method for storing the tensor.
        Subclasses should implement this method to define thir specific checks
        """
        raise NotImplementedError

    def __call__(self, image_name, tensor: torch.Tensor):
        if not self.sanity_check(tensor):
            raise ValueError(
                f"Sanity check failed for tensor with shape {tensor.shape}"
            )
        self.store(image_name, tensor)


class RGBStorer(Storer):

    def sanity_check(self, tensor: torch.Tensor) -> bool:
        return tensor.ndim == 3 and tensor.shape[0] == 3

    def store(self, image_name: str, tensor: torch.Tensor) -> None:
        cv2_image = self.to_cv2(tensor.clamp(0, 1))
        cv2.imwrite(self.path.joinpath(image_name).as_posix(), cv2_image)


class AssetStorer:

    def __init__(
        self,
        out_path: Path,
    ):
        self.out_path = out_path
        out_path.mkdir(exist_ok=True)

        self.viz_storer = RGBStorer(out_path.joinpath("viz"))

    def __call__(
        self,
        image_name: str,
        viz_tensor: Union[torch.Tensor],  # 3 X H X W
    ):
        self.viz_storer(image_name, viz_tensor)
