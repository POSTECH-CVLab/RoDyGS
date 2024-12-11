# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import json
from pathlib import Path
from PIL import Image

import numpy as np


# used for camera direction conversion(inverse direction)
gl_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# used for world coordinate conversion
opencv_matrix = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def quaternion_to_rotation_matrix(quaternion):
    q = np.array(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm == 0:
        return np.eye(3)
    q /= norm
    w, x, y, z = q
    rotation_matrix = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2],
        ]
    )
    return rotation_matrix


def kubric2opencv(extrinsic):
    extrinsic = opencv_matrix @ extrinsic @ gl_matrix

    return extrinsic


def convert2format(args):
    input_dir = Path(args.input_dir)
    train_dirpath = input_dir.joinpath("train")
    test_dirpath = input_dir.joinpath("test")

    outdirpath = Path(args.output_dir)
    outdirpath.mkdir(exist_ok=True, parents=True)

    for split, dirpath in zip(
        ["train", "val", "test"], [train_dirpath, test_dirpath, test_dirpath]
    ):
        metadata = dirpath.joinpath("metadata.json")
        outimgdir = outdirpath.joinpath(split)
        outimgdir.mkdir(exist_ok=True, parents=True)

        with open(metadata.as_posix(), "r") as fp:
            metadata = json.load(fp)

        transforms = dict()

        H, W = metadata["metadata"]["resolution"]
        fov = np.rad2deg(metadata["camera"]["field_of_view"])
        camera_angle_x, camera_angle_y = fov, fov

        transforms["camera_angle_x"] = camera_angle_x
        transforms["camera_angle_y"] = camera_angle_y

        transforms["frames"] = []
        num_frames = metadata["metadata"]["num_frames"]

        if split == "train":
            iterator = range(num_frames)
        elif split == "val":
            iterator = np.array(range(num_frames))[::10]
        else:
            # use the rest of the frames for testing
            iterator = np.array([idx for idx in range(num_frames) if idx % 10 != 0])

        for frame_idx in iterator:
            frame = dict()

            image_fname = f"rgba_{frame_idx:05d}.png"
            image_fpath = dirpath.joinpath(image_fname)
            image = Image.open(image_fpath)
            out_image_fpath = outimgdir.joinpath(image_fname)
            image.save(out_image_fpath)

            frame["time"] = frame_idx / num_frames
            frame["file_path"] = Path(split, image_fname).as_posix()
            frame["width"] = W
            frame["height"] = H

            c2w = np.eye(4)
            quaternion = metadata["camera"]["quaternions"][frame_idx]
            c2w[:3, :3] = quaternion_to_rotation_matrix(quaternion)
            c2w[:3, 3] = np.array(metadata["camera"]["positions"][frame_idx])

            # change coordinate system to opencv format
            # world coordinate : blender -> opencv
            # camera (local) type : opengl -> opencv
            c2w = kubric2opencv(c2w)

            frame["transform_matrix"] = c2w.tolist()
            transforms["frames"].append(frame)

        with open(outdirpath.joinpath(f"{split}_transforms.json"), "w") as fp:
            json.dump(transforms, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", required=True, type=str, help="path to generated kubric-mrig"
    )
    parser.add_argument(
        "--output_dir", required=True, type=str, help="path to store converted assets"
    )
    args = parser.parse_args()

    convert2format(args)
