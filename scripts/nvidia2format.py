# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import numpy as np
import torch
import glob
import argparse
from PIL import Image
import json
import math

img_downsample = 2


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def convert2format(train_dir, test_dir, output_dir):

    train_poses_bounds = np.load(
        os.path.join(train_dir, "poses_bounds.npy")
    )  # (N_images, 17)
    train_image_paths = sorted(glob.glob(os.path.join(train_dir, "images_2/*")))
    test_image_paths = sorted(glob.glob(os.path.join(test_dir, "*.png")))

    train_poses = train_poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    H, W, focal = train_poses[0, :, -1]  # original intrinsics, same for all images

    H, W, focal = (
        H / img_downsample,
        W / img_downsample,
        focal / img_downsample,
    )  # original images are 2x downscaled (same as RodynRF training setup)

    print(f"img_h={H}, img_w={W}")
    FoVx = focal2fov(focal, W) * 180 / math.pi
    FoVy = focal2fov(focal, H) * 180 / math.pi

    # Original poses has rotation in form "down right back"(LLFF), change to "right down front"(OpenCV)
    train_poses = np.concatenate(
        [train_poses[..., 1:2], train_poses[..., :1], -train_poses[..., 2:4]], axis=-1
    )
    padding = np.array([0, 0, 0, 1]).reshape(1, 1, 4)
    train_poses = np.concatenate(
        [train_poses, np.tile(padding, (train_poses.shape[0], 1, 1))], axis=-2
    )

    train_outimgdir = os.path.join(output_dir, "train")
    os.makedirs(train_outimgdir, exist_ok=True)
    test_outimgdir = os.path.join(output_dir, "test")
    os.makedirs(test_outimgdir, exist_ok=True)

    train_transforms = dict()
    train_transforms["camera_angle_x"] = FoVx
    train_transforms["camera_angle_y"] = FoVy
    train_transforms["frames"] = []
    test_transforms = dict()
    test_transforms["camera_angle_x"] = FoVx
    test_transforms["camera_angle_y"] = FoVy
    test_transforms["frames"] = []

    for i, train_dirpath in enumerate(train_image_paths):

        frame = dict()
        image_fname = f"rgba_{i:05d}.png"
        image = Image.open(train_dirpath)
        out_image_fpath = os.path.join(train_outimgdir, image_fname)
        image.save(out_image_fpath)

        frame["time"] = i / len(train_image_paths)
        frame["file_path"] = os.path.join("train", image_fname)
        frame["width"] = int(W)
        frame["height"] = int(H)

        c2w = train_poses[i]
        frame["transform_matrix"] = c2w.tolist()
        train_transforms["frames"].append(frame)

        # c2w of all test cam == c2w of first train cam
        if i == 0:
            for j, test_dirpath in enumerate(test_image_paths):
                frame = dict()
                image_fname = f"rgba_{j:05d}.png"
                image = Image.open(test_dirpath)
                out_image_fpath = os.path.join(test_outimgdir, image_fname)
                image.save(out_image_fpath)

                frame["time"] = j / len(test_image_paths)
                frame["file_path"] = os.path.join("test", image_fname)
                frame["width"] = int(W)
                frame["height"] = int(H)

                frame["transform_matrix"] = c2w.tolist()
                test_transforms["frames"].append(frame)
            with open(os.path.join(output_dir, "test_transforms.json"), "w") as fp:
                json.dump(test_transforms, fp, indent=4)

    with open(os.path.join(output_dir, "train_transforms.json"), "w") as fp:
        json.dump(train_transforms, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, type=str)
    parser.add_argument("--test_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    args = parser.parse_args()

    convert2format(args.train_dir, args.test_dir, args.output_dir)
