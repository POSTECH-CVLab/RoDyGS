# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import os
import numpy as np
import argparse
from PIL import Image
import json
import math


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def convert2format(data_dir, output_dir, resolution):

    split_path = os.path.join(data_dir, "splits")
    with open(os.path.join(split_path, "train.json"), "r") as fp:
        train_json = json.load(fp)

    img_paths = []
    cam_paths = []
    for frame_name in train_json["frame_names"]:
        if resolution == 1:
            img_paths.append(os.path.join(data_dir, "rgb", "1x", frame_name + ".png"))
        else:
            img_paths.append(os.path.join(data_dir, "rgb", "2x", frame_name + ".png"))
        cam_paths.append(os.path.join(data_dir, "camera", frame_name + ".json"))

    os.makedirs(output_dir, exist_ok=True)
    train_outimgdir = os.path.join(output_dir, "train")
    os.makedirs(train_outimgdir, exist_ok=True)
    test_outimgdir = os.path.join(output_dir, "test")
    os.makedirs(test_outimgdir, exist_ok=True)

    with open(cam_paths[0], "r") as fp:
        cam_0 = json.load(fp)
    train_transforms = dict()
    train_transforms["camera_angle_x"] = (
        focal2fov(cam_0["focal_length"], 720) * 180 / math.pi
    )
    train_transforms["camera_angle_y"] = (
        focal2fov(cam_0["focal_length"], 960) * 180 / math.pi
    )
    train_transforms["frames"] = []
    test_transforms = dict()
    test_transforms["camera_angle_x"] = (
        focal2fov(cam_0["focal_length"], 720) * 180 / math.pi
    )
    test_transforms["camera_angle_y"] = (
        focal2fov(cam_0["focal_length"], 960) * 180 / math.pi
    )
    test_transforms["frames"] = []

    # Get train & test images from only the train cameras
    frame_idx = 0
    train_id = 0
    test_id = 0
    for img, cam in zip(img_paths, cam_paths):
        frame = dict()
        image_fname = f"rgba_{frame_idx:05d}.png"
        image = Image.open(img)

        frame["time"] = frame_idx / len(img_paths)
        frame["file_path"] = image_fname  # tmp
        frame["width"] = int(720 / resolution)
        frame["height"] = int(960 / resolution)

        with open(cam, "r") as fp:
            cam = json.load(fp)
        w2c_rot = np.array(cam["orientation"])
        c2w_rot = np.linalg.inv(w2c_rot)

        c2w = np.eye(4)
        c2w[:3, :3] = c2w_rot
        c2w[:3, 3] = np.array(cam["position"])
        frame["transform_matrix"] = c2w.tolist()

        if (frame_idx + 4) % 8 == 0:
            image_fname = f"rgba_{train_id:05d}.png"
            frame["file_path"] = os.path.join("test", image_fname)
            test_transforms["frames"].append(frame)
            test_out_image_fpath = os.path.join(test_outimgdir, image_fname)
            image.save(test_out_image_fpath)
            train_id += 1
        else:
            image_fname = f"rgba_{test_id:05d}.png"
            frame["file_path"] = os.path.join("train", image_fname)
            train_transforms["frames"].append(frame)
            train_out_image_fpath = os.path.join(train_outimgdir, image_fname)
            image.save(train_out_image_fpath)
            test_id += 1

        frame_idx += 1

    with open(os.path.join(output_dir, "train_transforms.json"), "w") as fp:
        json.dump(train_transforms, fp, indent=4)
    with open(os.path.join(output_dir, "test_transforms.json"), "w") as fp:
        json.dump(test_transforms, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--resolution", type=int, default=1)
    args = parser.parse_args()
    assert args.resolution in [1, 2], "assume resolution is 1x or 2x"

    convert2format(args.data_dir, args.output_dir, args.resolution)
