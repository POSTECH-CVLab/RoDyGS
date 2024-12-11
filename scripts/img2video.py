# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import imageio
import os
import argparse


def convert_png_to_video(data_dir):

    imgs_path = os.path.join(data_dir, "train")
    images = [img for img in os.listdir(imgs_path) if img.endswith((".png"))]
    images.sort()
    output_video = os.path.join(data_dir, "train.mp4")

    if not images:
        print("No images!")
    else:
        with imageio.get_writer(
            output_video, fps=30, codec="libx264", macro_block_size=1
        ) as writer:
            for image in images:
                writer.append_data(imageio.imread(os.path.join(imgs_path, image)))
        print(f"Generate {output_video} ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    args = parser.parse_args()

    convert_png_to_video(args.data_dir)
