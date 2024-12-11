# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
from PIL import Image
import os
import argparse


def convert_npy_to_png(input_dir, output_dir):
    os.makedirs(os.path.join(output_dir, "tam_mask"), exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".npy"):
            input_file_path = os.path.join(input_dir, file_name)
            output_file_name = f"{int(os.path.splitext(file_name)[0]):06d}" + ".png"
            output_tam_path = os.path.join(output_dir, "tam_mask", output_file_name)

            try:
                motion_mask = np.load(input_file_path)
                binary_image = (motion_mask * 255).astype(np.uint8)

                Image.fromarray(binary_image).save(output_tam_path)
                print(f"Converted: {input_file_path} -> {output_tam_path}")
            except Exception as e:
                print(f"Failed to convert {input_file_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    args = parser.parse_args()

    convert_npy_to_png(args.npy_dir, args.output_dir)
