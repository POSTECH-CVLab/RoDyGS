# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse

from pathlib import Path
from omegaconf import OmegaConf

from src.utils.configs import str2bool, instantiate_from_config
from src.utils.general_utils import seed_all


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_path", type=str, required=True, help="path to log"
    )
    parser.add_argument(
        "-c",
        "--eval_config",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--dirpath",
        type=str,
        required=True,
        help="path to data directory",
    )
    parser.add_argument(
        "-t",
        "--task_name",
        type=str,
        default="eval",
        help="the name of evaluation task.",
    )
    parser.add_argument(
        "--verbose", type=str2bool, default=False, help="verbose printing"
    )
    parser.add_argument(
        "--debug", type=str2bool, default=False, help="debug mode running"
    )

    args, unknown = parser.parse_known_args()

    return args, unknown


if __name__ == "__main__":

    args, unknown = parse_args()

    model_path = Path(args.model_path)

    train_config_path = model_path.joinpath("train/config.yaml")
    train_config = OmegaConf.load(train_config_path)
    eval_config = OmegaConf.load(args.eval_config)
    config = OmegaConf.merge(train_config, eval_config)

    out_path = model_path.joinpath(args.task_name)
    out_path.mkdir(exist_ok=args.debug)

    static_ckpt_path = model_path.joinpath("train", "static_last.ckpt")
    dynamic_ckpt_path = model_path.joinpath("train", "dynamic_last.ckpt")

    seed_all(config.metadata.seed)

    normalize_cams = eval_config.get("normalize_cams", False)

    static_datamodule = instantiate_from_config(
        config.static_data, normalize_cams=normalize_cams, ckpt_path=static_ckpt_path
    )
    dynamic_datamodule = instantiate_from_config(
        config.dynamic_data, normalize_cams=normalize_cams, ckpt_path=dynamic_ckpt_path
    )
    static_model = instantiate_from_config(config.static_model)
    dynamic_model = instantiate_from_config(config.dynamic_model)
    evaluator = instantiate_from_config(
        eval_config.evaluator,
        dirpath=args.dirpath,
        static_datamodule=static_datamodule,
        dynamic_datamodule=dynamic_datamodule,
        static_model=static_model,
        dynamic_model=dynamic_model,
        out_path=out_path,
        static_ckpt_path=static_ckpt_path,
        dynamic_ckpt_path=dynamic_ckpt_path,
    )

    evaluator.eval()
