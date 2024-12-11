# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import os
import logging
import sys
import shutil

from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from src.utils.configs import str2bool, is_instantiable, instantiate_from_config
from src.utils.general_utils import seed_all
from src.pipelines.utils import StreamToLogger


def check_argument_sanity(args: argparse.Namespace) -> None:
    # check whether arguments are given properly.
    assert args.name != "", "provide an appropriate expname"

    config_paths = args.base
    assert len(config_paths) != 0, "no config given for training"
    for config_path in config_paths:
        assert os.path.exists(config_path), f"no config exists in path {config_path}"

    if args.verbose:
        # When running verbose mode, set the os.environ to globally share
        # the running mode in a conveninent way
        os.environ["VERBOSE_RUN"] = "True"

    if args.debug:
        os.environ["DEBUG_RUN"] = "True"

    assert args.group != "", "specify the group name"


def check_config(config: OmegaConf) -> None:
    # check whether config has a proper structure to run training
    assert is_instantiable(config, "static_data")
    assert is_instantiable(config, "static_model")
    assert is_instantiable(config, "dynamic_data")
    assert is_instantiable(config, "dynamic_model")
    assert is_instantiable(config, "trainer")


def set_traindir(logdir: str, name: str, group: str, seed: int, debug: bool) -> Path:
    # explogdir (logdir/group/name)
    is_overridable = debug

    # in the case of group, previous experiments might already have made the directory
    path_logdir = Path(logdir, group)
    path_logdir_posix = path_logdir.as_posix()
    # group logdir is always overridable
    os.makedirs(path_logdir_posix, exist_ok=True)

    # but each experiment logdir should not exist
    expname = f"{name}_{str(seed).zfill(4)}"
    path_expdir = path_logdir.joinpath(expname)
    path_expdir_posix = path_expdir.as_posix()
    os.makedirs(path_expdir_posix, exist_ok=is_overridable)

    traindir = path_expdir.joinpath("train")
    traindir_posix = traindir.as_posix()
    os.makedirs(traindir_posix, exist_ok=is_overridable)

    return traindir


def set_logger(logdir: Path) -> logging.Logger:
    # set logger in "logdir/train.log"

    logger = logging.getLogger("__main__")
    formatter = logging.Formatter(
        "%(asctime)s;[%(levelname)s];%(message)s", "%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(logdir.joinpath("train.log").as_posix())
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # Change the print function to
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.INFO)

    return logger


def store_args_and_config(logdir: Path, args: argparse.Namespace, config: DictConfig):
    # store args as runnable metadata
    config["metadata"] = vars(args)
    store_path = logdir.joinpath("config.yaml")

    # yaml_format = OmegaConf.to_yaml(config)
    OmegaConf.save(config, store_path)


def store_code(logdir: Path):
    code_path = "./src"
    dst_path = logdir.joinpath("code").as_posix()
    if not os.path.exists(dst_path):
        shutil.copytree(
            src=code_path,
            dst=dst_path,
            ignore=shutil.ignore_patterns("*__pycache__*"),
        )


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="configs/default.yaml",
        default=list(),
        help="path to config files",
    )
    parser.add_argument(
        "-d",
        "--dirpath",
        type=str,
        required=True,
        help="path to data directory",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=777,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="./logs",
        help="path to store logs",
    )
    parser.add_argument(
        "-g",
        "--group",
        type=str,
        default="",
        help="group name",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="verbose printing",
    )
    parser.add_argument(
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="run with debug mode",
    )

    args, unknown = parser.parse_known_args()

    return args, unknown


def override_config(static_data_config, dynamic_data_config, trainer_config):
    # override static_config with dynamic_config
    iteration = static_data_config["params"]["train_dloader_config"]["params"][
        "num_iterations"
    ]
    if (
        dynamic_data_config["params"]["train_dloader_config"]["params"][
            "num_iterations"
        ]
        != iteration
    ):
        print("Override num_iterations...")

        dynamic_data_config["params"]["train_dloader_config"]["params"][
            "num_iterations"
        ] = iteration

        trainer_config["params"]["static"]["params"]["num_iterations"] = iteration
        trainer_config["params"]["static"]["params"]["camera_opt_config"]["params"][
            "total_steps"
        ] = iteration
        trainer_config["params"]["static"]["params"][
            "position_lr_max_steps"
        ] = iteration

        trainer_config["params"]["dynamic"]["params"]["num_iterations"] = iteration
        trainer_config["params"]["dynamic"]["params"]["camera_opt_config"]["params"][
            "total_steps"
        ] = iteration
        trainer_config["params"]["dynamic"]["params"][
            "position_lr_max_steps"
        ] = iteration
        trainer_config["params"]["dynamic"]["params"]["deform_lr_max_steps"] = iteration

    return dynamic_data_config, trainer_config


if __name__ == "__main__":

    args, unknown = parse_args()

    # argument sanity check
    check_argument_sanity(args)

    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # config sanity check
    check_config(config)
    config.static_data.params.dirpath = args.dirpath
    config.dynamic_data.params.dirpath = args.dirpath

    seed_all(args.seed)

    logdir: Path = set_traindir(
        logdir=args.logdir,
        name=args.name,
        group=args.group,
        seed=args.seed,
        debug=args.debug,
    )
    logger = set_logger(logdir=logdir)
    store_args_and_config(logdir, args, config)
    store_code(logdir)

    config.dynamic_data, config.trainer = override_config(
        config.static_data, config.dynamic_data, config.trainer
    )

    static_datamodule = instantiate_from_config(config.static_data)
    dynamic_datamodule = instantiate_from_config(config.dynamic_data)

    print("Init Static GS model")
    static_model = instantiate_from_config(config.static_model)

    print("Init Dynamic GS model")
    dynamic_model = instantiate_from_config(config.dynamic_model)
    trainer = instantiate_from_config(
        config.trainer,
        static_datamodule=static_datamodule,
        dynamic_datamodule=dynamic_datamodule,
        static_model=static_model,
        dynamic_model=dynamic_model,
        logdir=logdir,
    )

    trainer.train()
