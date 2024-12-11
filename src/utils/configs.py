# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import importlib

from omegaconf import OmegaConf, DictConfig


def str2bool(v):
    # a util function that accepts various kinds of flag options.
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def is_instantiable(config: DictConfig, key: str):
    # check whether the configuration is an instantiable format.
    subconfig = config.get(key)
    is_config = OmegaConf.is_config(subconfig)
    if not is_config:
        return False

    # target and params should be in keys.
    key_list = subconfig.keys()
    has_params = "params" in key_list
    has_target = "target" in key_list

    # no other keys are needed.
    no_more_keys = len(key_list) == 2

    return has_params and has_target and no_more_keys


def instantiate_from_config(config: DictConfig, **kwargs):
    # instantiate an object from configuration.
    # if some arguments are passed in kwargs, this function overloads the configruation.
    assert "target" in config.keys(), "target not exists"

    params = dict()
    params.update(config.get("params", dict()))
    params.update(kwargs)
    return get_obj_from_str(config["target"])(**params)


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    # from a str, return the corresponding object.
    module, obj_name = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), obj_name)
