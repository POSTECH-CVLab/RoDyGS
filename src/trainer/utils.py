# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import torch
import torch.nn as nn


def replace_tensor_to_optimizer(optimizer, tensor, name):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if len(group["params"]) > 1:
            # for these cases, we don't want to modify the optimizer
            continue
        if group["name"] == name:
            stored_state = optimizer.state.get(group["params"][0], None)
            stored_state["exp_avg"] = torch.zeros_like(tensor)
            stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
    return optimizable_tensors


def cat_tensors_to_optimizer(optimizer, tensors_dict):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if len(group["params"]) > 1:
            # for these cases, we don't want to modify the optimizer
            continue
        extension_tensor = tensors_dict[group["name"]]
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:

            stored_state["exp_avg"] = torch.cat(
                (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
            )
            stored_state["exp_avg_sq"] = torch.cat(
                (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0
            )

            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(
                    True
                )
            )
            optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(
                torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(
                    True
                )
            )
            optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors


def prune_optimizer(optimizer, mask):
    optimizable_tensors = {}
    for group in optimizer.param_groups:
        if len(group["params"]) > 1:
            # for these cases, we don't want to modify the optimizer
            continue
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][mask]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

            del optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(
                (group["params"][0][mask].requires_grad_(True))
            )
            optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]
        else:
            group["params"][0] = nn.Parameter(
                group["params"][0][mask].requires_grad_(True)
            )
            optimizable_tensors[group["name"]] = group["params"][0]

    return optimizable_tensors
