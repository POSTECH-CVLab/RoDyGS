# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import math

import torch
from functools import partial

VERBOSE = os.environ.get("VERBOSE_RUN", False)


def linear_warmup_cosine_annealing_func(step, max_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step / warmup_steps)
    else:
        # Cosine annealing
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return max_lr * cosine_decay


class CameraQuatOptimizer:

    eps = 1e-15

    def __init__(
        self,
        dataset,
        camera_rotation_lr,
        camera_translation_lr,
        camera_lr_warmup,
        total_steps,
        spatial_lr_scale,
    ):

        self.spatial_lr_scale = spatial_lr_scale

        cam_R_params, cam_T_params = [], []
        for name, param in dataset.named_parameters():
            if "R_" in name:
                cam_R_params.append(param)
            elif "T_" in name:
                cam_T_params.append(param)
            else:
                raise NameError(f"Unknown parameter {name}")

        l = (
            {"params": cam_R_params, "lr": 0.0, "name": "camera_R"},
            {"params": cam_T_params, "lr": 0.0, "name": "camera_T"},
        )

        self._optimizer = torch.optim.Adam(l, lr=0.0, eps=self.eps)
        self.R_lr_fn = partial(
            linear_warmup_cosine_annealing_func,
            max_lr=camera_rotation_lr,
            warmup_steps=camera_lr_warmup,
            total_steps=total_steps,
        )
        self.T_lr_fn = partial(
            linear_warmup_cosine_annealing_func,
            max_lr=camera_translation_lr,
            warmup_steps=camera_lr_warmup,
            total_steps=total_steps,
        )

    def update_learning_rate(self, iter):
        for param_group in self._optimizer.param_groups:
            if param_group["name"] == "camera_R":
                param_group["lr"] = self.R_lr_fn(iter)
            elif param_group["name"] == "camera_T":
                param_group["lr"] = self.T_lr_fn(iter)
            else:
                raise NameError(f"Unknown param_group {param_group['name']}")

    def state_dict(self):
        return self._optimizer.state_dict()

    def zero_grad(self, set_to_none):
        return self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self):
        self._optimizer.step()
