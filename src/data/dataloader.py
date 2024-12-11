# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
from torch.utils.data import Dataset


class WarmupDataLoader:
    # Pick a random viewpoint over selectable viewpoints.
    # Selectable viewpoints are incrementally increasing as the `register_frame` is called.
    is_incremental: bool = True
    warmup_rate = 20

    def __init__(self, dataset: Dataset, init_num_frames: int, return_idx=False):
        self.dataset = dataset
        self.curr_num_frames = init_num_frames
        self.total_len = len(dataset)
        self.return_idx = return_idx

        num_list = int(np.ceil(self.num_iterations / self.total_len))
        cat_list = [np.random.permutation(self.total_len) for _ in range(num_list)]
        self.idx_list = [i for idx, i in cat_list if idx % self.warmup_rate == 0]

    def __iter__(self):
        for random_idx in self.idx_list:
            if self.return_idx:
                yield random_idx, self.dataset[random_idx]
            else:
                yield self.dataset[random_idx]

    def register_all_frame(self):
        num_list = int(np.ceil(self.num_iterations / self.total_len))
        cat_list = [np.random.permutation(self.total_len) for _ in range(num_list)]
        remainder = self.num_iterations % self.total_len
        if remainder != 0:
            cat_list[-1] = cat_list[-1][:remainder]
        self.idx_list = np.concatenate(cat_list)


class PermutationSingleDataLoader:
    is_incremental: bool = False

    def __init__(self, dataset: Dataset, num_iterations: int, return_idx: bool = False):
        self.dataset = dataset
        self.total_len = len(dataset)
        self.return_idx = return_idx
        self.num_iterations = num_iterations
        self.idx_list = self.get_permuted_idx_list()

    def get_permuted_idx_list(self):
        num_list = int(np.ceil(self.num_iterations / self.total_len))
        cat_list = [np.random.permutation(self.total_len) for _ in range(num_list)]
        remainder = self.num_iterations % self.total_len
        if remainder != 0:
            cat_list[-1] = cat_list[-1][:remainder]
        idx_list = np.concatenate(cat_list)
        return idx_list

    def __iter__(self):
        for random_idx in self.idx_list:
            if self.return_idx:
                yield random_idx, self.dataset[random_idx]
            else:
                yield self.dataset[random_idx]


class SequentialSingleDataLoader:
    # Call all viewpoints in a sequential manner.
    is_incremental: bool = False

    def __init__(self, dataset: Dataset, return_idx: bool = False, **kwargs):
        self.dataset = dataset
        self.total_len = len(dataset)
        self.return_idx = return_idx

    def __len__(self):
        return self.total_len

    def __iter__(self):
        for idx in range(self.total_len):
            if self.return_idx:
                yield idx, self.dataset[idx]
            else:
                yield self.dataset[idx]
