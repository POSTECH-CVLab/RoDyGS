# The MIT License (MIT)

# Copyright (c) 2024 Yoonwoo Jeong, Junmyeong Lee, Hoseung Choi, and Minsu Cho (POSTECH)

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import torch
import torch.nn as nn

from src.utils.point_utils import BasicPointCloud

from .rodygs_static import StaticRoDyGS


class DynRoDyGS(StaticRoDyGS):

    def __init__(
        self,
        sh_degree: int,
        deform_netwidth: int,
        deform_t_emb_multires: int,
        deform_t_log_sampling: bool,
        num_basis: int,
        isotropic: bool = False,
        inverse_motion: bool = False,
        activation="gelu",
    ):
        super(DynRoDyGS, self).__init__(sh_degree, isotropic)
        self.deform_netwidth = deform_netwidth
        self.deform_t_emb_multires = deform_t_emb_multires
        self.deform_t_log_sampling = deform_t_log_sampling
        self.num_basis = num_basis

        self._deform_network = None
        self._motion_coeff = nn.Parameter(torch.randn(0, 0, 0), requires_grad=True)
        self._time_embeddings = None
        self.inverse_motion = inverse_motion

        self.timetokey = lambda time: int(torch.trunc(torch.tensor(time) * 1000).item())
        self.tokeylist = lambda time_list: [self.timetokey(time) for time in time_list]
        self.real_times = None
        self.unique_times = None

        self.activation = activation

        # temporal motion dictionary
        self.temporal_motion_table = None

    @property
    def num_gaussians(self):
        return self._xyz.shape[0]

    def sync_gaussian_to_time_ind(
        self,
    ):
        # sample unique time steps from self._timestep
        if self.real_times is None:
            self.real_times, _ = torch.sort(torch.unique(self.gaussian_to_time))
            self.unique_times = list(
                set(self.tokeylist(self.gaussian_to_time.tolist()))
            )

            # sorting time steps and inverse mapping
            self.unique_times = sorted(self.unique_times)
            self.time_to_ind = {time: ind for ind, time in enumerate(self.unique_times)}
        # get gaussian to time index
        # each gaussian has a special time step(index)
        # tensor

        self.gaussian_to_time_ind = torch.tensor(
            [self.time_to_ind[self.timetokey(time)] for time in self.gaussian_to_time]
        ).cuda()  # [num_gaussians], gaussian -> time ind

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        max_radii2D, xyz_gradient_accum, denom = super().create_from_pcd(
            pcd, spatial_lr_scale
        )
        self._deform_network = MLPBasisNetwork(
            self.deform_netwidth,
            self.num_basis,
            self.deform_t_emb_multires,
            self.deform_t_log_sampling,
            activation=self.activation,
        ).cuda()
        self._motion_coeff = nn.Parameter(
            torch.zeros(self.num_gaussians, 1, self.num_basis).cuda(),
            requires_grad=True,
        )
        self.gaussian_to_time = torch.from_numpy(pcd.time).float().cuda()

        self.sync_gaussian_to_time_ind()
        self._time_embeddings = self._deform_network.t_embedder(
            torch.tensor([0.0]).cuda()
        )
        self._time_batch_embeddings = self._deform_network.batch_embedding(
            self.real_times
        )

        return max_radii2D, xyz_gradient_accum, denom

    def create_from_state_dict(self, state_dict, spatial_lr_scale):
        super().create_from_state_dict(state_dict, spatial_lr_scale)
        self._motion_coeff = nn.Parameter(state_dict["model"]["_motion_coeff"])
        self._deform_network = MLPBasisNetwork(
            self.deform_netwidth,
            self.num_basis,
            self.deform_t_emb_multires,
            self.deform_t_log_sampling,
        ).cuda()
        self._deform_network.load_state_dict(state_dict["model"]["_deform_network"])
        self.gaussian_to_time = state_dict["model"]["_timestep"]
        self.sync_gaussian_to_time_ind()
        self._time_batch_embeddings = self._deform_network.batch_embedding(
            self.real_times
        )

    def get_gaussian_deformation(self, time):
        translation, rotation = self._deform_network(self._motion_coeff, time)
        if self.inverse_motion:
            delta_gaussians = self.get_total_motion_table()  # [timesteps, num_basis, 7]

            # get the inverse motion
            # motion coeff : [num_gaussians, num_basis]
            delta_gaussians = (
                self._motion_coeff @ delta_gaussians[self.gaussian_to_time_ind]
            ).squeeze()

            # sample unique time steps from self._timestep
            translation = translation - delta_gaussians[..., :3]
            rotation = rotation - delta_gaussians[..., 3:]
        scaled_translation = translation * self.spatial_lr_scale

        return scaled_translation, rotation

    def get_total_motion_table(
        self,
    ):
        if self.temporal_motion_table is None:
            self.temporal_motion_table = self._deform_network.batch_inference(
                self._time_batch_embeddings
            ).squeeze()
        return self.temporal_motion_table

    def clean_motion_table(
        self,
    ):
        self.temporal_motion_table = None

    def get_motion_for_times(self, timesteps, time_indices=None):
        if self.temporal_motion_table is None:
            self.get_total_motion_table()

        if time_indices is not None:
            return self.temporal_motion_table[time_indices]

        return self.temporal_motion_table[self.tokeylist(timesteps)]

    def render(
        self,
        viewpoint_camera,
        bg_color: torch.Tensor,
        scaling_modifier=1,
        override_color=None,
        enable_sh_grad=False,
        enable_cov_grad=False,
        use_deform=True,
    ):
        scaled_translation, rotation = (
            self.get_gaussian_deformation(viewpoint_camera.time)
            if use_deform
            else (0, 0)
        )
        return super(DynRoDyGS, self).render(
            viewpoint_camera,
            bg_color,
            scaling_modifier,
            override_color,
            enable_sh_grad,
            enable_cov_grad,
            scaled_translation,
            rotation,
        )


class TimestepEmbedder(nn.Module):

    periodic_funcs = [torch.sin, torch.cos]
    include_input: bool = True

    def __init__(self, emb_multires, input_dims, log_sampling):
        super(TimestepEmbedder, self).__init__()
        self.num_freqs = emb_multires
        self.max_freq_log2 = emb_multires - 1
        self.input_dims = input_dims
        self.log_sampling = log_sampling

    def forward(self, timestep: float):

        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, self.num_freqs - 1, self.num_freqs)
        else:
            freq_bands = torch.linspace(
                1.0, 2.0 ** (self.num_freqs - 1), self.num_freqs
            )

        freq_bands = freq_bands * np.pi
        emb = []
        if self.include_input:
            emb.append(timestep)

        for freq in freq_bands:
            for func in self.periodic_funcs:
                emb.append(func(timestep * freq))

        return torch.stack(emb)


class MLPMotionBasis(nn.Module):

    def __init__(self, input_dim, output_dim, activation):
        super(MLPMotionBasis, self).__init__()

        self.basis = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            activation,
            nn.Linear(input_dim // 2, output_dim),
        )

        for module in self.basis.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=1e-2)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.basis(x)


class MLPBasisNetwork(nn.Module):

    time_input_dim: int = 1
    trans_dim: int = 3
    rot_dim: int = 4

    def __init__(
        self, netwidth, num_basis, t_emb_multires, t_log_sampling, activation="gelu"
    ):
        super(MLPBasisNetwork, self).__init__()

        self.netwidth = netwidth
        self.num_basis = num_basis
        self.t_embed_dim = (
            t_emb_multires * self.time_input_dim * 2 + self.time_input_dim
        )
        self.t_embedder = TimestepEmbedder(
            t_emb_multires, self.time_input_dim, t_log_sampling
        )

        self.activation = (
            nn.GELU() if activation.lower() != "relu" else nn.ReLU(inplace=False)
        )
        print("initiatlize network activation to ", self.activation)
        self.timenet = nn.Sequential(
            nn.Linear(self.t_embed_dim, netwidth),
            self.activation,
            nn.Linear(netwidth, netwidth),
            self.activation,
            nn.Linear(netwidth, netwidth // 2),
            self.activation,
        )

        for module in self.timenet.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=1e-2)
                nn.init.constant_(module.bias, 0)

        self.basis_xyz = nn.ModuleList(
            [
                MLPMotionBasis(
                    netwidth // 2, self.trans_dim + self.rot_dim, self.activation
                )
                for _ in range(num_basis)
            ]
        )

    def batch_embedding(self, timesteps):
        t_embs = torch.stack([self.t_embedder(t) for t in timesteps]).cuda().squeeze()
        # times = torch.tensor([0.1 for _ in range(64)]).cuda().view(-1, 1)
        # t_embed = torch.stack([motion_network.t_embedder(time) for time in times]).cuda().squeeze()
        return t_embs

    def batch_inference(self, t_embs):
        out = self.timenet(t_embs)

        motions = []
        for basis in self.basis_xyz:
            motion = basis(out)
            motions.append(motion)
        tot_motion_basis = torch.stack(motions)
        tot_motion_basis = torch.transpose(tot_motion_basis, 0, 1)

        return tot_motion_basis

    def forward(self, coeff, timestep):
        t_emb = self.t_embedder(timestep)
        out = self.timenet(t_emb)

        motions = []

        for basis in self.basis_xyz:
            motion = basis(out)
            motions.append(motion)
        tot_motion_basis = torch.stack(motions)

        coeff = torch.squeeze(coeff)
        tot_motion = torch.squeeze(coeff @ tot_motion_basis)

        translation, rotation = (
            tot_motion[..., : self.trans_dim],
            tot_motion[..., self.trans_dim :],
        )

        return translation, rotation
