from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from tdmpc2.runtime import build_mlp, orthogonal_init


@contextmanager
def freeze_parameters(modules: list[nn.Module]):
    previous: list[list[bool]] = []
    for module in modules:
        flags = [parameter.requires_grad for parameter in module.parameters()]
        previous.append(flags)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    try:
        yield
    finally:
        for module, flags in zip(modules, previous):
            for parameter, flag in zip(module.parameters(), flags):
                parameter.requires_grad_(flag)


class SimNorm(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = int(groups)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-1] % self.groups != 0:
            raise ValueError(f"Latent dim {tensor.shape[-1]} must be divisible by simnorm groups {self.groups}.")
        group_dim = tensor.shape[-1] // self.groups
        reshaped = tensor.view(*tensor.shape[:-1], self.groups, group_dim)
        normalized = F.softmax(reshaped, dim=-1)
        return normalized.view(*tensor.shape[:-1], tensor.shape[-1])


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        self.net = build_mlp(input_dim, hidden_dims, output_dim)
        self.apply(orthogonal_init)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.net(tensor)


class StateEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: list[int], latent_dim: int, simnorm_groups: int):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims, latent_dim)
        self.norm = SimNorm(simnorm_groups)
        self.apply(orthogonal_init)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(observation))


class LatentDynamics(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dims: list[int], simnorm_groups: int):
        super().__init__()
        self.delta = ProjectionHead(latent_dim + action_dim, hidden_dims, latent_dim)
        self.norm = SimNorm(simnorm_groups)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        next_latent = latent + self.delta(torch.cat([latent, action], dim=-1))
        return self.norm(next_latent)


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        action_dim: int,
        min_std: float = 0.05,
        max_std: float = 1.0,
    ):
        super().__init__()
        trunk_dims = hidden_dims[:-1] if hidden_dims else []
        last_dim = hidden_dims[-1] if hidden_dims else latent_dim
        self.trunk = build_mlp(latent_dim, trunk_dims, last_dim)
        self.mean_head = nn.Linear(last_dim, action_dim)
        self.std_head = nn.Linear(last_dim, action_dim)
        self.min_std = min_std
        self.max_std = max_std
        self.apply(orthogonal_init)

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(latent)
        mean = self.mean_head(hidden)
        std = F.softplus(self.std_head(hidden) + 2.0) + self.min_std
        std = std.clamp(self.min_std, self.max_std)
        return mean, std

    def sample(self, latent: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self(latent)
        dist = torch.distributions.Normal(mean, std)
        raw_action = mean if deterministic else dist.rsample()
        action = torch.tanh(raw_action)
        correction = torch.log1p(-action.pow(2) + 1e-6)
        log_prob = dist.log_prob(raw_action) - correction
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action, log_prob, entropy


class TwinQ(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.q1 = ProjectionHead(latent_dim + action_dim, hidden_dims, 1)
        self.q2 = ProjectionHead(latent_dim + action_dim, hidden_dims, 1)

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([latent, action], dim=-1)
        return self.q1(features), self.q2(features)


def copy_module(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)
