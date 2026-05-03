from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from hamworld.runtime import build_mlp, orthogonal_init


class StateEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims, latent_dim)
        self.apply(orthogonal_init)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        self.net = build_mlp(input_dim, hidden_dims, output_dim)
        self.apply(orthogonal_init)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.net(tensor)


def symlog(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sign(tensor) * torch.log1p(tensor.abs())


def symexp(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sign(tensor) * (torch.exp(tensor.abs()) - 1.0)


def twohot_encode(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    values = values.squeeze(-1)
    clipped = values.clamp(min=float(bins[0].item()), max=float(bins[-1].item()))
    upper = torch.bucketize(clipped, bins).clamp(1, bins.shape[0] - 1)
    lower = upper - 1

    lower_val = bins[lower]
    upper_val = bins[upper]
    upper_weight = (clipped - lower_val) / (upper_val - lower_val).clamp_min(1e-8)
    lower_weight = 1.0 - upper_weight

    encoded = torch.zeros(*clipped.shape, bins.shape[0], device=values.device, dtype=values.dtype)
    encoded.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
    encoded.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return encoded


def twohot_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    target_dist = twohot_encode(symlog(targets), bins)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1, keepdim=True)


def twohot_loss(logits: torch.Tensor, targets: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    return twohot_cross_entropy(logits, targets, bins).mean()


def twohot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return symexp((probs * bins).sum(dim=-1, keepdim=True))


def soft_categorical_loss(logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    target_probs = torch.softmax(target_logits.detach(), dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1, keepdim=True)


def zero_last_layer(module: nn.Module) -> None:
    for submodule in reversed(list(module.modules())):
        if isinstance(submodule, nn.Linear):
            nn.init.zeros_(submodule.weight)
            nn.init.zeros_(submodule.bias)
            return


class ActionPriorHead(nn.Module):
    """Deterministic policy prior used to warm-start planner candidates."""

    def __init__(self, latent_dim: int, hidden_dims: list[int], action_dim: int):
        super().__init__()
        self.net = build_mlp(latent_dim, hidden_dims, action_dim)
        self.apply(orthogonal_init)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class SelectiveScanLayer(nn.Module):
    def __init__(self, model_dim: int, state_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.delta_proj = nn.Linear(model_dim, state_dim)
        self.state_proj = nn.Linear(model_dim, state_dim)
        self.readout_proj = nn.Linear(model_dim, state_dim)
        self.gate_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(state_dim, model_dim)
        self.a_log = nn.Parameter(torch.zeros(state_dim))
        self.ffn = build_mlp(model_dim, [model_dim * 2], model_dim)
        self.apply(orthogonal_init)

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.a_log.numel(), device=device)

    def step(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = self.norm(x)
        delta = F.softplus(self.delta_proj(x)) + 1e-3
        state_update = self.state_proj(x)
        readout = self.readout_proj(x)
        decay = torch.exp(-torch.exp(self.a_log)[None, :] * delta)
        next_state = decay * state + state_update
        mixed = readout * next_state
        gated = torch.sigmoid(self.gate_proj(x)) * self.out_proj(mixed)
        hidden = residual + gated
        hidden = hidden + self.ffn(hidden)
        return hidden, next_state


class MambaMemory(nn.Module):
    """A lightweight selective state-space memory used as the hidden rollout context."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        model_dim: int,
        state_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.input_proj = nn.Linear(latent_dim + action_dim, model_dim)
        self.layers = nn.ModuleList([SelectiveScanLayer(model_dim, state_dim) for _ in range(num_layers)])
        self.output_norm = nn.LayerNorm(model_dim)
        self.apply(orthogonal_init)

    def init_state(self, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        return [layer.init_state(batch_size, device) for layer in self.layers]

    def _stack_state(
        self,
        batch_size: int,
        device: torch.device,
        state: list[torch.Tensor] | None,
    ) -> list[torch.Tensor]:
        if state is None:
            return self.init_state(batch_size, device)
        return state

    def step(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        state: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        batch_size = latent.shape[0]
        state = self._stack_state(batch_size, latent.device, state)
        hidden = self.input_proj(torch.cat([latent, action], dim=-1))
        next_state: list[torch.Tensor] = []
        for layer, layer_state in zip(self.layers, state):
            hidden, updated_state = layer.step(hidden, layer_state)
            next_state.append(updated_state)
        return self.output_norm(hidden), next_state


class CanonicalDynamicsCore(nn.Module):
    """Small planner-facing canonical core on (q, p) with residual updates."""

    def __init__(
        self,
        q_dim: int,
        p_dim: int,
        c_dim: int,
        action_dim: int,
        memory_dim: int,
        hidden_dims: list[int],
        use_control_map: bool = True,
    ):
        super().__init__()
        self.q_dim = q_dim
        self.p_dim = p_dim
        self.action_dim = action_dim
        self.use_control_map = bool(use_control_map)
        input_dim = q_dim + p_dim + c_dim + action_dim + memory_dim
        output_dim = q_dim + p_dim
        self.trunk = build_mlp(input_dim, hidden_dims, hidden_dims[-1] if hidden_dims else output_dim)
        trunk_dim = hidden_dims[-1] if hidden_dims else output_dim
        self.delta_head = nn.Linear(trunk_dim, output_dim)
        self.gate_head = nn.Linear(trunk_dim, output_dim)
        self.control_head = nn.Linear(trunk_dim, p_dim * action_dim) if self.use_control_map else None
        self.apply(orthogonal_init)

    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        c: torch.Tensor,
        action: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.trunk(torch.cat([q, p, c, action, memory], dim=-1))
        delta = torch.tanh(self.delta_head(features)) * torch.sigmoid(self.gate_head(features))
        dq_net, dp_net = torch.split(delta, [self.q_dim, self.p_dim], dim=-1)
        if self.control_head is None:
            control = torch.zeros_like(p)
        else:
            control_matrix = torch.tanh(self.control_head(features)).view(-1, self.p_dim, self.action_dim)
            control = torch.bmm(control_matrix, action.unsqueeze(-1)).squeeze(-1)
        return dq_net, dp_net, control


class AuxResidualUpdater(nn.Module):
    """Residual context updater that keeps task/reward semantics outside the canonical core."""

    def __init__(
        self,
        q_dim: int,
        p_dim: int,
        c_dim: int,
        action_dim: int,
        memory_dim: int,
        hidden_dims: list[int],
    ):
        super().__init__()
        input_dim = q_dim + p_dim + c_dim + action_dim + memory_dim
        self.trunk = build_mlp(input_dim, hidden_dims, hidden_dims[-1] if hidden_dims else c_dim)
        trunk_dim = hidden_dims[-1] if hidden_dims else c_dim
        self.delta_head = nn.Linear(trunk_dim, c_dim)
        self.gate_head = nn.Linear(trunk_dim, c_dim)
        self.apply(orthogonal_init)

    def forward(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
        c: torch.Tensor,
        action: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        features = self.trunk(torch.cat([q, p, c, action, memory], dim=-1))
        return torch.tanh(self.delta_head(features)) * torch.sigmoid(self.gate_head(features))


class HamiltonianHead(nn.Module):
    """Soft energy regularizer only used on the small canonical subspace."""

    def __init__(self, q_dim: int, p_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.net = build_mlp(q_dim + p_dim, hidden_dims, 1)
        self.apply(orthogonal_init)

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([q, p], dim=-1))


def copy_module(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)
