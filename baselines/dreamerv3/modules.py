from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamerv3.runtime import build_mlp, orthogonal_init


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor
    logits: torch.Tensor


def symlog(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sign(tensor) * torch.log1p(tensor.abs())


def symexp(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sign(tensor) * (torch.exp(tensor.abs()) - 1.0)


def twohot_encode(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Soft two-hot encoding of scalar values onto a fixed bin set."""
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
    """Per-example cross-entropy between logits and twohot-encoded symlog targets."""
    target_dist = twohot_encode(symlog(targets), bins)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1, keepdim=True)


def twohot_loss(logits: torch.Tensor, targets: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss between predicted logits and twohot-encoded symlog targets."""
    return twohot_cross_entropy(logits, targets, bins).mean()


def twohot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Expected value from predicted distribution, decoded from symlog space."""
    probs = torch.softmax(logits, dim=-1)
    return symexp((probs * bins).sum(dim=-1, keepdim=True))


def soft_categorical_loss(logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
    """Cross-entropy to a target categorical distribution parameterized by logits."""
    target_probs = torch.softmax(target_logits.detach(), dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1, keepdim=True)


class PercentileNormalizer(nn.Module):
    """Running percentile normalizer for DreamerV3 return scaling."""

    def __init__(self, low: float = 5.0, high: float = 95.0, decay: float = 0.99):
        super().__init__()
        self.low = float(low)
        self.high = float(high)
        self.decay = float(decay)
        self.register_buffer("_lo", torch.tensor(0.0))
        self.register_buffer("_hi", torch.tensor(1.0))
        self.register_buffer("_initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def update(self, returns: torch.Tensor) -> None:
        flat = returns.detach().float().reshape(-1)
        lo = torch.quantile(flat, self.low / 100.0)
        hi = torch.quantile(flat, self.high / 100.0)
        if not bool(self._initialized.item()):
            self._lo.copy_(lo)
            self._hi.copy_(hi)
            self._initialized.fill_(True)
        else:
            momentum = 1.0 - self.decay
            self._lo.lerp_(lo, momentum)
            self._hi.lerp_(hi, momentum)

    def scale(self) -> torch.Tensor:
        return torch.clamp(self._hi - self._lo, min=1.0)

    def normalize(self, returns: torch.Tensor, center: bool = False) -> torch.Tensor:
        offset = self._lo if center else 0.0
        return (returns - offset) / self.scale()

    @torch.no_grad()
    def update_and_normalize(self, returns: torch.Tensor, center: bool = False) -> torch.Tensor:
        self.update(returns)
        return self.normalize(returns, center=center)


def flatten_stoch(stoch: torch.Tensor) -> torch.Tensor:
    return stoch.reshape(*stoch.shape[:-2], stoch.shape[-2] * stoch.shape[-1])


def stack_states(states: list[RSSMState], dim: int = 1) -> RSSMState:
    return RSSMState(
        deter=torch.stack([state.deter for state in states], dim=dim),
        stoch=torch.stack([state.stoch for state in states], dim=dim),
        logits=torch.stack([state.logits for state in states], dim=dim),
    )


def detach_state(state: RSSMState) -> RSSMState:
    return RSSMState(
        deter=state.deter.detach(),
        stoch=state.stoch.detach(),
        logits=state.logits.detach(),
    )


def slice_state(state: RSSMState, start: int | None = None, stop: int | None = None) -> RSSMState:
    return RSSMState(
        deter=state.deter[:, start:stop],
        stoch=state.stoch[:, start:stop],
        logits=state.logits[:, start:stop],
    )


def flatten_time(state: RSSMState) -> RSSMState:
    stoch_size, classes = state.stoch.shape[-2:]
    return RSSMState(
        deter=state.deter.reshape(-1, state.deter.shape[-1]),
        stoch=state.stoch.reshape(-1, stoch_size, classes),
        logits=state.logits.reshape(-1, stoch_size, classes),
    )


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


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
        super().__init__()
        self.net = build_mlp(input_dim, hidden_dims, output_dim)
        self.apply(orthogonal_init)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.net(tensor)


class StateEncoder(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: list[int], embed_dim: int):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_dims, embed_dim)
        self.apply(orthogonal_init)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


class DiscreteLatentHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], stoch_size: int, classes: int):
        super().__init__()
        self.stoch_size = stoch_size
        self.classes = classes
        self.net = build_mlp(input_dim, hidden_dims, stoch_size * classes)
        self.apply(orthogonal_init)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        logits = self.net(tensor)
        return logits.view(*logits.shape[:-1], self.stoch_size, self.classes)


class DiscreteRSSM(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        deter_dim: int,
        stoch_size: int,
        classes: int,
        hidden_dims: list[int],
        unimix_ratio: float = 0.01,
    ):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_size = stoch_size
        self.classes = classes
        self.unimix_ratio = float(unimix_ratio)
        stoch_dim = stoch_size * classes

        self.input_proj = nn.Linear(stoch_dim + action_dim, deter_dim)
        self.gru = nn.GRUCell(deter_dim, deter_dim)
        self.prior_head = DiscreteLatentHead(deter_dim, hidden_dims, stoch_size, classes)
        self.posterior_head = DiscreteLatentHead(deter_dim + embed_dim, hidden_dims, stoch_size, classes)
        self.apply(orthogonal_init)

    def _apply_unimix(self, logits: torch.Tensor) -> torch.Tensor:
        if self.unimix_ratio <= 0.0:
            return logits
        probs = torch.softmax(logits, dim=-1)
        uniform = torch.full_like(probs, 1.0 / self.classes)
        probs = (1.0 - self.unimix_ratio) * probs + self.unimix_ratio * uniform
        return torch.log(probs.clamp_min(1e-8))

    def initial(self, batch_size: int, device: torch.device) -> RSSMState:
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        logits = torch.zeros(batch_size, self.stoch_size, self.classes, device=device)
        stoch = torch.zeros_like(logits)
        stoch[..., 0] = 1.0
        return RSSMState(deter=deter, stoch=stoch, logits=logits)

    def _sample_stoch(self, logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            index = torch.argmax(logits, dim=-1)
            return F.one_hot(index, num_classes=self.classes).to(dtype=logits.dtype)
        dist = torch.distributions.OneHotCategorical(logits=logits)
        sample = dist.sample()
        probs = dist.probs
        return sample + probs - probs.detach()

    def img_step(self, prev_state: RSSMState, action: torch.Tensor, deterministic: bool = False) -> RSSMState:
        hidden = torch.cat([flatten_stoch(prev_state.stoch), action], dim=-1)
        hidden = torch.tanh(self.input_proj(hidden))
        deter = self.gru(hidden, prev_state.deter)
        logits = self._apply_unimix(self.prior_head(deter))
        stoch = self._sample_stoch(logits, deterministic=deterministic)
        return RSSMState(deter=deter, stoch=stoch, logits=logits)

    def obs_step(self, prior_state: RSSMState, embed: torch.Tensor, deterministic: bool = False) -> RSSMState:
        logits = self._apply_unimix(self.posterior_head(torch.cat([prior_state.deter, embed], dim=-1)))
        stoch = self._sample_stoch(logits, deterministic=deterministic)
        return RSSMState(deter=prior_state.deter, stoch=stoch, logits=logits)

    def features(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deter, flatten_stoch(state.stoch)], dim=-1)


class DreamerActor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        action_dim: int,
        min_std: float = 0.1,
        max_std: float = 1.0,
    ):
        super().__init__()
        trunk_dims = hidden_dims[:-1] if hidden_dims else []
        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.trunk = build_mlp(input_dim, trunk_dims, last_dim)
        self.mean_head = nn.Linear(last_dim, action_dim)
        self.std_head = nn.Linear(last_dim, action_dim)
        self.min_std = min_std
        self.max_std = max_std
        self.apply(orthogonal_init)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        mean = self.mean_head(hidden)
        std = F.softplus(self.std_head(hidden) + 2.0) + self.min_std
        std = std.clamp(self.min_std, self.max_std)
        return mean, std

    def sample(self, features: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self(features)
        dist = torch.distributions.Normal(mean, std)
        raw_action = mean if deterministic else dist.rsample()
        action = torch.tanh(raw_action)
        raw_action_detached = raw_action.detach()
        action_detached = torch.tanh(raw_action_detached)
        correction = torch.log1p(-action_detached.pow(2) + 1e-6)
        log_prob = dist.log_prob(raw_action_detached) - correction
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return action, log_prob, entropy


def copy_module(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)


def zero_last_layer(module: nn.Module) -> None:
    for submodule in reversed(list(module.modules())):
        if isinstance(submodule, nn.Linear):
            nn.init.zeros_(submodule.weight)
            nn.init.zeros_(submodule.bias)
            return
