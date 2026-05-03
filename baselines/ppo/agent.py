from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from ppo.runtime import build_mlp, count_parameters, orthogonal_init, resolve_device


def _atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    clamped = x.clamp(min=-(1.0 - eps), max=1.0 - eps)
    return 0.5 * (torch.log1p(clamped) - torch.log1p(-clamped))


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, config: dict[str, Any]):
        super().__init__()
        model_cfg = config.get("model", {})
        actor_hidden_dims = list(model_cfg.get("actor_hidden_dims", [256, 256]))
        critic_hidden_dims = list(model_cfg.get("critic_hidden_dims", [256, 256]))
        init_log_std = float(model_cfg.get("init_log_std", -0.5))

        self.actor = build_mlp(obs_dim, actor_hidden_dims, action_dim)
        self.critic = build_mlp(obs_dim, critic_hidden_dims, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), init_log_std, dtype=torch.float32))

        self.apply(orthogonal_init)
        self._scale_last_layer(self.actor, gain=0.01)
        self._scale_last_layer(self.critic, gain=1.0)

    @staticmethod
    def _scale_last_layer(module: nn.Sequential, gain: float) -> None:
        for layer in reversed(module):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain)
                nn.init.zeros_(layer.bias)
                return

    def distribution(self, observations: torch.Tensor) -> Normal:
        mean = self.actor(observations)
        log_std = self.log_std.clamp(min=-5.0, max=2.0)
        std = torch.exp(log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(observations)


@dataclass
class PPOUpdate:
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    learning_rate: float
    explained_variance: float
    grad_norm: float
    value_mean: float
    advantage_mean: float
    return_mean: float


class PPOAgent:
    def __init__(
        self,
        config: dict[str, Any],
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ):
        self.config = config
        self.device = resolve_device(config["experiment"]["device"])
        self.model = PPOActorCritic(obs_dim, action_dim, config).to(self.device)

        optim_cfg = config.get("optim", {})
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(optim_cfg.get("lr", 3e-4)),
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
            eps=float(optim_cfg.get("eps", 1e-5)),
        )

        algo_cfg = config.get("algorithm", {})
        self.clip_ratio = float(algo_cfg.get("clip_ratio", 0.2))
        self.value_coef = float(algo_cfg.get("value_coef", 0.5))
        self.entropy_coef = float(algo_cfg.get("entropy_coef", 0.0))
        self.target_kl = float(algo_cfg.get("target_kl", 0.05))
        self.normalize_advantages = bool(algo_cfg.get("normalize_advantages", True))
        self.max_grad_norm = float(optim_cfg.get("grad_clip", 0.5))
        self.update_epochs = int(config["training"].get("update_epochs", 10))
        self.minibatch_size = int(config["training"].get("minibatch_size", 256))

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)

    @property
    def num_parameters(self) -> int:
        return count_parameters(self.model)

    def _scale_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        return self.action_low + (normalized_action + 1.0) * 0.5 * (self.action_high - self.action_low)

    def _log_prob_from_normalized_action(self, dist: Normal, normalized_action: torch.Tensor) -> torch.Tensor:
        pre_tanh = _atanh(normalized_action)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - normalized_action.square() + 1e-6)
        return log_prob.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray,
        eval_mode: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.model.distribution(obs_tensor)
        pre_tanh = dist.mean if eval_mode else dist.rsample()
        normalized_action = torch.tanh(pre_tanh)
        log_prob = self._log_prob_from_normalized_action(dist, normalized_action)
        value = self.model.value(obs_tensor)
        action = self._scale_action(normalized_action)
        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            normalized_action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def predict_value(self, observation: np.ndarray) -> float:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.model.value(obs_tensor).item())

    def _evaluate_actions(
        self,
        observations: torch.Tensor,
        normalized_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.model.distribution(observations)
        log_prob = self._log_prob_from_normalized_action(dist, normalized_actions)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        values = self.model.value(observations)
        return log_prob, entropy, values

    def update(self, batch: dict[str, torch.Tensor]) -> PPOUpdate:
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        old_log_probs = batch["log_probs"].to(self.device)
        returns = batch["returns"].to(self.device)
        advantages = batch["advantages"].to(self.device)
        old_values = batch["values"].to(self.device)

        if self.normalize_advantages:
            advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)

        batch_size = observations.shape[0]
        minibatch_size = min(self.minibatch_size, batch_size)
        indices = torch.arange(batch_size, device=self.device)

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        last_total_loss = 0.0
        last_approx_kl = 0.0
        last_clip_fraction = 0.0
        last_grad_norm = 0.0

        for _ in range(self.update_epochs):
            permutation = indices[torch.randperm(batch_size, device=self.device)]
            for start in range(0, batch_size, minibatch_size):
                mb_indices = permutation[start : start + minibatch_size]
                mb_obs = observations[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_old_values = old_values[mb_indices]

                new_log_probs, entropy, values = self._evaluate_actions(mb_obs, mb_actions)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = log_ratio.exp()

                unclipped = ratio * mb_advantages
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_pred_clipped = mb_old_values + (values - mb_old_values).clamp(-self.clip_ratio, self.clip_ratio)
                value_losses = (values - mb_returns).square()
                value_losses_clipped = (value_pred_clipped - mb_returns).square()
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                entropy_term = entropy.mean()
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_term

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean()
                    clip_fraction = (torch.abs(ratio - 1.0) > self.clip_ratio).float().mean()

                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy = float(entropy_term.item())
                last_total_loss = float(total_loss.item())
                last_approx_kl = float(approx_kl.item())
                last_clip_fraction = float(clip_fraction.item())
                last_grad_norm = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

            if self.target_kl > 0.0 and last_approx_kl > 1.5 * self.target_kl:
                break

        returns_np = returns.detach().cpu().numpy().reshape(-1)
        old_values_np = old_values.detach().cpu().numpy().reshape(-1)
        explained_variance = 1.0 - np.var(returns_np - old_values_np) / max(np.var(returns_np), 1e-6)

        return PPOUpdate(
            total_loss=last_total_loss,
            policy_loss=last_policy_loss,
            value_loss=last_value_loss,
            entropy=last_entropy,
            approx_kl=last_approx_kl,
            clip_fraction=last_clip_fraction,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            explained_variance=float(explained_variance),
            grad_norm=last_grad_norm,
            value_mean=float(old_values.mean().item()),
            advantage_mean=float(advantages.mean().item()),
            return_mean=float(returns.mean().item()),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
