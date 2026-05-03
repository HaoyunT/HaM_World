from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from sac.runtime import build_mlp, count_parameters, orthogonal_init, resolve_device, soft_update


def _atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    clamped = x.clamp(min=-(1.0 - eps), max=1.0 - eps)
    return 0.5 * (torch.log1p(clamped) - torch.log1p(-clamped))


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.backbone = build_mlp(obs_dim, hidden_dims, action_dim * 2)
        self.apply(orthogonal_init)
        self._scale_last_layer(self.backbone, gain=0.01)

    @staticmethod
    def _scale_last_layer(module: nn.Sequential, gain: float) -> None:
        for layer in reversed(module):
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain)
                nn.init.zeros_(layer.bias)
                return

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.backbone(observations).chunk(2, dim=-1)
        log_std = log_std.clamp(min=-5.0, max=2.0)
        return mean, log_std


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.net = build_mlp(obs_dim + action_dim, hidden_dims, 1)
        self.apply(orthogonal_init)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([observations, actions], dim=-1))


@dataclass
class SACUpdate:
    critic_loss: float
    actor_loss: float
    alpha_loss: float
    alpha: float
    q1_mean: float
    q2_mean: float
    target_q_mean: float
    log_prob_mean: float
    reward_mean: float


class SACAgent:
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
        model_cfg = config.get("model", {})
        actor_hidden_dims = list(model_cfg.get("actor_hidden_dims", [256, 256]))
        critic_hidden_dims = list(model_cfg.get("critic_hidden_dims", [256, 256]))

        self.actor = SquashedGaussianActor(obs_dim, action_dim, actor_hidden_dims).to(self.device)
        self.critic1 = Critic(obs_dim, action_dim, critic_hidden_dims).to(self.device)
        self.critic2 = Critic(obs_dim, action_dim, critic_hidden_dims).to(self.device)
        self.target_critic1 = Critic(obs_dim, action_dim, critic_hidden_dims).to(self.device)
        self.target_critic2 = Critic(obs_dim, action_dim, critic_hidden_dims).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        optim_cfg = config.get("optim", {})
        actor_lr = float(optim_cfg.get("actor_lr", optim_cfg.get("lr", 3e-4)))
        critic_lr = float(optim_cfg.get("critic_lr", optim_cfg.get("lr", 3e-4)))
        alpha_lr = float(optim_cfg.get("alpha_lr", optim_cfg.get("lr", 3e-4)))
        weight_decay = float(optim_cfg.get("weight_decay", 0.0))

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=weight_decay)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr,
            weight_decay=weight_decay,
        )

        algo_cfg = config.get("algorithm", {})
        self.discount = float(config["training"].get("discount", 0.99))
        self.tau = float(optim_cfg.get("target_tau", algo_cfg.get("target_tau", 0.01)))
        self.actor_update_interval = int(algo_cfg.get("actor_update_interval", 1))
        self.target_update_interval = int(algo_cfg.get("target_update_interval", 1))
        self.learnable_temperature = bool(algo_cfg.get("learnable_temperature", True))
        init_temperature = float(algo_cfg.get("init_temperature", 0.1))
        target_entropy_scale = float(algo_cfg.get("target_entropy_scale", 1.0))
        self.target_entropy = -target_entropy_scale * float(action_dim)

        self.log_alpha = torch.tensor(math.log(init_temperature), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @property
    def num_parameters(self) -> int:
        return (
            count_parameters(self.actor)
            + count_parameters(self.critic1)
            + count_parameters(self.critic2)
        )

    def _scale_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        return self.action_low + (normalized_action + 1.0) * 0.5 * (self.action_high - self.action_low)

    def _distribution(self, observations: torch.Tensor) -> Normal:
        mean, log_std = self.actor(observations)
        return Normal(mean, log_std.exp())

    def _sample_action_and_log_prob(
        self,
        observations: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._distribution(observations)
        pre_tanh = dist.mean if deterministic else dist.rsample()
        normalized_action = torch.tanh(pre_tanh)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - normalized_action.square() + 1e-6)
        return normalized_action, log_prob.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def act(self, observation: np.ndarray, eval_mode: bool = False) -> tuple[np.ndarray, np.ndarray]:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        normalized_action, _ = self._sample_action_and_log_prob(obs_tensor, deterministic=eval_mode)
        action = self._scale_action(normalized_action)
        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            normalized_action.squeeze(0).cpu().numpy().astype(np.float32),
        )

    def update(self, batch: dict[str, torch.Tensor], step: int) -> SACUpdate:
        observations = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_observations = batch["next_observations"].to(self.device)
        dones = batch["dones"].to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self._sample_action_and_log_prob(next_observations)
            next_q1 = self.target_critic1(next_observations, next_actions)
            next_q2 = self.target_critic2(next_observations, next_actions)
            target_q = torch.min(next_q1, next_q2) - self.alpha.detach() * next_log_probs
            target_value = rewards + self.discount * (1.0 - dones) * target_q

        current_q1 = self.critic1(observations, actions)
        current_q2 = self.critic2(observations, actions)
        critic_loss = 0.5 * ((current_q1 - target_value).square().mean() + (current_q2 - target_value).square().mean())

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        alpha_loss = torch.tensor(0.0, device=self.device)
        log_prob_mean = torch.tensor(0.0, device=self.device)

        if step % self.actor_update_interval == 0:
            sampled_actions, log_probs = self._sample_action_and_log_prob(observations)
            q1_pi = self.critic1(observations, sampled_actions)
            q2_pi = self.critic2(observations, sampled_actions)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (self.alpha.detach() * log_probs - q_pi).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            log_prob_mean = log_probs.mean()
            if self.learnable_temperature:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optimizer.step()

        if step % self.target_update_interval == 0:
            soft_update(self.target_critic1, self.critic1, self.tau)
            soft_update(self.target_critic2, self.critic2, self.tau)

        return SACUpdate(
            critic_loss=float(critic_loss.item()),
            actor_loss=float(actor_loss.item()),
            alpha_loss=float(alpha_loss.item()),
            alpha=float(self.alpha.detach().item()),
            q1_mean=float(current_q1.mean().item()),
            q2_mean=float(current_q2.mean().item()),
            target_q_mean=float(target_value.mean().item()),
            log_prob_mean=float(log_prob_mean.item()),
            reward_mean=float(rewards.mean().item()),
        )

    def sample_random_action(self) -> np.ndarray:
        normalized_action = torch.rand_like(self.action_low) * 2.0 - 1.0
        action = self._scale_action(normalized_action)
        return action.cpu().numpy().astype(np.float32)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        normalized = 2.0 * (action_tensor - self.action_low) / (self.action_high - self.action_low) - 1.0
        return normalized.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.target_critic1.load_state_dict(state["target_critic1"])
        self.target_critic2.load_state_dict(state["target_critic2"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        log_alpha = state.get("log_alpha")
        if log_alpha is not None:
            self.log_alpha.data.copy_(log_alpha.to(self.device))
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
