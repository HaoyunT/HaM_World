from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.utils as nn_utils

from tdmpc2.planner import TDMPC2Planner
from tdmpc2.runtime import ReplayBatch, resolve_device
from tdmpc2.world_model import TDMPC2WorldModel


@dataclass
class UpdateOutput:
    model_loss: float
    policy_loss: float
    consistency_loss: float
    reward_loss: float
    q_loss: float
    q_mean: float
    td_target_mean: float
    planner_value: float
    policy_std: float
    model_grad_norm: float
    policy_grad_norm: float
    extra_metrics: dict[str, float] = field(default_factory=dict)


class TDMPC2Agent:
    def __init__(self, config: dict, obs_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray):
        self.config = config
        self.device = resolve_device(config["experiment"]["device"])
        self.world_model = TDMPC2WorldModel(config, obs_dim, action_dim).to(self.device)
        self.planner = TDMPC2Planner(config)

        optim_cfg = config["optim"]
        policy_cfg = config.get("policy", {})
        self.model_optimizer = torch.optim.AdamW(
            self.world_model.model_parameters(),
            lr=float(optim_cfg.get("model_lr", optim_cfg.get("lr", 3e-4))),
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        )
        self.policy_optimizer = torch.optim.AdamW(
            self.world_model.policy_parameters(),
            lr=float(policy_cfg.get("lr", optim_cfg.get("lr", 3e-4))),
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        )
        self.grad_clip = float(optim_cfg.get("grad_clip", 10.0))
        self.target_tau = float(optim_cfg.get("target_tau", 0.01))
        self.exploration_std = float(policy_cfg.get("exploration_std", 0.2))

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self.action_scale = 0.5 * (self.action_high - self.action_low)
        self.action_bias = 0.5 * (self.action_high + self.action_low)
        self.prev_mean: torch.Tensor | None = None
        self.last_planner_value = torch.tensor(0.0, device=self.device)

    def reset(self) -> None:
        self.prev_mean = None
        self.last_planner_value = torch.tensor(0.0, device=self.device)

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        scale = self.action_scale.clamp_min(1e-6)
        return ((action - self.action_bias) / scale).clamp(-1.0, 1.0)

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (self.action_bias + self.action_scale * action).clamp(self.action_low, self.action_high)

    def _normalize_batch(self, batch: ReplayBatch) -> ReplayBatch:
        normalized_actions = self._normalize_action(batch.actions)
        return ReplayBatch(
            observations=batch.observations,
            actions=normalized_actions,
            rewards=batch.rewards,
            discounts=batch.discounts,
            dones=batch.dones,
        )

    @torch.no_grad()
    def act(self, observation: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        latent = self.world_model.encode(obs_tensor)
        action, mean, planner_value = self.planner.plan(self.world_model, latent, self.prev_mean)
        self.prev_mean = self.planner.shift_mean(mean.detach())
        self.last_planner_value = planner_value.detach()
        if not eval_mode:
            action = (action + torch.randn_like(action) * self.exploration_std).clamp(-1.0, 1.0)
        env_action = self._denormalize_action(action)
        return env_action.cpu().numpy().astype(np.float32)

    def update(self, batch: ReplayBatch) -> UpdateOutput:
        batch = self._normalize_batch(batch)

        self.model_optimizer.zero_grad(set_to_none=True)
        model_losses = self.world_model.compute_model_loss(batch)
        model_losses.total.backward()
        model_grad_norm = float(nn_utils.clip_grad_norm_(list(self.world_model.model_parameters()), self.grad_clip))
        self.model_optimizer.step()

        self.policy_optimizer.zero_grad(set_to_none=True)
        with self.world_model.freeze_for_policy():
            policy_losses = self.world_model.compute_policy_loss(batch)
        policy_losses.policy_loss.backward()
        policy_grad_norm = float(nn_utils.clip_grad_norm_(list(self.world_model.policy_parameters()), self.grad_clip))
        self.policy_optimizer.step()

        self.world_model.update_targets(self.target_tau)
        diagnostics = self.world_model.compute_diagnostics(batch)

        return UpdateOutput(
            model_loss=float(model_losses.total.item()),
            policy_loss=float(policy_losses.policy_loss.item()),
            consistency_loss=float(model_losses.consistency_loss.item()),
            reward_loss=float(model_losses.reward_loss.item()),
            q_loss=float(model_losses.q_loss.item()),
            q_mean=float(model_losses.q_mean.item()),
            td_target_mean=float(model_losses.td_target_mean.item()),
            planner_value=float(self.last_planner_value.item()),
            policy_std=float(policy_losses.policy_std.item()),
            model_grad_norm=model_grad_norm,
            policy_grad_norm=policy_grad_norm,
            extra_metrics={f"train/{key}": float(value.item()) for key, value in diagnostics.items()},
        )

    def state_dict(self) -> dict:
        return {
            "model": self.world_model.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "prev_mean": None if self.prev_mean is None else self.prev_mean.detach().cpu(),
            "last_planner_value": self.last_planner_value.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.world_model.load_state_dict(state["model"])
        model_optimizer_state = state.get("model_optimizer")
        if model_optimizer_state is not None:
            self.model_optimizer.load_state_dict(model_optimizer_state)
        policy_optimizer_state = state.get("policy_optimizer")
        if policy_optimizer_state is not None:
            self.policy_optimizer.load_state_dict(policy_optimizer_state)
        self.prev_mean = None if state.get("prev_mean") is None else state["prev_mean"].to(self.device)
        self.last_planner_value = state.get("last_planner_value", torch.tensor(0.0)).to(self.device)
