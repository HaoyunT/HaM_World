from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np
import torch
import torch.nn.utils as nn_utils

from hamworld.planner import CEMPlanner
from hamworld.runtime import resolve_device
from hamworld.world_model import CanonicalDynamicsWorldModel


@dataclass
class UpdateOutput:
    total_loss: float
    repr_loss: float
    dyn_loss: float
    reward_loss: float
    value_loss: float
    value_ce_loss: float
    value_slow_loss: float
    reward_pred_mae: float
    reward_pred_mse: float
    value_pred_mae: float
    value_pred_mse: float
    roll_loss: float
    sa_loss: float
    energy_loss: float
    energy_drift: float
    policy_prior_loss: float
    hamiltonian_loss: float
    temp_loss: float
    decouple_loss: float
    c_sparse_loss: float
    delta_q_norm: float
    delta_p_norm: float
    delta_c_norm: float
    q_norm: float
    p_norm: float
    c_norm: float
    latent_norm: float
    control_norm: float
    planner_latency_ms: float
    hamiltonian_grad_norm: float
    rollout_latent_drift: float
    pred_reward_mean: float
    pred_value_mean: float
    one_step_latent_mse: float
    three_step_latent_mse: float
    roll_weight: float
    sa_weight: float
    energy_weight: float
    policy_weight: float
    hamiltonian_weight: float
    temp_weight: float
    decouple_weight: float
    c_weight: float
    alpha: float
    extra_metrics: dict[str, float] = field(default_factory=dict)


class HaMWorldAgent:
    def __init__(self, config: dict, obs_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray):
        self.config = config
        self.device = resolve_device(config["experiment"]["device"])
        self.world_model = CanonicalDynamicsWorldModel(config, obs_dim, action_dim).to(self.device)
        self.base_lr = float(config["optim"]["lr"])
        self.optimizer = torch.optim.AdamW(
            self.world_model.parameters(),
            lr=self.base_lr,
            weight_decay=float(config["optim"].get("weight_decay", 0.0)),
        )
        self.grad_clip = float(config["optim"]["grad_clip"])
        self.encoder_tau = float(config["optim"]["encoder_tau"])
        self.value_tau = float(config["optim"]["value_tau"])
        self.lr_schedule = str(config["optim"].get("lr_schedule", "constant")).lower()
        self.lr_decay_start = float(config["optim"].get("lr_decay_start", 0.0))
        self.lr_decay_end = float(config["optim"].get("lr_decay_end", config["training"]["total_steps"]))
        self.lr_final_scale = float(config["optim"].get("lr_final_scale", 1.0))
        self.planner = CEMPlanner(config)
        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self.memory_state = self.world_model.reset_memory_state(batch_size=1, device=self.device)
        self.last_latent: torch.Tensor | None = None
        self.last_action: torch.Tensor | None = None
        self.prev_plan_mean: torch.Tensor | None = None
        self.action_queue: list[torch.Tensor] = []
        self.exploration_std = float(config["planner"].get("exploration_std", 0.3))
        self.replan_every = max(1, int(config["planner"].get("replan_every", 1)))
        self.last_planner_latency_ms = 0.0
        self.current_step: int | None = None

    def _resolve_schedule_point(self, raw_value: float, total_steps: int) -> float:
        if raw_value <= 1.0:
            return raw_value * float(total_steps)
        return raw_value

    def _scheduled_lr(self, step: int | None) -> float:
        if step is None or self.lr_schedule == "constant":
            return self.base_lr
        total_steps = max(1, int(self.config["training"]["total_steps"]))
        start = self._resolve_schedule_point(self.lr_decay_start, total_steps)
        end = self._resolve_schedule_point(self.lr_decay_end, total_steps)
        if self.lr_schedule == "linear":
            if step <= start:
                scale = 1.0
            elif step >= end:
                scale = self.lr_final_scale
            else:
                progress = (float(step) - start) / max(1e-6, end - start)
                scale = 1.0 + (self.lr_final_scale - 1.0) * progress
            return self.base_lr * scale
        return self.base_lr

    def _apply_lr(self, step: int | None) -> float:
        lr = self._scheduled_lr(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def reset(self) -> None:
        self.memory_state = self.world_model.reset_memory_state(batch_size=1, device=self.device)
        self.last_latent = None
        self.last_action = None
        self.prev_plan_mean = None
        self.action_queue = []
        self.last_planner_latency_ms = 0.0

    def set_training_step(self, step: int | None) -> None:
        self.current_step = None if step is None else max(0, int(step))
        self.world_model.set_schedule_step(self.current_step)

    @torch.no_grad()
    def act(self, observation: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        self.world_model.set_schedule_step(self.current_step)
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        current_latent = self.world_model.encode(obs_tensor)

        if self.last_latent is not None and self.last_action is not None:
            self.memory_state = self.world_model.advance_memory(self.last_latent, self.last_action, self.memory_state)

        if self.action_queue:
            action = self.action_queue.pop(0)
            if self.prev_plan_mean is not None:
                self.prev_plan_mean = self.planner.shift_mean(self.prev_plan_mean)
        else:
            planner_start = time.perf_counter()
            plan_sequence, _ = self.planner.plan(
                self.world_model,
                current_latent,
                self.memory_state,
                self.action_low,
                self.action_high,
                init_mean=self.prev_plan_mean,
            )
            self.last_planner_latency_ms = (time.perf_counter() - planner_start) * 1000.0
            queued = [plan_sequence[index].detach() for index in range(min(self.replan_every, plan_sequence.shape[0]))]
            action = queued[0]
            self.action_queue = queued[1:]
            self.prev_plan_mean = self.planner.shift_mean(plan_sequence.detach())

        if not eval_mode:
            noise = torch.randn_like(action) * self.exploration_std * (self.action_high - self.action_low)
            action = (action + noise).clamp(self.action_low, self.action_high)
        action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0).clamp(self.action_low, self.action_high)

        self.last_latent = current_latent.detach()
        self.last_action = action.unsqueeze(0).detach()
        return action.cpu().numpy()

    def update(self, batch, step: int | None = None) -> UpdateOutput:
        self.set_training_step(step)
        current_lr = self._apply_lr(step)
        self.optimizer.zero_grad(set_to_none=True)
        total_loss, losses = self.world_model.compute_losses(batch, self.config, step=step)
        total_loss.backward()
        nn_utils.clip_grad_norm_(self.world_model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.world_model.update_targets(self.encoder_tau, self.value_tau)
        extra_metric_keys = [
            "latent_mse_k1",
            "latent_mse_k3",
            "latent_mse_k5",
            "latent_mse_k10",
            "latent_mse_k15",
            "rollout_drift_k1",
            "rollout_drift_k3",
            "rollout_drift_k5",
            "rollout_drift_k10",
            "rollout_drift_k15",
            "small_action_fraction",
            "abs_energy_delta_mean",
            "abs_energy_delta_control_mean",
            "abs_energy_delta_sq_mean",
            "control_norm_sq_mean",
            "energy_control_corr",
        ]
        return UpdateOutput(
            total_loss=float(total_loss.item()),
            repr_loss=float(losses["repr_loss"].item()),
            dyn_loss=float(losses["dyn_loss"].item()),
            reward_loss=float(losses["reward_loss"].item()),
            value_loss=float(losses["value_loss"].item()),
            value_ce_loss=float(losses["value_ce_loss"].item()),
            value_slow_loss=float(losses["value_slow_loss"].item()),
            reward_pred_mae=float(losses["reward_pred_mae"].item()),
            reward_pred_mse=float(losses["reward_pred_mse"].item()),
            value_pred_mae=float(losses["value_pred_mae"].item()),
            value_pred_mse=float(losses["value_pred_mse"].item()),
            roll_loss=float(losses["roll_loss"].item()),
            sa_loss=float(losses["sa_loss"].item()),
            energy_loss=float(losses["energy_loss"].item()),
            energy_drift=float(losses["energy_drift"].item()),
            policy_prior_loss=float(losses["policy_prior_loss"].item()),
            hamiltonian_loss=float(losses["hamiltonian_loss"].item()),
            temp_loss=float(losses["temp_loss"].item()),
            decouple_loss=float(losses["decouple_loss"].item()),
            c_sparse_loss=float(losses["c_sparse_loss"].item()),
            delta_q_norm=float(losses["delta_q_norm"].item()),
            delta_p_norm=float(losses["delta_p_norm"].item()),
            delta_c_norm=float(losses["delta_c_norm"].item()),
            q_norm=float(losses["q_norm"].item()),
            p_norm=float(losses["p_norm"].item()),
            c_norm=float(losses["c_norm"].item()),
            latent_norm=float(losses["latent_norm"].item()),
            control_norm=float(losses["control_norm"].item()),
            planner_latency_ms=float(self.last_planner_latency_ms),
            hamiltonian_grad_norm=float(losses["hamiltonian_grad_norm"].item()),
            rollout_latent_drift=float(losses["rollout_latent_drift"].item()),
            pred_reward_mean=float(losses["pred_reward_mean"].item()),
            pred_value_mean=float(losses["pred_value_mean"].item()),
            one_step_latent_mse=float(losses["one_step_latent_mse"].item()),
            three_step_latent_mse=float(losses["three_step_latent_mse"].item()),
            roll_weight=float(losses["roll_weight"].item()),
            sa_weight=float(losses["sa_weight"].item()),
            energy_weight=float(losses["energy_weight"].item()),
            policy_weight=float(losses["policy_weight"].item()),
            hamiltonian_weight=float(losses["hamiltonian_weight"].item()),
            temp_weight=float(losses["temp_weight"].item()),
            decouple_weight=float(losses["decouple_weight"].item()),
            c_weight=float(losses["c_weight"].item()),
            alpha=float(losses["alpha"].item()),
            extra_metrics={
                f"train/{key}": float(losses[key].item())
                for key in extra_metric_keys
                if key in losses
            }
            | {"train/lr": float(current_lr)},
        )

    def state_dict(self) -> dict:
        return {
            "model": self.world_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "memory_state": [tensor.detach().cpu() for tensor in self.memory_state],
            "last_latent": None if self.last_latent is None else self.last_latent.detach().cpu(),
            "last_action": None if self.last_action is None else self.last_action.detach().cpu(),
            "prev_plan_mean": None if self.prev_plan_mean is None else self.prev_plan_mean.detach().cpu(),
            "action_queue": [tensor.detach().cpu() for tensor in self.action_queue],
            "current_step": self.current_step,
        }

    def load_state_dict(self, state: dict) -> None:
        self.world_model.load_state_dict(state["model"])
        optimizer_state = state.get("optimizer")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        memory_state = state.get("memory_state")
        if memory_state is not None:
            self.memory_state = [tensor.to(self.device) for tensor in memory_state]
        else:
            self.memory_state = self.world_model.reset_memory_state(batch_size=1, device=self.device)
        self.last_latent = None if state.get("last_latent") is None else state["last_latent"].to(self.device)
        self.last_action = None if state.get("last_action") is None else state["last_action"].to(self.device)
        self.prev_plan_mean = None if state.get("prev_plan_mean") is None else state["prev_plan_mean"].to(self.device)
        self.action_queue = [tensor.to(self.device) for tensor in state.get("action_queue", [])]
        self.set_training_step(state.get("current_step"))
