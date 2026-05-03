from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.utils as nn_utils

from dreamerv3.modules import RSSMState
from dreamerv3.runtime import ReplayBatch, resolve_device
from dreamerv3.world_model import DreamerV3WorldModel


@dataclass
class UpdateOutput:
    world_model_loss: float
    actor_loss: float
    critic_loss: float
    recon_loss: float
    reward_loss: float
    continue_loss: float
    dyn_kl: float
    rep_kl: float
    policy_std: float
    model_grad_norm: float
    actor_grad_norm: float
    critic_grad_norm: float
    imagine_return: float
    value_mean: float
    prior_entropy: float
    post_entropy: float
    extra_metrics: dict[str, float] = field(default_factory=dict)


class DreamerV3Agent:
    def __init__(self, config: dict, obs_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray):
        self.config = config
        self.device = resolve_device(config["experiment"]["device"])
        self.world_model = DreamerV3WorldModel(config, obs_dim, action_dim).to(self.device)
        optim_cfg = config["optim"]
        behavior_cfg = config.get("behavior", {})
        self.model_optimizer = torch.optim.AdamW(
            self.world_model.world_model_parameters(),
            lr=float(optim_cfg.get("model_lr", optim_cfg.get("lr", 1e-4))),
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.world_model.actor_parameters(),
            lr=float(behavior_cfg.get("actor_lr", optim_cfg.get("lr", 1e-4))),
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.world_model.critic_parameters(),
            lr=float(behavior_cfg.get("critic_lr", optim_cfg.get("lr", 1e-4))),
            weight_decay=float(optim_cfg.get("weight_decay", 0.0)),
        )
        self.grad_clip = float(optim_cfg.get("grad_clip", 10.0))
        self.target_tau = float(optim_cfg.get("critic_tau", 0.01))

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=self.device)
        self.action_scale = 0.5 * (self.action_high - self.action_low)
        self.action_bias = 0.5 * (self.action_high + self.action_low)
        self.exploration_std = float(config.get("behavior", {}).get("exploration_std", 0.3))

        self.state = self.world_model.initial_state(batch_size=1, device=self.device)
        self.prev_action = torch.zeros(1, action_dim, device=self.device)

    def reset(self) -> None:
        self.state = self.world_model.initial_state(batch_size=1, device=self.device)
        self.prev_action = torch.zeros(1, self.prev_action.shape[-1], device=self.device)

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
        self.state = self.world_model.observe_step(self.state, self.prev_action, obs_tensor, deterministic=eval_mode)
        features = self.world_model.feature(self.state)
        action, _, _ = self.world_model.actor.sample(features, deterministic=eval_mode)
        if not eval_mode:
            action = (action + torch.randn_like(action) * self.exploration_std).clamp(-1.0, 1.0)
        env_action = self._denormalize_action(action)
        self.prev_action = action.detach()
        return env_action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, batch: ReplayBatch) -> UpdateOutput:
        batch = self._normalize_batch(batch)

        self.model_optimizer.zero_grad(set_to_none=True)
        wm_losses = self.world_model.compute_world_model_loss(batch)
        wm_losses.total.backward()
        model_grad_norm = float(nn_utils.clip_grad_norm_(list(self.world_model.world_model_parameters()), self.grad_clip))
        self.model_optimizer.step()

        self.actor_optimizer.zero_grad(set_to_none=True)
        with self.world_model.freeze_for_actor():
            actor_losses = self.world_model.compute_actor_loss(batch)
        actor_losses.actor_loss.backward()
        actor_grad_norm = float(nn_utils.clip_grad_norm_(list(self.world_model.actor_parameters()), self.grad_clip))
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad(set_to_none=True)
        with self.world_model.freeze_for_critic():
            critic_losses = self.world_model.compute_critic_loss(batch)
        critic_losses.critic_loss.backward()
        critic_grad_norm = float(nn_utils.clip_grad_norm_(list(self.world_model.critic_parameters()), self.grad_clip))
        self.critic_optimizer.step()

        self.world_model.update_targets(self.target_tau)
        diagnostics = self.world_model.compute_diagnostics(batch)

        return UpdateOutput(
            world_model_loss=float(wm_losses.total.item()),
            actor_loss=float(actor_losses.actor_loss.item()),
            critic_loss=float(critic_losses.critic_loss.item()),
            recon_loss=float(wm_losses.recon_loss.item()),
            reward_loss=float(wm_losses.reward_loss.item()),
            continue_loss=float(wm_losses.continue_loss.item()),
            dyn_kl=float(wm_losses.dyn_kl.item()),
            rep_kl=float(wm_losses.rep_kl.item()),
            policy_std=float(actor_losses.policy_std.item()),
            model_grad_norm=model_grad_norm,
            actor_grad_norm=actor_grad_norm,
            critic_grad_norm=critic_grad_norm,
            imagine_return=float(actor_losses.imagine_return.item()),
            value_mean=float(critic_losses.value_mean.item()),
            prior_entropy=float(wm_losses.prior_entropy.item()),
            post_entropy=float(wm_losses.post_entropy.item()),
            extra_metrics={f"train/{key}": float(value.item()) for key, value in diagnostics.items()},
        )

    def state_dict(self) -> dict:
        return {
            "model": self.world_model.state_dict(),
            "model_optimizer": self.model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "state": {
                "deter": self.state.deter.detach().cpu(),
                "stoch": self.state.stoch.detach().cpu(),
                "logits": self.state.logits.detach().cpu(),
            },
            "prev_action": self.prev_action.detach().cpu(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.world_model.load_state_dict(state["model"])
        model_optimizer_state = state.get("model_optimizer")
        if model_optimizer_state is not None:
            self.model_optimizer.load_state_dict(model_optimizer_state)
        actor_optimizer_state = state.get("actor_optimizer")
        if actor_optimizer_state is not None:
            self.actor_optimizer.load_state_dict(actor_optimizer_state)
        critic_optimizer_state = state.get("critic_optimizer")
        if critic_optimizer_state is not None:
            self.critic_optimizer.load_state_dict(critic_optimizer_state)
        raw_state = state.get("state")
        if raw_state is None:
            self.state = self.world_model.initial_state(batch_size=1, device=self.device)
        else:
            self.state = RSSMState(
                deter=raw_state["deter"].to(self.device),
                stoch=raw_state["stoch"].to(self.device),
                logits=raw_state["logits"].to(self.device),
            )
        self.prev_action = state.get("prev_action", torch.zeros_like(self.prev_action)).to(self.device)
