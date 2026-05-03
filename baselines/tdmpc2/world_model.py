from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tdmpc2.modules import LatentDynamics, ProjectionHead, StateEncoder, TanhGaussianPolicy, TwinQ, copy_module, freeze_parameters
from tdmpc2.runtime import soft_update


DIAGNOSTIC_HORIZONS = (1, 3, 5, 10, 15)


@dataclass
class ModelLosses:
    total: torch.Tensor
    consistency_loss: torch.Tensor
    reward_loss: torch.Tensor
    q_loss: torch.Tensor
    q_mean: torch.Tensor
    td_target_mean: torch.Tensor


@dataclass
class PolicyLosses:
    policy_loss: torch.Tensor
    q_mean: torch.Tensor
    policy_std: torch.Tensor


class TDMPC2WorldModel(nn.Module):
    def __init__(self, config: dict, obs_dim: int, action_dim: int):
        super().__init__()
        model_cfg = config.get("model", {})
        latent_cfg = config.get("latent", {})
        policy_cfg = config.get("policy", {})

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount = float(config["training"]["discount"])
        self.consistency_scale = float(config.get("losses", {}).get("consistency_scale", 1.0))
        self.reward_scale = float(config.get("losses", {}).get("reward_scale", 1.0))
        self.q_scale = float(config.get("losses", {}).get("q_scale", 1.0))
        latent_dim = int(latent_cfg.get("latent_dim", 128))
        simnorm_groups = int(latent_cfg.get("simnorm_groups", 8))

        self.encoder = StateEncoder(obs_dim, list(model_cfg.get("encoder_hidden_dims", [256, 256])), latent_dim, simnorm_groups)
        self.target_encoder = copy_module(self.encoder)
        self.dynamics = LatentDynamics(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dims=list(model_cfg.get("dynamics_hidden_dims", [256, 256])),
            simnorm_groups=simnorm_groups,
        )
        self.reward_head = ProjectionHead(latent_dim + action_dim, list(model_cfg.get("reward_hidden_dims", [256, 256])), 1)
        self.policy = TanhGaussianPolicy(
            latent_dim=latent_dim,
            hidden_dims=list(model_cfg.get("policy_hidden_dims", [256, 256])),
            action_dim=action_dim,
            min_std=float(policy_cfg.get("min_std", 0.05)),
            max_std=float(policy_cfg.get("max_std", 1.0)),
        )
        self.q = TwinQ(latent_dim=latent_dim, action_dim=action_dim, hidden_dims=list(model_cfg.get("q_hidden_dims", [256, 256])))
        self.q_target = copy_module(self.q)
        self._freeze_targets()

    def _freeze_targets(self) -> None:
        for module in (self.target_encoder, self.q_target):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def model_parameters(self):
        modules = [self.encoder, self.dynamics, self.reward_head, self.q]
        for module in modules:
            yield from module.parameters()

    def policy_parameters(self):
        yield from self.policy.parameters()

    def encode(self, observation: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        encoder = self.target_encoder if use_target else self.encoder
        return encoder(observation)

    def imagine_step(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.dynamics(latent, action)

    def predict_reward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.reward_head(torch.cat([latent, action], dim=-1))

    def q_values(self, latent: torch.Tensor, action: torch.Tensor, use_target: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        q_module = self.q_target if use_target else self.q
        return q_module(latent, action)

    def terminal_value(self, latent: torch.Tensor) -> torch.Tensor:
        action, _, _ = self.policy.sample(latent, deterministic=True)
        q1, q2 = self.q_values(latent, action, use_target=True)
        return torch.min(q1, q2)

    def compute_model_loss(self, batch) -> ModelLosses:
        batch_size, horizon, _ = batch.actions.shape
        target_next = self.encode(batch.observations[:, 1:].reshape(-1, self.obs_dim), use_target=True).view(batch_size, horizon, -1).detach()

        # Roll out from first observation; collect latents for Q training
        rollout_latent = self.encode(batch.observations[:, 0])
        consistency_terms = []
        reward_terms = []
        rollout_latents = []
        for index in range(horizon):
            rollout_latents.append(rollout_latent)
            reward_pred = self.predict_reward(rollout_latent, batch.actions[:, index])
            next_latent = self.imagine_step(rollout_latent, batch.actions[:, index])
            consistency_terms.append(F.mse_loss(next_latent, target_next[:, index]))
            reward_terms.append(F.mse_loss(reward_pred, batch.rewards[:, index]))
            rollout_latent = next_latent

        consistency_loss = torch.stack(consistency_terms).mean()
        reward_loss = torch.stack(reward_terms).mean()

        # Q loss on rolled-out latents (matches TD-MPC2 paper, not re-encoded obs)
        flat_current = torch.stack(rollout_latents, dim=1).reshape(-1, rollout_latents[0].shape[-1])
        flat_actions = batch.actions.reshape(-1, batch.actions.shape[-1])
        q1, q2 = self.q(flat_current, flat_actions)
        q1 = q1.view(batch_size, horizon, 1)
        q2 = q2.view(batch_size, horizon, 1)

        with torch.no_grad():
            next_action, _, _ = self.policy.sample(target_next.reshape(-1, target_next.shape[-1]), deterministic=False)
            tq1, tq2 = self.q_target(target_next.reshape(-1, target_next.shape[-1]), next_action)
            target_q = torch.min(tq1, tq2).view(batch_size, horizon, 1)
            td_target = batch.rewards + self.discount * batch.discounts * target_q

        q_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        total = self.consistency_scale * consistency_loss + self.reward_scale * reward_loss + self.q_scale * q_loss
        return ModelLosses(
            total=total,
            consistency_loss=consistency_loss,
            reward_loss=reward_loss,
            q_loss=q_loss,
            q_mean=torch.min(q1, q2).mean(),
            td_target_mean=td_target.mean(),
        )

    @torch.no_grad()
    def compute_diagnostics(self, batch, horizons: tuple[int, ...] = DIAGNOSTIC_HORIZONS) -> dict[str, torch.Tensor]:
        batch_size, horizon, _ = batch.actions.shape
        encoded_obs = self.encode(batch.observations.reshape(-1, self.obs_dim)).view(batch_size, horizon + 1, -1)
        current_latent = encoded_obs[:, :-1]
        next_online = encoded_obs[:, 1:]
        target_next = self.encode(batch.observations[:, 1:].reshape(-1, self.obs_dim), use_target=True).view(batch_size, horizon, -1)

        reward_preds = []
        rollout_latents = []
        rollout_latent = current_latent[:, 0]
        for index in range(horizon):
            rollout_latents.append(rollout_latent)
            reward_preds.append(self.predict_reward(rollout_latent, batch.actions[:, index]))
            rollout_latent = self.imagine_step(rollout_latent, batch.actions[:, index])

        reward_pred = torch.stack(reward_preds, dim=1)
        reward_pred_mae = (reward_pred - batch.rewards).abs().mean()
        reward_pred_mse = F.mse_loss(reward_pred, batch.rewards)

        flat_current = torch.stack(rollout_latents, dim=1).reshape(-1, rollout_latents[0].shape[-1])
        flat_actions = batch.actions.reshape(-1, batch.actions.shape[-1])
        q1, q2 = self.q(flat_current, flat_actions)
        q_min = torch.min(q1, q2).view(batch_size, horizon, 1)
        next_action, _, _ = self.policy.sample(target_next.reshape(-1, target_next.shape[-1]), deterministic=False)
        tq1, tq2 = self.q_target(target_next.reshape(-1, target_next.shape[-1]), next_action)
        td_target = (batch.rewards + self.discount * batch.discounts * torch.min(tq1, tq2).view(batch_size, horizon, 1)).detach()
        value_pred_mae = (q_min - td_target).abs().mean()
        value_pred_mse = F.mse_loss(q_min, td_target)

        requested = tuple(sorted(set(int(item) for item in horizons)))
        max_horizon = max(requested) if requested else 1
        mse_terms = {item: [] for item in requested}
        drift_terms = {item: [] for item in requested}
        nan = torch.full((), float("nan"), device=batch.observations.device)
        for start in range(horizon):
            rollout_latent = current_latent[:, start]
            origin = rollout_latent
            for delta in range(1, max_horizon + 1):
                action_index = start + delta - 1
                if action_index >= horizon:
                    break
                rollout_latent = self.imagine_step(rollout_latent, batch.actions[:, action_index])
                if delta in mse_terms:
                    target = next_online[:, start + delta - 1].detach()
                    mse_terms[delta].append((rollout_latent - target).pow(2).sum(dim=-1).mean())
                    drift_terms[delta].append(torch.norm(rollout_latent - origin, dim=-1).mean())

        diagnostics = {
            "reward_pred_mae": reward_pred_mae,
            "reward_pred_mse": reward_pred_mse,
            "value_pred_mae": value_pred_mae,
            "value_pred_mse": value_pred_mse,
        }
        for item in requested:
            diagnostics[f"latent_mse_k{item}"] = torch.stack(mse_terms[item]).mean() if mse_terms[item] else nan
            diagnostics[f"rollout_drift_k{item}"] = torch.stack(drift_terms[item]).mean() if drift_terms[item] else nan
        return diagnostics

    def compute_policy_loss(self, batch) -> PolicyLosses:
        with torch.no_grad():
            latent = self.encode(batch.observations[:, :-1].reshape(-1, self.obs_dim))
        action, _, entropy = self.policy.sample(latent, deterministic=False)
        q1, q2 = self.q(latent, action)
        mean, std = self.policy(latent)
        del mean
        q_min = torch.min(q1, q2)
        policy_loss = -(q_min.mean() + 1e-3 * entropy.mean())
        return PolicyLosses(policy_loss=policy_loss, q_mean=q_min.mean(), policy_std=std.mean())

    def freeze_for_policy(self):
        modules = [self.encoder, self.target_encoder, self.dynamics, self.reward_head, self.q, self.q_target]
        return freeze_parameters(modules)

    def update_targets(self, tau: float) -> None:
        soft_update(self.target_encoder, self.encoder, tau)
        soft_update(self.q_target, self.q, tau)
