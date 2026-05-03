from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from hamworld.modules import (
    ActionPriorHead,
    AuxResidualUpdater,
    CanonicalDynamicsCore,
    HamiltonianHead,
    MambaMemory,
    ProjectionHead,
    StateEncoder,
    copy_module,
    soft_categorical_loss,
    twohot_cross_entropy,
    twohot_decode,
    twohot_loss,
    zero_last_layer,
)
from hamworld.runtime import soft_update


@dataclass
class TransitionInfo:
    memory: torch.Tensor
    dq_net: torch.Tensor
    dp_net: torch.Tensor
    dH_dq: torch.Tensor
    dH_dp: torch.Tensor
    control: torch.Tensor
    dq: torch.Tensor
    dp: torch.Tensor
    dc: torch.Tensor
    q: torch.Tensor
    p: torch.Tensor
    c: torch.Tensor
    q_next: torch.Tensor
    p_next: torch.Tensor
    c_next: torch.Tensor
    energy: torch.Tensor
    next_energy: torch.Tensor
    hamiltonian_grad_norm: torch.Tensor


def _clone_state(state: list[torch.Tensor]) -> list[torch.Tensor]:
    return [tensor.clone() for tensor in state]


class CanonicalDynamicsWorldModel(nn.Module):
    def __init__(self, config: dict, obs_dim: int, action_dim: int):
        super().__init__()
        latent_cfg = config.get("latent", {})
        model_cfg = config.get("model", {})

        self.latent_dim = int(latent_cfg.get("latent_dim", 48))
        if self.latent_dim < 3:
            raise ValueError("latent_dim must be at least 3 for the q/p/c split.")
        self.q_dim, self.p_dim, self.c_dim = self._resolve_latent_dims(latent_cfg)
        self.action_dim = action_dim
        self.training_total_steps = max(1, int(config.get("training", {}).get("total_steps", 1)))
        self.training_seed_steps = max(0, int(config.get("training", {}).get("seed_steps", 0)))
        self.hamiltonian_alpha_base = float(model_cfg.get("hamiltonian_alpha", 0.1))
        self.hamiltonian_alpha_growth_rate = float(model_cfg.get("hamiltonian_alpha_growth_rate", 0.0))
        self.hamiltonian_alpha_max = float(model_cfg.get("hamiltonian_alpha_max", max(0.5, self.hamiltonian_alpha_base)))
        self.hamiltonian_alpha_growth_start = model_cfg.get("hamiltonian_alpha_growth_start", 0.0)
        self._hamiltonian_schedule_step: int | None = None
        self.hamiltonian_grad_clip = float(model_cfg.get("hamiltonian_grad_clip", 5.0))
        self.latent_norm_clip = float(model_cfg.get("latent_norm_clip", 50.0))
        self.detach_hamiltonian_on_explode = bool(model_cfg.get("detach_hamiltonian_on_explode", True))
        self.detach_hamiltonian_threshold = float(model_cfg.get("detach_hamiltonian_threshold", 25.0))

        encoder_hidden = list(model_cfg.get("encoder_hidden_dims", [256, 256]))
        projector_hidden = list(model_cfg.get("projector_hidden_dims", [128]))
        head_hidden = list(model_cfg.get("head_hidden_dims", [128, 128]))
        policy_hidden = list(model_cfg.get("policy_hidden_dims", head_hidden))
        dynamics_hidden = list(model_cfg.get("dynamics_hidden_dims", [128, 128]))
        aux_hidden = list(model_cfg.get("aux_hidden_dims", dynamics_hidden))
        energy_hidden = list(model_cfg.get("energy_hidden_dims", [128, 128]))
        memory_dim = int(model_cfg.get("memory_model_dim", 128))
        memory_state_dim = int(model_cfg.get("memory_state_dim", 128))
        memory_layers = int(model_cfg.get("memory_layers", 2))
        projection_dim = int(model_cfg.get("projection_dim", 64))
        scalar_num_bins = int(model_cfg.get("scalar_num_bins", 255))
        scalar_bin_lo = float(model_cfg.get("scalar_bin_lo", -20.0))
        scalar_bin_hi = float(model_cfg.get("scalar_bin_hi", 20.0))
        zero_init_scalar_heads = bool(model_cfg.get("zero_init_scalar_heads", True))
        losses_cfg = config.get("losses", {})
        self.value_lambda_return = float(losses_cfg.get("value_lambda_return", 0.95))
        self.value_slow_reg = float(losses_cfg.get("value_slow_reg", 1.0))

        self.encoder = StateEncoder(obs_dim, encoder_hidden, self.latent_dim)
        self.target_encoder = copy_module(self.encoder)
        self.projector = ProjectionHead(self.latent_dim, projector_hidden, projection_dim)
        self.target_projector = copy_module(self.projector)
        self.register_buffer("reward_bins", torch.linspace(scalar_bin_lo, scalar_bin_hi, scalar_num_bins))
        self.register_buffer("value_bins", torch.linspace(scalar_bin_lo, scalar_bin_hi, scalar_num_bins))
        self.memory = MambaMemory(
            latent_dim=self.latent_dim,
            action_dim=action_dim,
            model_dim=memory_dim,
            state_dim=memory_state_dim,
            num_layers=memory_layers,
        )
        self.canonical_core = CanonicalDynamicsCore(
            q_dim=self.q_dim,
            p_dim=self.p_dim,
            c_dim=self.c_dim,
            action_dim=action_dim,
            memory_dim=memory_dim,
            hidden_dims=dynamics_hidden,
            use_control_map=bool(model_cfg.get("use_control_map", True)),
        )
        self.aux_updater = AuxResidualUpdater(
            q_dim=self.q_dim,
            p_dim=self.p_dim,
            c_dim=self.c_dim,
            action_dim=action_dim,
            memory_dim=memory_dim,
            hidden_dims=aux_hidden,
        )
        self.energy_head = HamiltonianHead(self.q_dim, self.p_dim, energy_hidden)
        self.reward_head = ProjectionHead(self.latent_dim, head_hidden, scalar_num_bins)
        self.value_head = ProjectionHead(self.latent_dim, head_hidden, scalar_num_bins)
        if zero_init_scalar_heads:
            zero_last_layer(self.reward_head)
            zero_last_layer(self.value_head)
        self.policy_prior = ActionPriorHead(self.latent_dim, policy_hidden, action_dim)
        self.value_target = copy_module(self.value_head)
        self.rollout_steps = int(config.get("losses", {}).get("rollout_steps", 3))
        self._freeze_target_modules()

    def _resolve_latent_dims(self, latent_cfg: dict) -> tuple[int, int, int]:
        requested = [latent_cfg.get("q_dim"), latent_cfg.get("p_dim"), latent_cfg.get("c_dim")]
        provided = sum(value is not None for value in requested)
        if provided == 0:
            canonical_dim = self.latent_dim // 3
            return canonical_dim, canonical_dim, self.latent_dim - (2 * canonical_dim)
        if provided != 3:
            raise ValueError("Either provide all of q_dim/p_dim/c_dim or omit all three.")
        q_dim, p_dim, c_dim = (int(value) for value in requested)
        if min(q_dim, p_dim, c_dim) <= 0:
            raise ValueError("q_dim, p_dim, and c_dim must all be positive.")
        if q_dim + p_dim + c_dim != self.latent_dim:
            raise ValueError(
                f"Configured latent split q={q_dim}, p={p_dim}, c={c_dim} does not sum to latent_dim={self.latent_dim}."
            )
        return q_dim, p_dim, c_dim

    def _freeze_target_modules(self) -> None:
        for module in (self.target_encoder, self.target_projector, self.value_target):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    @property
    def hamiltonian_alpha(self) -> float:
        return self.resolve_hamiltonian_alpha()

    def set_schedule_step(self, step: int | None) -> None:
        self._hamiltonian_schedule_step = None if step is None else max(0, int(step))

    def get_schedule_step(self) -> int | None:
        return self._hamiltonian_schedule_step

    def resolve_hamiltonian_alpha(self, step: int | None = None) -> float:
        effective_step = self._hamiltonian_schedule_step if step is None else max(0, int(step))
        alpha = self.hamiltonian_alpha_base
        if effective_step is not None and self.hamiltonian_alpha_growth_rate != 0.0:
            growth_start = self._resolve_alpha_growth_start(self.hamiltonian_alpha_growth_start)
            active_steps = max(0.0, float(effective_step) - growth_start)
            alpha += self.hamiltonian_alpha_growth_rate * active_steps
        alpha = min(alpha, self.hamiltonian_alpha_max)
        return max(0.0, float(alpha))

    def _resolve_alpha_growth_start(self, raw_value) -> float:
        if raw_value is None:
            return 0.0
        if isinstance(raw_value, str):
            normalized = raw_value.strip().lower()
            if normalized in {"seed_end", "seed_steps"}:
                return float(self.training_seed_steps)
            if normalized.endswith("%"):
                return float(self.training_total_steps) * (float(normalized[:-1]) / 100.0)
            raw_value = float(normalized)
        if isinstance(raw_value, (int, float)):
            if raw_value <= 1.0:
                return float(raw_value) * float(self.training_total_steps)
            return float(raw_value)
        raise ValueError(f"Unsupported hamiltonian_alpha_growth_start={raw_value!r}")

    def encode(self, observations: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        encoder = self.target_encoder if use_target else self.encoder
        return self._clamp_latent_norm(encoder(observations))

    def split_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, p, c = torch.split(latent, [self.q_dim, self.p_dim, self.c_dim], dim=-1)
        return q, p, c

    def _clip_vector_norm(self, tensor: torch.Tensor, max_norm: float) -> torch.Tensor:
        if max_norm <= 0.0:
            return tensor
        norm = torch.norm(tensor, dim=-1, keepdim=True).clamp_min(1e-6)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return tensor * scale

    def _clamp_latent_norm(self, latent: torch.Tensor) -> torch.Tensor:
        return self._clip_vector_norm(latent, self.latent_norm_clip)

    def _cross_covariance_penalty(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        lhs_flat = lhs.reshape(-1, lhs.shape[-1])
        rhs_flat = rhs.reshape(-1, rhs.shape[-1])
        lhs_centered = lhs_flat - lhs_flat.mean(dim=0, keepdim=True)
        rhs_centered = rhs_flat - rhs_flat.mean(dim=0, keepdim=True)
        denom = max(1, lhs_centered.shape[0] - 1)
        covariance = lhs_centered.transpose(0, 1) @ rhs_centered / float(denom)
        return covariance.pow(2).mean()

    def _hamiltonian_terms(
        self,
        q: torch.Tensor,
        p: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        create_graph = torch.is_grad_enabled()
        if create_graph:
            q_for_energy = q
            p_for_energy = p
            energy = self.energy_head(q_for_energy, p_for_energy)
            dH_dq, dH_dp = torch.autograd.grad(
                energy.sum(),
                (q_for_energy, p_for_energy),
                create_graph=True,
                retain_graph=True,
            )
        else:
            with torch.enable_grad():
                q_for_energy = q.detach().requires_grad_(True)
                p_for_energy = p.detach().requires_grad_(True)
                energy = self.energy_head(q_for_energy, p_for_energy)
                dH_dq, dH_dp = torch.autograd.grad(
                    energy.sum(),
                    (q_for_energy, p_for_energy),
                    create_graph=False,
                    retain_graph=False,
                )
            energy = energy.detach()
            dH_dq = dH_dq.detach()
            dH_dp = dH_dp.detach()

        raw_grad_norm = 0.5 * (
            torch.norm(dH_dq.detach(), dim=-1).mean() + torch.norm(dH_dp.detach(), dim=-1).mean()
        )
        dH_dq = self._clip_vector_norm(dH_dq, self.hamiltonian_grad_clip)
        dH_dp = self._clip_vector_norm(dH_dp, self.hamiltonian_grad_clip)
        if self.detach_hamiltonian_on_explode and raw_grad_norm.item() > self.detach_hamiltonian_threshold:
            dH_dq = dH_dq.detach()
            dH_dp = dH_dp.detach()
        return energy, dH_dq, dH_dp, raw_grad_norm

    def _resolve_weight(self, base_weight: float, config: dict, key: str, step: int | None) -> float:
        if step is None:
            return base_weight
        warmup_cfg = config.get("warmup", {})
        total_steps = max(1, int(config["training"]["total_steps"]))
        progress = min(1.0, max(0.0, float(step) / float(total_steps)))

        start = float(warmup_cfg.get(f"{key}_warmup_start", 0.0))
        end = float(warmup_cfg.get(f"{key}_warmup_end", 0.0))
        init_scale = float(warmup_cfg.get(f"{key}_warmup_init_scale", 1.0))

        if start > 1.0:
            start = start / float(total_steps)
        if end > 1.0:
            end = end / float(total_steps)

        if end <= start:
            scale = 1.0 if progress >= end else init_scale
        elif progress <= start:
            scale = init_scale
        elif progress >= end:
            scale = 1.0
        else:
            ramp = (progress - start) / max(1e-6, end - start)
            scale = init_scale + (1.0 - init_scale) * ramp
        return base_weight * scale

    def _resolve_optional_schedule(self, base_value: float, schedule_cfg: dict | None, config: dict, step: int | None) -> float:
        if step is None or not schedule_cfg:
            return base_value
        schedule_type = str(schedule_cfg.get("type", "constant")).lower()
        if schedule_type == "constant":
            return base_value

        total_steps = max(1, int(config["training"]["total_steps"]))
        start = float(schedule_cfg.get("start", 0.0))
        end = float(schedule_cfg.get("end", total_steps))
        if start <= 1.0:
            start *= float(total_steps)
        if end <= 1.0:
            end *= float(total_steps)

        if schedule_type == "linear_decay":
            final_scale = float(schedule_cfg.get("final_scale", 0.0))
            if step <= start:
                scale = 1.0
            elif step >= end:
                scale = final_scale
            else:
                progress = (float(step) - start) / max(1e-6, end - start)
                scale = 1.0 + (final_scale - 1.0) * progress
            return base_value * scale
        return base_value

    def _reduce_masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.to(dtype=values.dtype)
        denom = mask.sum()
        if denom.item() == 0:
            return torch.zeros((), dtype=values.dtype, device=values.device)
        return (values * mask).sum() / denom

    def _decode_reward(self, reward_logits: torch.Tensor) -> torch.Tensor:
        return twohot_decode(reward_logits, self.reward_bins)

    def _decode_value(self, value_logits: torch.Tensor) -> torch.Tensor:
        return twohot_decode(value_logits, self.value_bins)

    def _lambda_returns(
        self,
        rewards: torch.Tensor,
        discounts: torch.Tensor,
        next_values: torch.Tensor,
    ) -> torch.Tensor:
        returns = torch.zeros_like(rewards)
        acc = next_values[:, -1]
        for index in reversed(range(rewards.shape[1])):
            acc = rewards[:, index] + discounts[:, index] * (
                (1.0 - self.value_lambda_return) * next_values[:, index] + self.value_lambda_return * acc
            )
            returns[:, index] = acc
        return returns

    def imagine_step(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        memory_state: list[torch.Tensor] | None = None,
        return_info: bool = False,
        step: int | None = None,
    ):
        memory, next_memory_state = self.memory.step(latent, action, memory_state)
        q, p, c = self.split_latent(latent)
        dq_net, dp_net, control = self.canonical_core(q, p, c, action, memory)
        energy, dH_dq, dH_dp, hamiltonian_grad_norm = self._hamiltonian_terms(q, p)
        alpha = self.resolve_hamiltonian_alpha(step)
        dq = (1.0 - alpha) * dq_net + alpha * dH_dp
        dp = (1.0 - alpha) * dp_net - alpha * dH_dq + control
        dc = self.aux_updater(q, p, c, action, memory)
        q_next_raw = q + dq
        p_next_raw = p + dp
        c_next_raw = c + dc
        next_latent = torch.cat([q_next_raw, p_next_raw, c_next_raw], dim=-1)
        next_latent = self._clamp_latent_norm(torch.nan_to_num(next_latent, nan=0.0, posinf=1e6, neginf=-1e6))
        q_next, p_next, c_next = self.split_latent(next_latent)

        if not return_info:
            return next_latent, next_memory_state

        info = TransitionInfo(
            memory=memory,
            dq_net=dq_net,
            dp_net=dp_net,
            dH_dq=dH_dq,
            dH_dp=dH_dp,
            control=control,
            dq=dq,
            dp=dp,
            dc=dc,
            q=q,
            p=p,
            c=c,
            q_next=q_next,
            p_next=p_next,
            c_next=c_next,
            energy=energy,
            next_energy=self.energy_head(q_next, p_next),
            hamiltonian_grad_norm=hamiltonian_grad_norm,
        )
        return next_latent, next_memory_state, info

    def _teacher_forced_rollout(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        step: int | None = None,
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]], list[TransitionInfo]]:
        batch_size, horizon = actions.shape[:2]
        memory_state = self.reset_memory_state(batch_size, latents.device)
        state_snapshots: list[list[torch.Tensor]] = []
        predictions = []
        infos: list[TransitionInfo] = []
        for index in range(horizon):
            state_snapshots.append(_clone_state(memory_state))
            pred_latent, memory_state, info = self.imagine_step(
                latents[:, index],
                actions[:, index],
                memory_state,
                return_info=True,
                step=step,
            )
            predictions.append(pred_latent)
            infos.append(info)
        return torch.stack(predictions, dim=1), state_snapshots, infos

    def _multi_step_rollout_loss(
        self,
        encoded_obs: torch.Tensor,
        actions: torch.Tensor,
        state_snapshots: list[list[torch.Tensor]],
        step: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, horizon = actions.shape[:2]
        max_k = max(1, min(self.rollout_steps, horizon))
        roll_terms = []
        drift_terms = []
        one_step_terms = []
        three_step_terms = []

        for start in range(horizon):
            rollout_latent = encoded_obs[:, start]
            rollout_state = _clone_state(state_snapshots[start])
            origin = rollout_latent
            for delta in range(1, max_k + 1):
                action_index = start + delta - 1
                if action_index >= horizon:
                    break
                rollout_latent, rollout_state = self.imagine_step(
                    rollout_latent,
                    actions[:, action_index],
                    rollout_state,
                    step=step,
                )
                target = encoded_obs[:, start + delta].detach()
                mse = F.mse_loss(rollout_latent, target)
                roll_terms.append(mse)
                drift_terms.append(torch.norm(rollout_latent - origin, dim=-1).mean())
                if delta == 1:
                    one_step_terms.append(mse.detach())
                if delta == 3:
                    three_step_terms.append(mse.detach())

        device = encoded_obs.device
        zero = torch.zeros((), device=device)
        roll_loss = torch.stack(roll_terms).mean() if roll_terms else zero
        drift = torch.stack(drift_terms).mean() if drift_terms else zero
        one_step_mse = torch.stack(one_step_terms).mean() if one_step_terms else zero
        three_step_mse = torch.stack(three_step_terms).mean() if three_step_terms else zero
        return roll_loss, drift, one_step_mse, three_step_mse

    def compute_losses(self, batch, config: dict, step: int | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size, horizon, obs_dim = batch.observations.shape[0], batch.actions.shape[1], batch.observations.shape[-1]
        flat_obs = batch.observations.reshape(batch_size * (horizon + 1), obs_dim)
        encoded_obs = self.encode(flat_obs).view(batch_size, horizon + 1, self.latent_dim)
        target_obs = self.encode(flat_obs, use_target=True).view(batch_size, horizon + 1, self.latent_dim).detach()

        current_z = encoded_obs[:, :-1]
        next_z_online = encoded_obs[:, 1:]
        next_z_target = target_obs[:, 1:]
        q_true, p_true, c_true = self.split_latent(current_z)
        q_next_true, p_next_true, c_next_true = self.split_latent(next_z_online.detach())
        pred_next_z, state_snapshots, infos = self._teacher_forced_rollout(current_z, batch.actions, step=step)

        repr_loss = F.mse_loss(
            self.projector(pred_next_z.reshape(batch_size * horizon, self.latent_dim)),
            self.target_projector(next_z_target.reshape(batch_size * horizon, self.latent_dim)),
        )
        dyn_loss = F.mse_loss(pred_next_z, next_z_online.detach())
        reward_logits = self.reward_head(pred_next_z)
        reward_loss = twohot_loss(reward_logits, batch.rewards, self.reward_bins)
        reward_pred = self._decode_reward(reward_logits)
        reward_pred_mae = (reward_pred - batch.rewards).abs().mean()
        reward_pred_mse = F.mse_loss(reward_pred, batch.rewards)

        discount = batch.discounts * float(config["training"]["discount"])
        value_logits = self.value_head(current_z)
        with torch.no_grad():
            next_value_logits = self.value_target(next_z_online.detach())
            value_targets = self._lambda_returns(batch.rewards, discount, self._decode_value(next_value_logits))
            slow_target_logits = self.value_target(current_z.detach())
        value_ce_loss = twohot_cross_entropy(value_logits, value_targets, self.value_bins)
        value_slow_loss = soft_categorical_loss(value_logits, slow_target_logits)
        value_loss = (value_ce_loss + self.value_slow_reg * value_slow_loss).mean()
        value_pred = self._decode_value(value_logits)
        value_pred_mae = (value_pred - value_targets).abs().mean()
        value_pred_mse = F.mse_loss(value_pred, value_targets)

        roll_loss, rollout_latent_drift, one_step_latent_mse, three_step_latent_mse = self._multi_step_rollout_loss(
            encoded_obs,
            batch.actions,
            state_snapshots,
            step=step,
        )
        policy_prior_pred = self.policy_action(current_z.detach())
        policy_prior_loss = F.mse_loss(policy_prior_pred, batch.actions)

        dq = torch.stack([info.dq for info in infos], dim=1)
        dp = torch.stack([info.dp for info in infos], dim=1)
        dq_net = torch.stack([info.dq_net for info in infos], dim=1)
        dp_net = torch.stack([info.dp_net for info in infos], dim=1)
        dH_dq = torch.stack([info.dH_dq for info in infos], dim=1)
        dH_dp = torch.stack([info.dH_dp for info in infos], dim=1)
        control = torch.stack([info.control for info in infos], dim=1)
        q = torch.stack([info.q for info in infos], dim=1)
        p = torch.stack([info.p for info in infos], dim=1)
        c = torch.stack([info.c for info in infos], dim=1)
        dc = torch.stack([info.dc for info in infos], dim=1)
        energy = torch.stack([info.energy for info in infos], dim=1)
        next_energy = torch.stack([info.next_energy for info in infos], dim=1)
        hamiltonian_grad_norm = torch.stack([info.hamiltonian_grad_norm for info in infos], dim=0).mean()

        regularization_cfg = config.get("regularization", {})
        threshold = float(regularization_cfg.get("small_action_threshold", 0.15))
        action_norm = torch.norm(batch.actions, dim=-1, keepdim=True)
        small_action_mask = (action_norm < threshold).to(dtype=pred_next_z.dtype)
        sa_values = dq.pow(2).mean(dim=-1, keepdim=True) + dp.pow(2).mean(dim=-1, keepdim=True)
        sa_loss = self._reduce_masked_mean(sa_values, small_action_mask)
        energy_delta = (next_energy - energy).pow(2)
        energy_loss = self._reduce_masked_mean(energy_delta, small_action_mask)
        energy_drift = (next_energy - energy).abs().mean()
        hamiltonian_loss = F.mse_loss(dq_net, dH_dp) + F.mse_loss(dp_net, -dH_dq)

        losses_cfg = config.get("losses", {})
        temp_balance = float(losses_cfg.get("temp_balance_lambda", 0.5))
        temp_loss = dq.pow(2).mean() - temp_balance * dp.pow(2).mean()
        decouple_loss = self._cross_covariance_penalty(q, p)
        c_sparse_loss = dc.abs().mean()
        delta_q_norm = torch.norm(q_next_true - q_true, dim=-1).mean()
        delta_p_norm = torch.norm(p_next_true - p_true, dim=-1).mean()
        delta_c_norm = torch.norm(c_next_true - c_true, dim=-1).mean()

        q_norm = torch.norm(q, dim=-1).mean()
        p_norm = torch.norm(p, dim=-1).mean()
        c_norm = torch.norm(c, dim=-1).mean()
        latent_norm = torch.norm(current_z, dim=-1).mean()
        control_norm = torch.norm(control, dim=-1).mean()
        pred_value_mean = value_pred.mean()
        pred_reward_mean = reward_pred.mean()

        repr_weight = float(losses_cfg.get("w_repr", 1.0))
        dyn_weight = float(losses_cfg.get("w_dyn", 1.0))
        reward_weight = float(losses_cfg.get("w_reward", 1.0))
        value_weight = float(losses_cfg.get("w_value", 0.5))
        roll_weight = self._resolve_weight(float(losses_cfg.get("w_roll", 0.5)), config, "roll", step)
        sa_weight = self._resolve_weight(float(losses_cfg.get("w_sa", 0.05)), config, "sa", step)
        energy_weight = self._resolve_weight(float(losses_cfg.get("w_energy", 0.01)), config, "energy", step)
        policy_weight = self._resolve_optional_schedule(
            float(losses_cfg.get("w_policy_prior", 0.1)),
            losses_cfg.get("policy_prior_schedule"),
            config,
            step,
        )
        hamiltonian_weight = float(losses_cfg.get("lambda_h", 0.05))
        temp_weight = float(losses_cfg.get("lambda_temp", 0.01))
        decouple_weight = float(losses_cfg.get("lambda_dec", 0.01))
        c_weight = float(losses_cfg.get("lambda_c", 0.001))

        total = (
            repr_weight * repr_loss
            + dyn_weight * dyn_loss
            + reward_weight * reward_loss
            + value_weight * value_loss
            + roll_weight * roll_loss
            + sa_weight * sa_loss
            + energy_weight * energy_loss
            + policy_weight * policy_prior_loss
            + hamiltonian_weight * hamiltonian_loss
            + temp_weight * temp_loss
            + decouple_weight * decouple_loss
            + c_weight * c_sparse_loss
        )

        losses = {
            "repr_loss": repr_loss,
            "dyn_loss": dyn_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
            "value_ce_loss": value_ce_loss.mean(),
            "value_slow_loss": value_slow_loss.mean(),
            "reward_pred_mae": reward_pred_mae,
            "reward_pred_mse": reward_pred_mse,
            "value_pred_mae": value_pred_mae,
            "value_pred_mse": value_pred_mse,
            "roll_loss": roll_loss,
            "sa_loss": sa_loss,
            "energy_loss": energy_loss,
            "energy_drift": energy_drift,
            "policy_prior_loss": policy_prior_loss,
            "hamiltonian_loss": hamiltonian_loss,
            "temp_loss": temp_loss,
            "decouple_loss": decouple_loss,
            "c_sparse_loss": c_sparse_loss,
            "delta_q_norm": delta_q_norm,
            "delta_p_norm": delta_p_norm,
            "delta_c_norm": delta_c_norm,
            "q_norm": q_norm,
            "p_norm": p_norm,
            "c_norm": c_norm,
            "latent_norm": latent_norm,
            "control_norm": control_norm,
            "hamiltonian_grad_norm": hamiltonian_grad_norm,
            "rollout_latent_drift": rollout_latent_drift,
            "pred_reward_mean": pred_reward_mean,
            "pred_value_mean": pred_value_mean,
            "one_step_latent_mse": one_step_latent_mse,
            "three_step_latent_mse": three_step_latent_mse,
            "roll_weight": torch.as_tensor(roll_weight, dtype=repr_loss.dtype, device=repr_loss.device),
            "sa_weight": torch.as_tensor(sa_weight, dtype=repr_loss.dtype, device=repr_loss.device),
            "energy_weight": torch.as_tensor(energy_weight, dtype=repr_loss.dtype, device=repr_loss.device),
            "policy_weight": torch.as_tensor(policy_weight, dtype=repr_loss.dtype, device=repr_loss.device),
            "hamiltonian_weight": torch.as_tensor(hamiltonian_weight, dtype=repr_loss.dtype, device=repr_loss.device),
            "temp_weight": torch.as_tensor(temp_weight, dtype=repr_loss.dtype, device=repr_loss.device),
            "decouple_weight": torch.as_tensor(decouple_weight, dtype=repr_loss.dtype, device=repr_loss.device),
            "c_weight": torch.as_tensor(c_weight, dtype=repr_loss.dtype, device=repr_loss.device),
            "alpha": torch.as_tensor(self.resolve_hamiltonian_alpha(step), dtype=repr_loss.dtype, device=repr_loss.device),
        }
        return total, losses

    def predict_reward(self, latent: torch.Tensor) -> torch.Tensor:
        reward_logits = self.reward_head(self._clamp_latent_norm(latent))
        return self._decode_reward(reward_logits)

    def predict_value(self, latent: torch.Tensor) -> torch.Tensor:
        value_logits = self.value_head(self._clamp_latent_norm(latent))
        return self._decode_value(value_logits)

    def policy_action(self, latent: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        latent = self._clamp_latent_norm(latent)
        action = self.policy_prior(latent)
        return torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)

    def advance_memory(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        memory_state: list[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        _, next_memory_state = self.memory.step(latent, action, memory_state)
        return next_memory_state

    def reset_memory_state(self, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        return self.memory.init_state(batch_size, device)

    def reset_prior_state(self, batch_size: int, device: torch.device) -> list[torch.Tensor]:
        return self.reset_memory_state(batch_size, device)

    def update_targets(self, encoder_tau: float, value_tau: float) -> None:
        soft_update(self.target_encoder, self.encoder, encoder_tau)
        soft_update(self.target_projector, self.projector, encoder_tau)
        soft_update(self.value_target, self.value_head, value_tau)


PGMWorldModel = CanonicalDynamicsWorldModel


def infer_checkpoint_step(payload: dict, checkpoint_path: str | Path | None = None) -> int | None:
    trainer_state = payload.get("trainer_state")
    if isinstance(trainer_state, dict) and trainer_state.get("step") is not None:
        return max(0, int(trainer_state["step"]))
    if checkpoint_path is None:
        return None
    match = re.search(r"checkpoint_(\d+)\.pt$", Path(checkpoint_path).name)
    if match is None:
        return None
    return int(match.group(1))
