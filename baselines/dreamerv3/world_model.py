from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from dreamerv3.modules import (
    DreamerActor,
    DiscreteRSSM,
    PercentileNormalizer,
    ProjectionHead,
    RSSMState,
    StateEncoder,
    copy_module,
    detach_state,
    flatten_time,
    freeze_parameters,
    slice_state,
    soft_categorical_loss,
    stack_states,
    symlog,
    twohot_cross_entropy,
    twohot_decode,
    twohot_loss,
    zero_last_layer,
)
from dreamerv3.runtime import soft_update


DIAGNOSTIC_HORIZONS = (1, 3, 5, 10, 15)


@dataclass
class WorldModelLosses:
    total: torch.Tensor
    recon_loss: torch.Tensor
    reward_loss: torch.Tensor
    continue_loss: torch.Tensor
    dyn_kl: torch.Tensor
    rep_kl: torch.Tensor
    prior_entropy: torch.Tensor
    post_entropy: torch.Tensor


@dataclass
class BehaviorLosses:
    actor_loss: torch.Tensor
    critic_loss: torch.Tensor
    imagine_return: torch.Tensor
    value_mean: torch.Tensor
    policy_std: torch.Tensor


class DreamerV3WorldModel(nn.Module):
    def __init__(self, config: dict, obs_dim: int, action_dim: int):
        super().__init__()
        model_cfg = config.get("model", {})
        latent_cfg = config.get("latent", {})
        behavior_cfg = config.get("behavior", {})
        losses_cfg = config.get("losses", {})

        embed_dim = int(model_cfg.get("embed_dim", 256))
        deter_dim = int(latent_cfg.get("deter_dim", 256))
        stoch_size = int(latent_cfg.get("stoch_size", 32))
        classes = int(latent_cfg.get("classes", 32))
        rssm_hidden = list(model_cfg.get("rssm_hidden_dims", [256, 256]))
        decoder_hidden = list(model_cfg.get("decoder_hidden_dims", [256, 256]))
        head_hidden = list(model_cfg.get("head_hidden_dims", [256, 256]))
        actor_hidden = list(model_cfg.get("actor_hidden_dims", [256, 256]))
        critic_hidden = list(model_cfg.get("critic_hidden_dims", [256, 256]))
        num_bins = int(model_cfg.get("num_bins", 255))
        bin_lo = float(model_cfg.get("bin_lo", -20.0))
        bin_hi = float(model_cfg.get("bin_hi", 20.0))

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount = float(config["training"]["discount"])
        self.free_nats = float(losses_cfg.get("free_nats", 1.0))
        self.dyn_scale = float(losses_cfg.get("dyn_scale", 0.5))
        self.rep_scale = float(losses_cfg.get("rep_scale", 0.1))
        self.recon_scale = float(losses_cfg.get("recon_scale", 1.0))
        self.reward_scale = float(losses_cfg.get("reward_scale", 1.0))
        self.continue_scale = float(losses_cfg.get("continue_scale", 1.0))
        self.imagine_horizon = int(behavior_cfg.get("imagine_horizon", 8))
        self.lambda_return = float(behavior_cfg.get("lambda_return", 0.95))
        self.entropy_scale = float(behavior_cfg.get("entropy_scale", 1e-3))
        self.slow_value_reg = float(behavior_cfg.get("slow_value_reg", 1.0))

        return_norm_low = float(behavior_cfg.get("return_percentile_low", 5.0))
        return_norm_high = float(behavior_cfg.get("return_percentile_high", 95.0))
        return_norm_decay = float(behavior_cfg.get("return_percentile_decay", 0.99))
        self._return_normalizer = PercentileNormalizer(
            low=return_norm_low,
            high=return_norm_high,
            decay=return_norm_decay,
        )

        self.register_buffer("reward_bins", torch.linspace(bin_lo, bin_hi, num_bins))
        self.register_buffer("value_bins", torch.linspace(bin_lo, bin_hi, num_bins))

        self.encoder = StateEncoder(obs_dim, list(model_cfg.get("encoder_hidden_dims", [256, 256])), embed_dim)
        self.rssm = DiscreteRSSM(
            embed_dim=embed_dim,
            action_dim=action_dim,
            deter_dim=deter_dim,
            stoch_size=stoch_size,
            classes=classes,
            hidden_dims=rssm_hidden,
            unimix_ratio=float(latent_cfg.get("unimix_ratio", model_cfg.get("unimix_ratio", 0.01))),
        )

        feature_dim = deter_dim + stoch_size * classes
        self.decoder = ProjectionHead(feature_dim, decoder_hidden, obs_dim)
        self.reward_head = ProjectionHead(feature_dim, head_hidden, num_bins)
        self.continue_head = ProjectionHead(feature_dim, head_hidden, 1)
        self.actor = DreamerActor(
            feature_dim,
            actor_hidden,
            action_dim,
            min_std=float(behavior_cfg.get("min_std", 0.1)),
            max_std=float(behavior_cfg.get("max_std", 1.5)),
        )
        self.critic = ProjectionHead(feature_dim, critic_hidden, num_bins)
        zero_last_layer(self.reward_head)
        zero_last_layer(self.critic)
        self.critic_target = copy_module(self.critic)
        self._freeze_targets()

    def _freeze_targets(self) -> None:
        self.critic_target.eval()
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)

    def _decode_reward(self, reward_logits: torch.Tensor) -> torch.Tensor:
        return twohot_decode(reward_logits, self.reward_bins)

    def _decode_value(self, value_logits: torch.Tensor) -> torch.Tensor:
        return twohot_decode(value_logits, self.value_bins)

    def _discount_weights(self, continues: torch.Tensor) -> torch.Tensor:
        prefix = torch.ones(continues.shape[0], 1, 1, device=continues.device, dtype=continues.dtype)
        return torch.cumprod(torch.cat([prefix, self.discount * continues[:, :-1]], dim=1), dim=1)

    def world_model_parameters(self):
        modules = [self.encoder, self.rssm, self.decoder, self.reward_head, self.continue_head]
        for module in modules:
            yield from module.parameters()

    def actor_parameters(self):
        yield from self.actor.parameters()

    def critic_parameters(self):
        yield from self.critic.parameters()

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        return self.rssm.initial(batch_size, device)

    def feature(self, state: RSSMState) -> torch.Tensor:
        return self.rssm.features(state)

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder(symlog(observation))

    def observe_step(
        self,
        prev_state: RSSMState,
        prev_action: torch.Tensor,
        observation: torch.Tensor,
        deterministic: bool = False,
    ) -> RSSMState:
        embed = self.encode(observation)
        prior = self.rssm.img_step(prev_state, prev_action, deterministic=deterministic)
        return self.rssm.obs_step(prior, embed, deterministic=deterministic)

    def imagine_step(self, state: RSSMState, action: torch.Tensor, deterministic: bool = False) -> RSSMState:
        return self.rssm.img_step(state, action, deterministic=deterministic)

    def observe_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[RSSMState, RSSMState]:
        batch_size, horizon, _ = actions.shape
        embeds = self.encode(observations.reshape(-1, self.obs_dim)).view(batch_size, horizon + 1, -1)

        prev_state = self.initial_state(batch_size, observations.device)
        zero_action = torch.zeros(batch_size, self.action_dim, device=observations.device, dtype=actions.dtype)
        init_prior = self.rssm.img_step(prev_state, zero_action, deterministic=deterministic)
        init_post = self.rssm.obs_step(init_prior, embeds[:, 0], deterministic=deterministic)

        posts = [init_post]
        priors = []
        state = init_post
        for index in range(horizon):
            prior = self.rssm.img_step(state, actions[:, index], deterministic=deterministic)
            post = self.rssm.obs_step(prior, embeds[:, index + 1], deterministic=deterministic)
            priors.append(prior)
            posts.append(post)
            state = post
        return stack_states(posts), stack_states(priors)

    def _categorical_kl(self, lhs_logits: torch.Tensor, rhs_logits: torch.Tensor) -> torch.Tensor:
        lhs = torch.distributions.Categorical(logits=lhs_logits)
        rhs = torch.distributions.Categorical(logits=rhs_logits)
        return torch.distributions.kl_divergence(lhs, rhs).sum(dim=-1, keepdim=True)

    def _entropy(self, logits: torch.Tensor) -> torch.Tensor:
        dist = torch.distributions.Categorical(logits=logits)
        return dist.entropy().sum(dim=-1, keepdim=True)

    def compute_world_model_loss(self, batch) -> WorldModelLosses:
        posts, priors = self.observe_sequence(batch.observations, batch.actions, deterministic=False)
        next_posts = slice_state(posts, start=1)
        flat_posts = flatten_time(next_posts)
        features = self.feature(flat_posts)

        recon = self.decoder(features).view_as(batch.observations[:, 1:])
        reward_logits = self.reward_head(features).view(*batch.rewards.shape[:-1], -1)
        continue_logits = self.continue_head(features).view_as(batch.discounts)

        recon_loss = F.mse_loss(recon, symlog(batch.observations[:, 1:]))
        reward_loss = twohot_loss(reward_logits, batch.rewards, self.reward_bins)
        continue_loss = F.binary_cross_entropy_with_logits(continue_logits, batch.discounts.clamp(0.0, 1.0))

        dyn_kl = self._categorical_kl(next_posts.logits.detach(), priors.logits)
        rep_kl = self._categorical_kl(next_posts.logits, priors.logits.detach())
        dyn_kl = dyn_kl.clamp_min(self.free_nats).mean()
        rep_kl = rep_kl.clamp_min(self.free_nats).mean()
        prior_entropy = self._entropy(priors.logits).mean()
        post_entropy = self._entropy(next_posts.logits).mean()

        total = (
            self.recon_scale * recon_loss
            + self.reward_scale * reward_loss
            + self.continue_scale * continue_loss
            + self.dyn_scale * dyn_kl
            + self.rep_scale * rep_kl
        )
        return WorldModelLosses(
            total=total,
            recon_loss=recon_loss,
            reward_loss=reward_loss,
            continue_loss=continue_loss,
            dyn_kl=dyn_kl,
            rep_kl=rep_kl,
            prior_entropy=prior_entropy,
            post_entropy=post_entropy,
        )

    def _state_at(self, state: RSSMState, index: int) -> RSSMState:
        return RSSMState(
            deter=state.deter[:, index],
            stoch=state.stoch[:, index],
            logits=state.logits[:, index],
        )

    @torch.no_grad()
    def compute_diagnostics(self, batch, horizons: tuple[int, ...] = DIAGNOSTIC_HORIZONS) -> dict[str, torch.Tensor]:
        batch_size, horizon, _ = batch.actions.shape
        posts, _ = self.observe_sequence(batch.observations, batch.actions, deterministic=True)
        current_states = slice_state(posts, stop=-1)
        next_states = slice_state(posts, start=1)
        current_features = self.feature(flatten_time(current_states)).view(batch_size, horizon, -1)
        next_features = self.feature(flatten_time(next_states)).view(batch_size, horizon, -1)

        reward_pred = self._decode_reward(self.reward_head(next_features.reshape(-1, next_features.shape[-1]))).view_as(batch.rewards)
        reward_pred_mae = (reward_pred - batch.rewards).abs().mean()
        reward_pred_mse = F.mse_loss(reward_pred, batch.rewards)

        value_logits = self.critic(current_features.reshape(-1, current_features.shape[-1])).view(batch_size, horizon, -1)
        value_pred = self._decode_value(value_logits)
        next_value = self._decode_value(self.critic_target(next_features.reshape(-1, next_features.shape[-1]))).view(batch_size, horizon, 1)
        value_target = batch.rewards + self.discount * batch.discounts * next_value
        value_pred_mae = (value_pred - value_target).abs().mean()
        value_pred_mse = F.mse_loss(value_pred, value_target)

        requested = tuple(sorted(set(int(item) for item in horizons)))
        max_horizon = max(requested) if requested else 1
        mse_terms = {item: [] for item in requested}
        drift_terms = {item: [] for item in requested}
        nan = torch.full((), float("nan"), device=batch.observations.device)
        for start in range(horizon):
            rollout_state = self._state_at(current_states, start)
            origin_feature = self.feature(rollout_state)
            for delta in range(1, max_horizon + 1):
                action_index = start + delta - 1
                if action_index >= horizon:
                    break
                rollout_state = self.imagine_step(rollout_state, batch.actions[:, action_index], deterministic=True)
                rollout_feature = self.feature(rollout_state)
                if delta in mse_terms:
                    target_feature = self.feature(self._state_at(posts, start + delta)).detach()
                    mse_terms[delta].append((rollout_feature - target_feature).pow(2).sum(dim=-1).mean())
                    drift_terms[delta].append(torch.norm(rollout_feature - origin_feature, dim=-1).mean())

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

    @torch.no_grad()
    def behavior_start_states(self, batch) -> RSSMState:
        posts, _ = self.observe_sequence(batch.observations, batch.actions, deterministic=False)
        return detach_state(flatten_time(slice_state(posts, stop=-1)))

    def _stack_imagination(
        self,
        start_states: RSSMState,
        deterministic_actor: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current_features = []
        next_features = []
        rewards = []
        continues = []
        log_probs = []
        entropies = []

        state = start_states
        for _ in range(self.imagine_horizon):
            features = self.feature(state)
            action, log_prob, entropy = self.actor.sample(features, deterministic=deterministic_actor)
            next_state = self.imagine_step(state, action, deterministic=False)
            next_feat = self.feature(next_state)

            current_features.append(features)
            next_features.append(next_feat)
            rewards.append(self._decode_reward(self.reward_head(next_feat)))
            continues.append(torch.sigmoid(self.continue_head(next_feat)))
            log_probs.append(log_prob)
            entropies.append(entropy)
            state = next_state

        return (
            torch.stack(current_features, dim=1),
            torch.stack(next_features, dim=1),
            torch.stack(rewards, dim=1),
            torch.stack(continues, dim=1),
            torch.stack(log_probs, dim=1),
            torch.stack(entropies, dim=1),
        )

    def _lambda_returns(self, rewards: torch.Tensor, continues: torch.Tensor, next_values: torch.Tensor) -> torch.Tensor:
        returns = torch.zeros_like(rewards)
        acc = next_values[:, -1]
        for index in reversed(range(rewards.shape[1])):
            acc = rewards[:, index] + self.discount * continues[:, index] * (
                (1.0 - self.lambda_return) * next_values[:, index] + self.lambda_return * acc
            )
            returns[:, index] = acc
        return returns

    def compute_actor_loss(self, batch) -> BehaviorLosses:
        start_states = self.behavior_start_states(batch)
        current_features, next_features, rewards, continues, log_probs, entropies = self._stack_imagination(
            start_states,
            deterministic_actor=False,
        )
        with torch.no_grad():
            baseline_values = self._decode_value(self.critic_target(current_features))
            next_values = self._decode_value(self.critic_target(next_features))
            returns = self._lambda_returns(rewards, continues, next_values)
            self._return_normalizer.update(returns)
            advantages = (returns - baseline_values) / self._return_normalizer.scale()
            weights = self._discount_weights(continues)

        actor_loss = -(weights * (log_probs * advantages + self.entropy_scale * entropies)).mean()
        mean, std = self.actor(current_features.reshape(-1, current_features.shape[-1]))
        del mean
        return BehaviorLosses(
            actor_loss=actor_loss,
            critic_loss=torch.zeros((), device=actor_loss.device),
            imagine_return=returns.mean(),
            value_mean=baseline_values.mean(),
            policy_std=std.mean(),
        )

    def compute_critic_loss(self, batch) -> BehaviorLosses:
        start_states = self.behavior_start_states(batch)
        with torch.no_grad():
            current_features, next_features, rewards, continues, _, _ = self._stack_imagination(
                start_states,
                deterministic_actor=False,
            )
            next_values = self._decode_value(self.critic_target(next_features))
            returns = self._lambda_returns(rewards, continues, next_values)
            weights = self._discount_weights(continues)
            target_logits = self.critic_target(current_features)

        value_logits = self.critic(current_features)
        value_loss = twohot_cross_entropy(value_logits, returns, self.value_bins)
        slow_value_loss = soft_categorical_loss(value_logits, target_logits)
        critic_loss = (weights * (value_loss + self.slow_value_reg * slow_value_loss)).mean()

        values = self._decode_value(value_logits)
        mean, std = self.actor(current_features.reshape(-1, current_features.shape[-1]))
        del mean
        return BehaviorLosses(
            actor_loss=torch.zeros((), device=critic_loss.device),
            critic_loss=critic_loss,
            imagine_return=returns.mean(),
            value_mean=values.mean(),
            policy_std=std.mean(),
        )

    def freeze_for_actor(self):
        modules = [self.encoder, self.rssm, self.decoder, self.reward_head, self.continue_head, self.critic, self.critic_target]
        return freeze_parameters(modules)

    def freeze_for_critic(self):
        modules = [self.encoder, self.rssm, self.decoder, self.reward_head, self.continue_head, self.actor, self.critic_target]
        return freeze_parameters(modules)

    def update_targets(self, tau: float) -> None:
        soft_update(self.critic_target, self.critic, tau)
