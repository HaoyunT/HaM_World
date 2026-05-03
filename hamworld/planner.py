from __future__ import annotations

import torch


def _repeat_state(state: list[torch.Tensor], repeat: int) -> list[torch.Tensor]:
    return [tensor.repeat(repeat, 1) for tensor in state]


def _clone_state(state: list[torch.Tensor]) -> list[torch.Tensor]:
    return [tensor.clone() for tensor in state]


class CEMPlanner:
    def __init__(self, config: dict):
        planner_cfg = config["planner"]
        self.horizon = int(planner_cfg["horizon"])
        self.iterations = int(planner_cfg["iterations"])
        self.candidates = int(planner_cfg["candidates"])
        self.elite = int(planner_cfg["elite"])
        self.init_std = float(planner_cfg["init_std"])
        self.min_std = float(planner_cfg.get("min_std", 0.05))
        self.policy_trajectories = int(planner_cfg.get("policy_trajectories", 0))
        self.policy_noise = float(planner_cfg.get("policy_noise", 0.15))
        self.use_policy_mean_init = bool(planner_cfg.get("use_policy_mean_init", False))
        self.temperature = float(planner_cfg.get("temperature", 1.0))
        self.gamma = float(config["training"]["discount"])

    def shift_mean(self, mean: torch.Tensor | None) -> torch.Tensor | None:
        if mean is None:
            return None
        shifted = torch.zeros_like(mean)
        shifted[:-1] = mean[1:]
        return shifted

    @torch.no_grad()
    def _policy_rollouts(
        self,
        world_model,
        latent: torch.Tensor,
        memory_state: list[torch.Tensor],
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        count: int,
        deterministic: bool,
    ) -> torch.Tensor:
        if count <= 0:
            raise ValueError("count must be positive for policy rollouts.")
        rollout_latent = latent.repeat(count, 1)
        rollout_memory = _repeat_state(_clone_state(memory_state), count)
        action_scale = (action_high - action_low).unsqueeze(0)
        actions = []
        for _ in range(self.horizon):
            action = world_model.policy_action(rollout_latent, deterministic=deterministic)
            if not deterministic and self.policy_noise > 0.0:
                action = action + torch.randn_like(action) * self.policy_noise * action_scale
            action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0).clamp(action_low, action_high)
            actions.append(action)
            rollout_latent, rollout_memory = world_model.imagine_step(rollout_latent, action, rollout_memory)
        return torch.stack(actions, dim=1)

    @torch.no_grad()
    def _initial_mean(
        self,
        world_model,
        latent: torch.Tensor,
        memory_state: list[torch.Tensor],
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        init_mean: torch.Tensor | None,
    ) -> torch.Tensor:
        action_dim = action_low.numel()
        if init_mean is not None:
            return init_mean.clone().clamp(action_low, action_high)
        if self.use_policy_mean_init:
            return self._policy_rollouts(
                world_model,
                latent,
                memory_state,
                action_low,
                action_high,
                count=1,
                deterministic=True,
            ).squeeze(0)
        return torch.zeros(self.horizon, action_dim, device=latent.device)

    @torch.no_grad()
    def plan(
        self,
        world_model,
        latent: torch.Tensor,
        memory_state: list[torch.Tensor],
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        init_mean: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self._initial_mean(world_model, latent, memory_state, action_low, action_high, init_mean)
        action_scale = action_high - action_low
        std = torch.ones_like(mean) * self.init_std * action_scale
        min_std = self.min_std * action_scale
        best_value = torch.full((1,), -1e9, device=latent.device)

        for _ in range(self.iterations):
            random_count = max(0, self.candidates - self.policy_trajectories)
            samples = []
            if random_count > 0:
                noise = torch.randn(random_count, self.horizon, action_low.numel(), device=latent.device)
                random_actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
                samples.append(random_actions)
            if self.policy_trajectories > 0:
                samples.append(
                    self._policy_rollouts(
                        world_model,
                        latent,
                        memory_state,
                        action_low,
                        action_high,
                        count=self.policy_trajectories,
                        deterministic=False,
                    )
                )
            if not samples:
                raise ValueError("Planner needs at least one random or policy trajectory candidate.")

            actions = torch.cat(samples, dim=0)
            actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).clamp(action_low, action_high)
            values = self.evaluate_candidates(world_model, latent, memory_state, actions)
            values = torch.nan_to_num(values, nan=-1e6, posinf=1e6, neginf=-1e6)
            elite_values, elite_indices = torch.topk(values, self.elite, dim=0)
            elite_actions = actions[elite_indices]
            weights = torch.softmax(elite_values / self.temperature, dim=0).unsqueeze(-1).unsqueeze(-1)
            mean = torch.sum(weights * elite_actions, dim=0)
            std = torch.sqrt(torch.sum(weights * (elite_actions - mean.unsqueeze(0)).pow(2), dim=0) + 1e-6)
            mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0).clamp(action_low, action_high)
            std = torch.nan_to_num(std, nan=self.init_std, posinf=self.init_std, neginf=self.init_std)
            std = torch.maximum(std, min_std.unsqueeze(0))
            best_value = elite_values[:1]
        return mean, best_value

    @torch.no_grad()
    def evaluate_candidates(
        self,
        world_model,
        latent: torch.Tensor,
        memory_state: list[torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        candidate_count = actions.shape[0]
        rollout_latent = latent.repeat(candidate_count, 1)
        rollout_memory = _repeat_state(_clone_state(memory_state), candidate_count)
        returns = torch.zeros(candidate_count, device=latent.device)
        discount = 1.0
        for step in range(self.horizon):
            rollout_latent, rollout_memory = world_model.imagine_step(rollout_latent, actions[:, step], rollout_memory)
            returns = returns + discount * world_model.predict_reward(rollout_latent).squeeze(-1)
            discount *= self.gamma
        returns = returns + discount * world_model.predict_value(rollout_latent).squeeze(-1)
        return returns
