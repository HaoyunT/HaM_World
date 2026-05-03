from __future__ import annotations

import torch


class TDMPC2Planner:
    def __init__(self, config: dict):
        planner_cfg = config["planner"]
        self.horizon = int(planner_cfg["horizon"])
        self.iterations = int(planner_cfg["iterations"])
        self.candidates = int(planner_cfg["candidates"])
        self.elite = int(planner_cfg["elite"])
        self.temperature = float(planner_cfg.get("temperature", 1.0))
        self.init_std = float(planner_cfg.get("init_std", 0.6))
        self.min_std = float(planner_cfg.get("min_std", 0.05))
        self.policy_trajectories = int(planner_cfg.get("policy_trajectories", 32))
        self.gamma = float(config["training"]["discount"])

    def shift_mean(self, mean: torch.Tensor | None) -> torch.Tensor | None:
        if mean is None:
            return None
        shifted = torch.zeros_like(mean)
        shifted[:-1] = mean[1:]
        return shifted

    @torch.no_grad()
    def _policy_rollouts(self, world_model, latent: torch.Tensor, count: int) -> torch.Tensor:
        rollout_latent = latent.repeat(count, 1)
        actions = []
        for _ in range(self.horizon):
            action, _, _ = world_model.policy.sample(rollout_latent, deterministic=False)
            actions.append(action)
            rollout_latent = world_model.imagine_step(rollout_latent, action)
        return torch.stack(actions, dim=1)

    @torch.no_grad()
    def evaluate_candidates(self, world_model, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        rollout_latent = latent.repeat(actions.shape[0], 1)
        returns = torch.zeros(actions.shape[0], device=latent.device)
        discount = 1.0
        for index in range(self.horizon):
            returns = returns + discount * world_model.predict_reward(rollout_latent, actions[:, index]).squeeze(-1)
            rollout_latent = world_model.imagine_step(rollout_latent, actions[:, index])
            discount *= self.gamma
        returns = returns + discount * world_model.terminal_value(rollout_latent).squeeze(-1)
        return returns

    @torch.no_grad()
    def plan(self, world_model, latent: torch.Tensor, prev_mean: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_dim = world_model.action_dim
        mean = self.shift_mean(prev_mean)
        if mean is None:
            mean = torch.zeros(self.horizon, action_dim, device=latent.device)
        std = torch.full_like(mean, self.init_std)
        best_value = torch.full((1,), -1e9, device=latent.device)

        for _ in range(self.iterations):
            random_count = max(0, self.candidates - self.policy_trajectories)
            samples = []
            if random_count > 0:
                noise = torch.randn(random_count, self.horizon, action_dim, device=latent.device)
                random_actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
                samples.append(random_actions)
            if self.policy_trajectories > 0:
                samples.append(self._policy_rollouts(world_model, latent, self.policy_trajectories))
            actions = torch.cat(samples, dim=0).clamp(-1.0, 1.0)
            values = self.evaluate_candidates(world_model, latent, actions)
            elite_values, elite_indices = torch.topk(values, self.elite, dim=0)
            elite_actions = actions[elite_indices]
            weights = torch.softmax(elite_values / self.temperature, dim=0).view(self.elite, 1, 1)
            mean = torch.sum(weights * elite_actions, dim=0)
            std = torch.sqrt(torch.sum(weights * (elite_actions - mean.unsqueeze(0)).pow(2), dim=0) + 1e-6)
            std = std.clamp_min(self.min_std)
            best_value = elite_values[:1]

        return mean[0], mean, best_value
