from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ppo.agent import PPOAgent
from ppo.runtime import (
    ExperimentLogger,
    capture_rng_state,
    make_env,
    restore_rng_state,
    seed_everything,
)


@dataclass
class RolloutStorage:
    observations: list[np.ndarray]
    actions: list[np.ndarray]
    log_probs: list[float]
    rewards: list[float]
    dones: list[bool]
    values: list[float]

    @classmethod
    def empty(cls) -> "RolloutStorage":
        return cls(observations=[], actions=[], log_probs=[], rewards=[], dones=[], values=[])

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def state_dict(self) -> dict[str, Any]:
        return {
            "observations": [item.copy() for item in self.observations],
            "actions": [item.copy() for item in self.actions],
            "log_probs": list(self.log_probs),
            "rewards": list(self.rewards),
            "dones": list(self.dones),
            "values": list(self.values),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any] | None) -> "RolloutStorage":
        storage = cls.empty()
        if not state:
            return storage
        storage.observations = [np.asarray(item, dtype=np.float32).copy() for item in state.get("observations", [])]
        storage.actions = [np.asarray(item, dtype=np.float32).copy() for item in state.get("actions", [])]
        storage.log_probs = [float(item) for item in state.get("log_probs", [])]
        storage.rewards = [float(item) for item in state.get("rewards", [])]
        storage.dones = [bool(item) for item in state.get("dones", [])]
        storage.values = [float(item) for item in state.get("values", [])]
        return storage

    def as_batch(self, last_value: float, gamma: float, gae_lambda: float, device: torch.device) -> dict[str, torch.Tensor]:
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0

        for index in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[index]
            next_value = last_value if index == len(rewards) - 1 else values[index + 1]
            delta = rewards[index] + gamma * next_non_terminal * next_value - values[index]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[index] = last_gae

        returns = advantages + values
        return {
            "observations": torch.as_tensor(np.stack(self.observations), dtype=torch.float32, device=device),
            "actions": torch.as_tensor(np.stack(self.actions), dtype=torch.float32, device=device),
            "log_probs": torch.as_tensor(np.asarray(self.log_probs, dtype=np.float32).reshape(-1, 1), dtype=torch.float32, device=device),
            "values": torch.as_tensor(values.reshape(-1, 1), dtype=torch.float32, device=device),
            "advantages": torch.as_tensor(advantages.reshape(-1, 1), dtype=torch.float32, device=device),
            "returns": torch.as_tensor(returns.reshape(-1, 1), dtype=torch.float32, device=device),
        }


class PPOTrainer:
    def __init__(self, config: dict[str, Any], resume_checkpoint: str | Path | None = None):
        self.config = config
        self.task = config["task"]
        self.seed = int(config["experiment"]["seed"])
        seed_everything(self.seed)

        self.resume_checkpoint = Path(resume_checkpoint).expanduser().resolve() if resume_checkpoint is not None else None
        self.resume_payload = None
        self.resume_run_dir: Path | None = None
        if self.resume_checkpoint is not None:
            self.resume_payload = torch.load(self.resume_checkpoint, map_location="cpu", weights_only=False)
            self.resume_run_dir = self.resume_checkpoint.parent.parent

        self.env, self.env_spec = make_env(self.task, self.seed)
        self.eval_env, _ = make_env(self.task, self.seed + 1000)
        self.total_steps = int(config["training"]["total_steps"])
        self.rollout_steps = int(config["training"].get("rollout_steps", 2048))
        self.eval_interval = int(config["training"].get("eval_interval", self.total_steps))
        self.eval_episodes = int(config["training"].get("eval_episodes", 3))
        self.checkpoint_interval = int(config["training"].get("checkpoint_interval", self.total_steps))
        self.discount = float(config["training"].get("discount", 0.99))
        self.gae_lambda = float(config.get("algorithm", {}).get("gae_lambda", 0.95))

        logging_cfg = config.get("logging", {})
        self.progress_interval = max(1, int(logging_cfg.get("progress_interval", 1000)))
        self.logger = ExperimentLogger(
            output_root=config["experiment"]["output_dir"],
            algorithm=config["experiment"]["algorithm"],
            task_id=self.task["id"],
            seed=self.seed,
            console_interval=int(logging_cfg.get("console_log_interval", self.eval_interval)),
            episode_interval=int(logging_cfg.get("episode_log_interval", 5)),
            plot_interval=int(logging_cfg.get("plot_interval", self.eval_interval)),
            enable_plots=bool(logging_cfg.get("enable_plots", True)),
            run_dir=self.resume_run_dir,
            resume=self.resume_payload is not None,
        )
        self.logger.save_config(config)

        self.agent = PPOAgent(
            config=config,
            obs_dim=self.env_spec.observation_shape[0],
            action_dim=self.env_spec.action_shape[0],
            action_low=self.env_spec.action_low,
            action_high=self.env_spec.action_high,
        )
        self.logger.info(f"PPO parameters: {self.agent.num_parameters:,}")
        self.rollout = RolloutStorage.empty()

    def _save_checkpoint(
        self,
        step: int,
        observation: np.ndarray,
        episode_return: float,
        episode_length: int,
        train_episode_idx: int,
    ) -> None:
        checkpoint_path = Path(self.logger.checkpoints_dir) / f"checkpoint_{step}.pt"
        payload = {
            "config": self.config,
            "agent_state": self.agent.state_dict(),
            "rollout_state": self.rollout.state_dict(),
            "trainer_state": {
                "step": int(step),
                "observation": np.asarray(observation, dtype=np.float32),
                "episode_return": float(episode_return),
                "episode_length": int(episode_length),
                "train_episode_idx": int(train_episode_idx),
            },
            "env_state": self.env.get_state() if hasattr(self.env, "get_state") else None,
            "rng_state": capture_rng_state(),
            "run_dir": str(self.logger.run_dir),
            "checkpoint_version": 1,
        }
        torch.save(payload, checkpoint_path)

    def _restore_from_checkpoint(self) -> tuple[int, np.ndarray, float, int, int]:
        if self.resume_payload is None:
            observation = self.env.reset(seed=self.seed)
            return 0, observation, 0.0, 0, 0

        payload = self.resume_payload
        self.agent.load_state_dict(payload["agent_state"])
        self.rollout = RolloutStorage.from_state_dict(payload.get("rollout_state"))

        env_state = payload.get("env_state")
        if env_state is not None and hasattr(self.env, "set_state"):
            self.env.set_state(env_state)
        restore_rng_state(payload.get("rng_state"))

        trainer_state = payload.get("trainer_state", {})
        observation = np.asarray(trainer_state.get("observation"), dtype=np.float32)
        episode_return = float(trainer_state.get("episode_return", 0.0))
        episode_length = int(trainer_state.get("episode_length", 0))
        train_episode_idx = int(trainer_state.get("train_episode_idx", 0))
        start_step = int(trainer_state.get("step", 0))
        self.logger.info(f"[resume] task={self.task['id']} step={start_step} run_dir={self.logger.run_dir}")
        return start_step, observation, episode_return, episode_length, train_episode_idx

    def run(self) -> None:
        start_step, observation, episode_return, episode_length, train_episode_idx = self._restore_from_checkpoint()

        if start_step >= self.total_steps:
            self.logger.info(
                f"[resume] checkpoint step={start_step} already reached total_steps={self.total_steps}; nothing to do."
            )
            self.logger.finalize()
            return

        for step in range(start_step + 1, self.total_steps + 1):
            action, normalized_action, log_prob, value = self.agent.act(observation, eval_mode=False)
            next_observation, reward, done, _ = self.env.step(action)
            self.rollout.add(observation, normalized_action, log_prob, reward, done, value)

            episode_return += reward
            episode_length += 1
            observation = next_observation

            if done:
                train_episode_idx += 1
                self.logger.log_episode("train", train_episode_idx, step, episode_return, episode_length)
                observation = self.env.reset()
                episode_return = 0.0
                episode_length = 0

            if step % self.progress_interval == 0 and step % self.eval_interval != 0:
                self.logger.info(
                    f"[progress] step={step}/{self.total_steps} episode_return={episode_return:.3f} "
                    f"episode_length={episode_length} rollout_size={len(self.rollout)}"
                )

            if len(self.rollout) >= self.rollout_steps or step == self.total_steps:
                last_value = 0.0 if done else self.agent.predict_value(observation)
                batch = self.rollout.as_batch(last_value, self.discount, self.gae_lambda, self.agent.device)
                update = self.agent.update(batch)
                self.logger.log_metrics(
                    step,
                    {
                        "train/total_loss": update.total_loss,
                        "train/policy_loss": update.policy_loss,
                        "train/value_loss": update.value_loss,
                        "train/entropy": update.entropy,
                        "train/approx_kl": update.approx_kl,
                        "train/clip_fraction": update.clip_fraction,
                        "train/learning_rate": update.learning_rate,
                        "train/explained_variance": update.explained_variance,
                        "train/grad_norm": update.grad_norm,
                        "train/value_mean": update.value_mean,
                        "train/advantage_mean": update.advantage_mean,
                        "train/return_mean": update.return_mean,
                    },
                )
                self.rollout.clear()

            if step % self.eval_interval == 0:
                eval_metrics = self.evaluate(step)
                self.logger.log_metrics(step, eval_metrics, force_console=True, force_plot=True)

            if step % self.checkpoint_interval == 0:
                self._save_checkpoint(step, observation, episode_return, episode_length, train_episode_idx)

        self.logger.finalize()

    @torch.no_grad()
    def evaluate(self, step: int) -> dict[str, float]:
        returns: list[float] = []
        lengths: list[int] = []
        for episode_idx in range(1, self.eval_episodes + 1):
            observation = self.eval_env.reset()
            done = False
            episode_return = 0.0
            episode_length = 0
            while not done:
                action, _, _, _ = self.agent.act(observation, eval_mode=True)
                observation, reward, done, _ = self.eval_env.step(action)
                episode_return += reward
                episode_length += 1
            returns.append(episode_return)
            lengths.append(episode_length)
            self.logger.log_episode("eval", episode_idx, step, episode_return, episode_length, force_console=False)
        return {
            "eval/return_mean": float(np.mean(returns)),
            "eval/return_std": float(np.std(returns)),
            "eval/length_mean": float(np.mean(lengths)),
        }
