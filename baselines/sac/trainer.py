from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from sac.agent import SACAgent
from sac.runtime import (
    ExperimentLogger,
    capture_rng_state,
    make_env,
    restore_rng_state,
    seed_everything,
)


class TransitionReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.index = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.index] = np.asarray(observation, dtype=np.float32)
        self.actions[self.index] = np.asarray(action, dtype=np.float32)
        self.rewards[self.index] = float(reward)
        self.next_observations[self.index] = np.asarray(next_observation, dtype=np.float32)
        self.dones[self.index] = float(done)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "observations": torch.as_tensor(self.observations[indices], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[indices], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            "next_observations": torch.as_tensor(self.next_observations[indices], dtype=torch.float32, device=device),
            "dones": torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "index": self.index,
            "size": self.size,
            "observations": self.observations[: self.size].copy(),
            "actions": self.actions[: self.size].copy(),
            "rewards": self.rewards[: self.size].copy(),
            "next_observations": self.next_observations[: self.size].copy(),
            "dones": self.dones[: self.size].copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.capacity = int(state["capacity"])
        self.obs_dim = int(state["obs_dim"])
        self.action_dim = int(state["action_dim"])
        self.__init__(self.capacity, self.obs_dim, self.action_dim)
        self.size = int(state["size"])
        self.index = int(state["index"])
        self.observations[: self.size] = np.asarray(state["observations"], dtype=np.float32)
        self.actions[: self.size] = np.asarray(state["actions"], dtype=np.float32)
        self.rewards[: self.size] = np.asarray(state["rewards"], dtype=np.float32)
        self.next_observations[: self.size] = np.asarray(state["next_observations"], dtype=np.float32)
        self.dones[: self.size] = np.asarray(state["dones"], dtype=np.float32)


class SACTrainer:
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

        training_cfg = config["training"]
        self.total_steps = int(training_cfg["total_steps"])
        self.seed_steps = int(training_cfg.get("seed_steps", 5000))
        self.batch_size = int(training_cfg.get("batch_size", 256))
        self.train_every = int(training_cfg.get("train_every", 1))
        self.gradient_steps = int(training_cfg.get("gradient_steps", 1))
        self.eval_interval = int(training_cfg.get("eval_interval", self.total_steps))
        self.eval_episodes = int(training_cfg.get("eval_episodes", 3))
        self.checkpoint_interval = int(training_cfg.get("checkpoint_interval", self.total_steps))

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

        self.agent = SACAgent(
            config=config,
            obs_dim=self.env_spec.observation_shape[0],
            action_dim=self.env_spec.action_shape[0],
            action_low=self.env_spec.action_low,
            action_high=self.env_spec.action_high,
        )
        self.buffer = TransitionReplayBuffer(
            capacity=int(training_cfg.get("replay_capacity", 300000)),
            obs_dim=self.env_spec.observation_shape[0],
            action_dim=self.env_spec.action_shape[0],
        )
        self.logger.info(f"SAC parameters: {self.agent.num_parameters:,}")

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
            "buffer_state": self.buffer.state_dict(),
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
        self.buffer.load_state_dict(payload["buffer_state"])

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
            if step <= self.seed_steps:
                action = self.agent.sample_random_action()
                normalized_action = self.agent.normalize_action(action)
            else:
                action, normalized_action = self.agent.act(observation, eval_mode=False)

            next_observation, reward, done, _ = self.env.step(action)
            self.buffer.add(observation, normalized_action, reward, next_observation, done)

            episode_return += reward
            episode_length += 1

            if step > self.seed_steps and step % self.train_every == 0 and len(self.buffer) >= self.batch_size:
                metrics = {}
                for _ in range(self.gradient_steps):
                    batch = self.buffer.sample(self.batch_size, self.agent.device)
                    update = self.agent.update(batch, step=step)
                    metrics = {
                        "train/critic_loss": update.critic_loss,
                        "train/actor_loss": update.actor_loss,
                        "train/alpha_loss": update.alpha_loss,
                        "train/alpha": update.alpha,
                        "train/q1_mean": update.q1_mean,
                        "train/q2_mean": update.q2_mean,
                        "train/target_q_mean": update.target_q_mean,
                        "train/log_prob_mean": update.log_prob_mean,
                        "train/reward_mean": update.reward_mean,
                    }
                if metrics:
                    self.logger.log_metrics(step, metrics)

            if step % self.progress_interval == 0 and step % self.eval_interval != 0:
                phase = "seed" if step <= self.seed_steps else "train"
                self.logger.info(
                    f"[progress] step={step}/{self.total_steps} phase={phase} "
                    f"episode_return={episode_return:.3f} episode_length={episode_length} replay_size={len(self.buffer)}"
                )

            if step % self.eval_interval == 0:
                eval_metrics = self.evaluate(step)
                self.logger.log_metrics(step, eval_metrics, force_console=True, force_plot=True)

            if done:
                train_episode_idx += 1
                self.logger.log_episode("train", train_episode_idx, step, episode_return, episode_length)
                observation = self.env.reset()
                episode_return = 0.0
                episode_length = 0
            else:
                observation = next_observation

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
                action, _ = self.agent.act(observation, eval_mode=True)
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
