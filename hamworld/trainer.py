from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import torch

from hamworld.agent import HaMWorldAgent
from hamworld.runtime import (
    ExperimentLogger,
    EpisodeReplayBuffer,
    capture_rng_state,
    count_parameters,
    make_env,
    resolve_device,
    restore_rng_state,
    seed_everything,
)


class HaMWorldTrainer:
    def __init__(self, config: dict, resume_checkpoint: str | Path | None = None):
        self.config = config
        latent_cfg = self.config.setdefault("latent", {})
        latent_dim = int(latent_cfg.get("latent_dim", 48))
        q_dim = latent_cfg.get("q_dim")
        p_dim = latent_cfg.get("p_dim")
        c_dim = latent_cfg.get("c_dim")
        provided = sum(value is not None for value in (q_dim, p_dim, c_dim))
        if provided == 0:
            canonical_dim = latent_dim // 3
            latent_cfg["q_dim"] = canonical_dim
            latent_cfg["p_dim"] = canonical_dim
            latent_cfg["c_dim"] = latent_dim - (2 * canonical_dim)
        elif provided != 3:
            raise ValueError("Either provide all of q_dim/p_dim/c_dim or omit all three.")
        elif int(q_dim) + int(p_dim) + int(c_dim) != latent_dim:
            raise ValueError(
                f"Configured latent split q={q_dim}, p={p_dim}, c={c_dim} does not sum to latent_dim={latent_dim}."
            )
        self.task = config["task"]
        seed = int(config["experiment"]["seed"])
        seed_everything(seed)
        self.resume_checkpoint = Path(resume_checkpoint).expanduser().resolve() if resume_checkpoint is not None else None
        self.resume_payload = None
        self.resume_run_dir: Path | None = None
        if self.resume_checkpoint is not None:
            self.resume_payload = torch.load(self.resume_checkpoint, map_location="cpu", weights_only=False)
            self.resume_run_dir = self.resume_checkpoint.parent.parent

        self.env, self.env_spec = make_env(self.task, seed)
        self.eval_env, _ = make_env(self.task, seed + 1000)
        self.device = resolve_device(config["experiment"]["device"])
        self.sequence_length = int(config["training"]["sequence_length"])
        self.batch_size = int(config["training"]["batch_size"])
        self.total_steps = int(config["training"]["total_steps"])
        self.seed_steps = int(config["training"]["seed_steps"])
        self.train_every = int(config["training"]["train_every"])
        self.gradient_steps = int(config["training"]["gradient_steps"])
        self.eval_interval = int(config["training"].get("eval_interval", self.total_steps))
        self.eval_episodes = int(config["training"].get("eval_episodes", 3))
        self.checkpoint_interval = int(config["training"].get("checkpoint_interval", self.total_steps))

        logging_cfg = config.get("logging", {})
        self.progress_interval = max(1, int(logging_cfg.get("progress_interval", 1000)))
        self.logger = ExperimentLogger(
            output_root=config["experiment"]["output_dir"],
            algorithm=config["experiment"]["algorithm"],
            task_id=self.task["id"],
            seed=seed,
            console_interval=int(logging_cfg.get("console_log_interval", self.eval_interval)),
            episode_interval=int(logging_cfg.get("episode_log_interval", 5)),
            plot_interval=int(logging_cfg.get("plot_interval", self.eval_interval)),
            enable_plots=bool(logging_cfg.get("enable_plots", True)),
            run_dir=self.resume_run_dir,
            resume=self.resume_payload is not None,
        )
        self.logger.save_config(config)
        self.agent = HaMWorldAgent(
            config=config,
            obs_dim=self.env_spec.observation_shape[0],
            action_dim=self.env_spec.action_shape[0],
            action_low=self.env_spec.action_low,
            action_high=self.env_spec.action_high,
        )
        self.logger.info(
            f"Canonical HaM-World parameters: {count_parameters(self.agent.world_model):,} | "
            f"latent split q/p/c={latent_cfg['q_dim']}/{latent_cfg['p_dim']}/{latent_cfg['c_dim']}"
        )

        self.buffer = EpisodeReplayBuffer(capacity_steps=int(config["training"]["replay_capacity"]))

    def _infer_resume_step(self) -> int:
        if self.resume_checkpoint is None:
            return 0
        match = re.search(r"checkpoint_(\d+)\.pt$", self.resume_checkpoint.name)
        return int(match.group(1)) if match else 0

    def _default_train_episode_idx(self) -> int:
        return len(self.logger.episode_history.get("train/episode_return", []))

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
            "model": self.agent.world_model.state_dict(),
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
            "checkpoint_version": 2,
        }
        torch.save(payload, checkpoint_path)

    def _restore_from_checkpoint(self) -> tuple[np.ndarray, float, int, int, bool]:
        if self.resume_payload is None:
            return self.env.reset(seed=int(self.config["experiment"]["seed"])), 0.0, 0, 0, False

        payload = self.resume_payload
        restored_full_state = False
        if "agent_state" in payload:
            self.agent.load_state_dict(payload["agent_state"])
            restored_full_state = True
        elif "model" in payload:
            self.agent.world_model.load_state_dict(payload["model"])
            self.agent.reset()
            self.logger.info(
                f"[resume] legacy checkpoint detected at {self.resume_checkpoint}; "
                "loaded model weights only, training state was not available."
            )
        if "buffer_state" in payload:
            self.buffer.load_state_dict(payload["buffer_state"])
            restored_full_state = True
        trainer_state = payload.get("trainer_state")
        env_state = payload.get("env_state")
        if env_state is not None and hasattr(self.env, "set_state"):
            self.env.set_state(env_state)
            restored_full_state = True
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            restore_rng_state(rng_state)

        if trainer_state is not None:
            observation = np.asarray(trainer_state["observation"], dtype=np.float32)
            episode_return = float(trainer_state["episode_return"])
            episode_length = int(trainer_state["episode_length"])
            train_episode_idx = int(trainer_state["train_episode_idx"])
            start_step = int(trainer_state["step"])
            restored_full_state = True
        else:
            start_step = self._infer_resume_step()
            observation = self.env.reset(seed=int(self.config["experiment"]["seed"]) + start_step)
            self.buffer.start_episode(observation)
            self.agent.reset()
            episode_return = 0.0
            episode_length = 0
            train_episode_idx = self._default_train_episode_idx()

        self.logger.info(
            f"[resume] task={self.task['id']} step={start_step} "
            f"mode={'full-state' if restored_full_state else 'warm-start'} run_dir={self.logger.run_dir}"
        )
        return observation, episode_return, episode_length, train_episode_idx, True

    def run(self) -> None:
        if self.resume_payload is not None:
            observation, episode_return, episode_length, train_episode_idx, resumed = self._restore_from_checkpoint()
            start_step = self.resume_payload.get("trainer_state", {}).get("step", self._infer_resume_step())
            if not resumed:
                self.buffer.start_episode(observation)
        else:
            observation = self.env.reset(seed=int(self.config["experiment"]["seed"]))
            self.buffer.start_episode(observation)
            self.agent.reset()
            episode_return = 0.0
            episode_length = 0
            train_episode_idx = 0
            start_step = 0

        if start_step >= self.total_steps:
            self.logger.info(
                f"[resume] checkpoint step={start_step} already reached total_steps={self.total_steps}; nothing to do."
            )
            self.logger.finalize()
            return

        for step in range(start_step + 1, self.total_steps + 1):
            self.agent.set_training_step(step)
            if step <= self.seed_steps:
                action = self._sample_random_action()
            else:
                action = self.agent.act(observation, eval_mode=False)

            next_observation, reward, done, info = self.env.step(action)
            self.buffer.add(action, reward, next_observation, done, info.get("discount", 1.0))
            episode_return += reward
            episode_length += 1

            if step > self.seed_steps and step % self.train_every == 0 and self.buffer.can_sample(self.batch_size, self.sequence_length):
                metrics = self._run_updates(step)
                self.logger.log_metrics(step, metrics)

            if step % self.progress_interval == 0 and step % self.eval_interval != 0:
                phase = "seed" if step <= self.seed_steps else "train"
                self.logger.info(
                    f"[progress] step={step}/{self.total_steps} phase={phase} "
                    f"episode_return={episode_return:.3f} episode_length={episode_length} "
                    f"replay_steps={len(self.buffer)}"
                )

            if step % self.eval_interval == 0:
                eval_metrics = self.evaluate(step)
                self.logger.log_metrics(step, eval_metrics, force_console=True, force_plot=True)

            if done:
                train_episode_idx += 1
                self.logger.log_episode("train", train_episode_idx, step, episode_return, episode_length)
                observation = self.env.reset()
                self.buffer.start_episode(observation)
                self.agent.reset()
                episode_return = 0.0
                episode_length = 0
            else:
                observation = next_observation

            if step % self.checkpoint_interval == 0:
                self._save_checkpoint(step, observation, episode_return, episode_length, train_episode_idx)

        self.logger.finalize()

    def _sample_random_action(self):
        action = torch.rand(self.env_spec.action_shape[0]) * 2.0 - 1.0
        scaled = self.env_spec.action_low + (action.numpy() + 1.0) * 0.5 * (self.env_spec.action_high - self.env_spec.action_low)
        return scaled.astype("float32")

    def _run_updates(self, step: int) -> dict[str, float]:
        metrics = {}
        for _ in range(self.gradient_steps):
            batch = self.buffer.sample(self.batch_size, self.sequence_length, self.device)
            update = self.agent.update(batch, step=step)
            metrics = {
                "train/total_loss": update.total_loss,
                "train/repr_loss": update.repr_loss,
                "train/dyn_loss": update.dyn_loss,
                "train/reward_loss": update.reward_loss,
                "train/value_loss": update.value_loss,
                "train/value_ce_loss": update.value_ce_loss,
                "train/value_slow_loss": update.value_slow_loss,
                "train/reward_pred_mae": update.reward_pred_mae,
                "train/reward_pred_mse": update.reward_pred_mse,
                "train/value_pred_mae": update.value_pred_mae,
                "train/value_pred_mse": update.value_pred_mse,
                "train/roll_loss": update.roll_loss,
                "train/sa_loss": update.sa_loss,
                "train/energy_loss": update.energy_loss,
                "train/energy_drift": update.energy_drift,
                "train/policy_prior_loss": update.policy_prior_loss,
                "train/hamiltonian_loss": update.hamiltonian_loss,
                "train/temp_loss": update.temp_loss,
                "train/decouple_loss": update.decouple_loss,
                "train/c_sparse_loss": update.c_sparse_loss,
                "train/delta_q_norm": update.delta_q_norm,
                "train/delta_p_norm": update.delta_p_norm,
                "train/delta_c_norm": update.delta_c_norm,
                "train/q_norm": update.q_norm,
                "train/p_norm": update.p_norm,
                "train/c_norm": update.c_norm,
                "train/latent_norm": update.latent_norm,
                "train/control_norm": update.control_norm,
                "train/planner_latency_ms": update.planner_latency_ms,
                "train/hamiltonian_grad_norm": update.hamiltonian_grad_norm,
                "train/rollout_latent_drift": update.rollout_latent_drift,
                "train/pred_reward_mean": update.pred_reward_mean,
                "train/pred_value_mean": update.pred_value_mean,
                "train/one_step_latent_mse": update.one_step_latent_mse,
                "train/three_step_latent_mse": update.three_step_latent_mse,
                "train/roll_weight": update.roll_weight,
                "train/sa_weight": update.sa_weight,
                "train/energy_weight": update.energy_weight,
                "train/policy_weight": update.policy_weight,
                "train/hamiltonian_weight": update.hamiltonian_weight,
                "train/temp_weight": update.temp_weight,
                "train/decouple_weight": update.decouple_weight,
                "train/c_weight": update.c_weight,
                "train/alpha": update.alpha,
            }
            metrics.update(update.extra_metrics)
        return metrics

    @torch.no_grad()
    def evaluate(self, step: int) -> dict[str, float]:
        self.agent.set_training_step(step)
        returns = []
        lengths = []
        for eval_idx in range(1, self.eval_episodes + 1):
            observation = self.eval_env.reset()
            self.agent.reset()
            done = False
            episode_return = 0.0
            episode_length = 0
            while not done:
                action = self.agent.act(observation, eval_mode=True)
                observation, reward, done, _ = self.eval_env.step(action)
                episode_return += reward
                episode_length += 1
            returns.append(episode_return)
            lengths.append(episode_length)
            self.logger.log_episode("eval", eval_idx, step, episode_return, episode_length, force_plot=False)
        return {
            "eval/return_mean": float(sum(returns) / len(returns)),
            "eval/return_std": float(np.std(returns)),
            "eval/return_p25": float(np.quantile(returns, 0.25)),
            "eval/return_p50": float(np.quantile(returns, 0.50)),
            "eval/return_p75": float(np.quantile(returns, 0.75)),
            "eval/return_max": float(max(returns)),
            "eval/length_mean": float(sum(lengths) / len(lengths)),
        }
