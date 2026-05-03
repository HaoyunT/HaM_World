from __future__ import annotations

import copy
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    if torch.cuda.is_available():
        torch_cuda_state = state.get("torch_cuda")
        if torch_cuda_state is not None:
            torch.cuda.set_rng_state_all(torch_cuda_state)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    activation: type[nn.Module] = nn.SiLU,
    output_activation: type[nn.Module] | None = None,
) -> nn.Sequential:
    dims = [input_dim, *hidden_dims, output_dim]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(activation())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.lerp_(source_param.data, tau)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def module_summary(module: nn.Module) -> str:
    lines = [f"{name}: {submodule.__class__.__name__}" for name, submodule in module.named_children()]
    lines.append(f"trainable_parameters: {count_parameters(module)}")
    return "\n".join(lines)


def orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain)
        nn.init.zeros_(module.bias)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must decode to a dictionary.")
    data.setdefault("_config_path", str(path))
    return data


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _set_by_dotted_path(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected key=value.")
        dotted_key, raw_value = override.split("=", 1)
        _set_by_dotted_path(updated, dotted_key, _coerce_value(raw_value))
    return updated


def expand_task_runs(config: dict[str, Any], task_filter: str | None = None) -> list[dict[str, Any]]:
    tasks = config.get("benchmark", {}).get("tasks", [])
    if not tasks:
        return [config]

    runs = []
    for task in tasks:
        task_id = task.get("id")
        if task_filter and task_id != task_filter:
            continue
        run_config = copy.deepcopy(config)
        run_config["task"] = task
        run_config["experiment"] = copy.deepcopy(config.get("experiment", {}))
        base_name = run_config["experiment"].get("name", "experiment")
        run_config["experiment"]["resolved_name"] = f"{base_name}_{task_id}"
        runs.append(run_config)

    if not runs:
        raise ValueError(f"Task '{task_filter}' was not found in the config task list.")
    return runs


@dataclass
class EnvSpec:
    observation_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    action_low: np.ndarray
    action_high: np.ndarray
    max_episode_steps: int


def _flatten_observation(observation: Any) -> np.ndarray:
    if isinstance(observation, dict):
        parts = [np.asarray(value, dtype=np.float32).reshape(-1) for key, value in sorted(observation.items())]
        return np.concatenate(parts, axis=0).astype(np.float32)
    return np.asarray(observation, dtype=np.float32).reshape(-1)


class DummyContinuousEnv:
    def __init__(self, action_repeat: int = 1, episode_length: int = 200):
        self.action_repeat = int(action_repeat)
        self.episode_length = int(episode_length)
        self.state = np.zeros(4, dtype=np.float32)
        self.steps = 0
        self.action_low = -np.ones(2, dtype=np.float32)
        self.action_high = np.ones(2, dtype=np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.uniform(-0.1, 0.1, size=4).astype(np.float32)
        self.steps = 0
        return self.state.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=self.action_high, neginf=self.action_low)
        action = np.clip(action, self.action_low, self.action_high)
        reward = 0.0
        done = False
        for _ in range(self.action_repeat):
            force = np.pad(action, (0, 2), mode="constant")
            self.state = 0.97 * self.state + 0.15 * force
            reward += float(1.0 - np.square(self.state).mean() - 0.1 * np.square(action).mean())
            self.steps += 1
            done = self.steps >= self.episode_length
            if done:
                break
        return self.state.copy(), reward, done, {"discount": 0.0 if done else 1.0}

    def spec(self) -> EnvSpec:
        return EnvSpec(
            observation_shape=(4,),
            action_shape=(2,),
            action_low=self.action_low.copy(),
            action_high=self.action_high.copy(),
            max_episode_steps=self.episode_length,
        )

    def get_state(self) -> dict[str, Any]:
        return {
            "state": self.state.copy(),
            "steps": int(self.steps),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        self.state = np.asarray(state["state"], dtype=np.float32).copy()
        self.steps = int(state["steps"])


class DMControlEnv:
    def __init__(self, task_config: dict[str, Any], seed: int):
        try:
            from dm_control import suite
        except ImportError as exc:
            raise ImportError("dm-control is required for DMControl tasks. Install dependencies first.") from exc

        self.action_repeat = int(task_config.get("action_repeat", 1))
        self.max_episode_steps = int(task_config.get("episode_length", 1000))
        self.initial_seed = int(seed)
        self.env = suite.load(
            domain_name=task_config["domain"],
            task_name=task_config["task"],
            task_kwargs={"random": seed},
            environment_kwargs={"flat_observation": True},
        )
        action_spec = self.env.action_spec()
        self.action_low = np.asarray(action_spec.minimum, dtype=np.float32)
        self.action_high = np.asarray(action_spec.maximum, dtype=np.float32)
        self.steps = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        del seed
        self.steps = 0
        time_step = self.env.reset()
        return _flatten_observation(time_step.observation)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=self.action_high, neginf=self.action_low)
        action = np.clip(action, self.action_low, self.action_high)
        reward = 0.0
        discount = 1.0
        done = False
        time_step = None
        for _ in range(self.action_repeat):
            time_step = self.env.step(action)
            reward += float(time_step.reward or 0.0)
            discount = float(time_step.discount if time_step.discount is not None else 1.0)
            self.steps += 1
            done = bool(time_step.last()) or self.steps >= self.max_episode_steps
            if done:
                break
        assert time_step is not None
        return _flatten_observation(time_step.observation), reward, done, {"discount": discount}

    def spec(self) -> EnvSpec:
        observation = _flatten_observation(self.env.reset().observation)
        return EnvSpec(
            observation_shape=observation.shape,
            action_shape=self.action_low.shape,
            action_low=self.action_low.copy(),
            action_high=self.action_high.copy(),
            max_episode_steps=self.max_episode_steps,
        )

    def get_state(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "physics_state": np.asarray(self.env.physics.get_state()).copy(),
            "steps": int(self.steps),
        }
        task_random = getattr(self.env.task, "_random", None)
        if task_random is not None and hasattr(task_random, "get_state"):
            payload["task_random_state"] = task_random.get_state()
        return payload

    def set_state(self, state: dict[str, Any]) -> None:
        self.reset(seed=self.initial_seed)
        self.steps = int(state.get("steps", 0))
        physics_state = state.get("physics_state")
        if physics_state is not None:
            self.env.physics.set_state(np.asarray(physics_state))
            self.env.physics.forward()
        task_random_state = state.get("task_random_state")
        task_random = getattr(self.env.task, "_random", None)
        if task_random_state is not None and task_random is not None and hasattr(task_random, "set_state"):
            task_random.set_state(task_random_state)


class GymnasiumEnv:
    def __init__(self, task_config: dict[str, Any], seed: int):
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError("gymnasium[mujoco] is required for MuJoCo tasks.") from exc
        try:
            import gymnasium_robotics  # noqa: F401
        except ImportError:
            pass

        self.action_repeat = int(task_config.get("action_repeat", 1))
        env_kwargs = dict(task_config.get("env_kwargs", {}))
        self.env = gym.make(task_config["env_name"], **env_kwargs)
        self.max_episode_steps = int(task_config.get("episode_length", self.env.spec.max_episode_steps))
        self.initial_seed = seed
        self.action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        self.steps = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        self.steps = 0
        observation, _ = self.env.reset(seed=self.initial_seed if seed is None else seed)
        return _flatten_observation(observation)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=self.action_high, neginf=self.action_low)
        action = np.clip(action, self.action_low, self.action_high)
        reward = 0.0
        terminated = False
        truncated = False
        observation = None
        for _ in range(self.action_repeat):
            observation, step_reward, terminated, truncated, _ = self.env.step(action)
            reward += float(step_reward)
            self.steps += 1
            if terminated or truncated or self.steps >= self.max_episode_steps:
                break
        assert observation is not None
        done = bool(terminated or truncated or self.steps >= self.max_episode_steps)
        discount = 0.0 if terminated else 1.0
        return _flatten_observation(observation), reward, done, {"discount": discount}

    def spec(self) -> EnvSpec:
        observation, _ = self.env.reset(seed=self.initial_seed)
        return EnvSpec(
            observation_shape=_flatten_observation(observation).shape,
            action_shape=self.action_low.shape,
            action_low=self.action_low.copy(),
            action_high=self.action_high.copy(),
            max_episode_steps=self.max_episode_steps,
        )

    def get_state(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"steps": int(self.steps)}
        unwrapped = getattr(self.env, "unwrapped", self.env)
        np_random = getattr(unwrapped, "np_random", None)
        if np_random is not None and hasattr(np_random, "bit_generator"):
            payload["np_random_state"] = copy.deepcopy(np_random.bit_generator.state)
        if hasattr(unwrapped, "data") and hasattr(unwrapped, "set_state"):
            payload["qpos"] = np.asarray(unwrapped.data.qpos, dtype=np.float64).copy()
            payload["qvel"] = np.asarray(unwrapped.data.qvel, dtype=np.float64).copy()
        classic_state = getattr(unwrapped, "state", None)
        if classic_state is not None:
            payload["classic_state"] = np.asarray(classic_state, dtype=np.float64).copy()
        goal = getattr(unwrapped, "goal", None)
        if goal is not None:
            payload["goal"] = np.asarray(goal).copy()
        elapsed_steps = getattr(self.env, "_elapsed_steps", None)
        if elapsed_steps is not None:
            payload["elapsed_steps"] = int(elapsed_steps)
        return payload

    def set_state(self, state: dict[str, Any]) -> None:
        self.reset(seed=self.initial_seed)
        self.steps = int(state.get("steps", 0))
        unwrapped = getattr(self.env, "unwrapped", self.env)
        np_random_state = state.get("np_random_state")
        np_random = getattr(unwrapped, "np_random", None)
        if np_random_state is not None and np_random is not None and hasattr(np_random, "bit_generator"):
            np_random.bit_generator.state = copy.deepcopy(np_random_state)
        if "qpos" in state and "qvel" in state and hasattr(unwrapped, "set_state"):
            unwrapped.set_state(np.asarray(state["qpos"]), np.asarray(state["qvel"]))
        elif "classic_state" in state and hasattr(unwrapped, "state"):
            unwrapped.state = np.asarray(state["classic_state"]).copy()
        if "goal" in state and hasattr(unwrapped, "goal"):
            unwrapped.goal = np.asarray(state["goal"]).copy()
        if "elapsed_steps" in state and hasattr(self.env, "_elapsed_steps"):
            self.env._elapsed_steps = int(state["elapsed_steps"])


def make_env(task_config: dict[str, Any], seed: int):
    suite_name = task_config["suite"].lower()
    if suite_name == "dmcontrol":
        env = DMControlEnv(task_config, seed)
    elif suite_name == "mujoco":
        env = GymnasiumEnv(task_config, seed)
    elif suite_name == "dummy":
        env = DummyContinuousEnv(
            action_repeat=task_config.get("action_repeat", 1),
            episode_length=task_config.get("episode_length", 200),
        )
    else:
        raise ValueError(f"Unsupported benchmark suite '{suite_name}'.")
    return env, env.spec()


@dataclass
class ReplayBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    dones: torch.Tensor


class EpisodeReplayBuffer:
    def __init__(self, capacity_steps: int):
        self.capacity_steps = int(capacity_steps)
        self.episodes: deque[dict[str, np.ndarray]] = deque()
        self.num_steps = 0
        self.current_episode: dict[str, list[np.ndarray]] | None = None

    def start_episode(self, initial_observation: np.ndarray) -> None:
        self.current_episode = {
            "observations": [np.asarray(initial_observation, dtype=np.float32)],
            "actions": [],
            "rewards": [],
            "discounts": [],
            "dones": [],
        }

    def add(
        self,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        discount: float,
    ) -> None:
        if self.current_episode is None:
            raise RuntimeError("Call start_episode() before adding transitions.")
        self.current_episode["actions"].append(np.asarray(action, dtype=np.float32))
        self.current_episode["rewards"].append(np.asarray([reward], dtype=np.float32))
        self.current_episode["discounts"].append(np.asarray([discount], dtype=np.float32))
        self.current_episode["dones"].append(np.asarray([float(done)], dtype=np.float32))
        self.current_episode["observations"].append(np.asarray(next_observation, dtype=np.float32))
        self.num_steps += 1
        if done:
            self._finalize_current_episode()

    def _finalize_current_episode(self) -> None:
        if self.current_episode is None:
            return
        episode = {key: np.stack(value, axis=0) for key, value in self.current_episode.items()}
        self.episodes.append(episode)
        self.current_episode = None
        while self.num_steps > self.capacity_steps and self.episodes:
            oldest = self.episodes.popleft()
            self.num_steps -= len(oldest["actions"])

    def __len__(self) -> int:
        return self.num_steps

    @staticmethod
    def _clone_episode(episode: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: np.asarray(value, dtype=np.float32).copy() for key, value in episode.items()}

    @staticmethod
    def _clone_current_episode(current_episode: dict[str, list[np.ndarray]] | None) -> dict[str, list[np.ndarray]] | None:
        if current_episode is None:
            return None
        return {
            key: [np.asarray(value, dtype=np.float32).copy() for value in values]
            for key, values in current_episode.items()
        }

    def _candidate_episodes(self) -> list[dict[str, np.ndarray]]:
        candidates = list(self.episodes)
        if self.current_episode is not None and len(self.current_episode["actions"]) > 0:
            candidates.append({key: np.stack(value, axis=0) for key, value in self.current_episode.items()})
        return candidates

    def can_sample(self, batch_size: int, sequence_length: int) -> bool:
        valid = [episode for episode in self._candidate_episodes() if len(episode["actions"]) >= sequence_length]
        return len(valid) > 0 and self.num_steps >= batch_size * sequence_length

    def sample(self, batch_size: int, sequence_length: int, device: torch.device) -> ReplayBatch:
        valid = [episode for episode in self._candidate_episodes() if len(episode["actions"]) >= sequence_length]
        if not valid:
            raise RuntimeError("Replay buffer does not contain a full sequence yet.")

        obs_batch, action_batch, reward_batch, discount_batch, done_batch = [], [], [], [], []
        for _ in range(batch_size):
            episode = valid[np.random.randint(0, len(valid))]
            start = np.random.randint(0, len(episode["actions"]) - sequence_length + 1)
            stop = start + sequence_length
            obs_batch.append(episode["observations"][start : stop + 1])
            action_batch.append(episode["actions"][start:stop])
            reward_batch.append(episode["rewards"][start:stop])
            discount_batch.append(episode["discounts"][start:stop])
            done_batch.append(episode["dones"][start:stop])

        return ReplayBatch(
            observations=torch.as_tensor(np.stack(obs_batch), dtype=torch.float32, device=device),
            actions=torch.as_tensor(np.stack(action_batch), dtype=torch.float32, device=device),
            rewards=torch.as_tensor(np.stack(reward_batch), dtype=torch.float32, device=device),
            discounts=torch.as_tensor(np.stack(discount_batch), dtype=torch.float32, device=device),
            dones=torch.as_tensor(np.stack(done_batch), dtype=torch.float32, device=device),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity_steps": int(self.capacity_steps),
            "num_steps": int(self.num_steps),
            "episodes": [self._clone_episode(episode) for episode in self.episodes],
            "current_episode": self._clone_current_episode(self.current_episode),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.capacity_steps = int(state.get("capacity_steps", self.capacity_steps))
        self.num_steps = int(state.get("num_steps", 0))
        self.episodes = deque(
            [self._clone_episode(episode) for episode in state.get("episodes", [])]
        )
        self.current_episode = self._clone_current_episode(state.get("current_episode"))
        while self.num_steps > self.capacity_steps and self.episodes:
            oldest = self.episodes.popleft()
            self.num_steps -= len(oldest["actions"])


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _plot_series_grid(plt, data: dict[str, list[tuple[int, float]]], title: str, output_path: Path) -> None:
    if not data:
        return
    names = sorted(data.keys())
    cols = 2
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.5 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for axis, name in zip(axes, names):
        pairs = data[name]
        steps = [step for step, _ in pairs]
        values = [value for _, value in pairs]
        axis.plot(steps, values, linewidth=2.0)
        axis.set_title(name)
        axis.set_xlabel("step")
        axis.grid(alpha=0.3)
    for axis in axes[len(names) :]:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_metric_plots(
    metric_history: dict[str, list[tuple[int, float]]],
    episode_history: dict[str, list[tuple[int, float]]],
    plots_dir: Path,
) -> bool:
    plt = _try_import_matplotlib()
    if plt is None:
        return False

    plots_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for name, values in metric_history.items():
        prefix = name.split("/", 1)[0] if "/" in name else "misc"
        grouped.setdefault(prefix, {})[name] = values

    for prefix, data in grouped.items():
        _plot_series_grid(plt, data, f"{prefix.title()} Metrics", plots_dir / f"{prefix}_metrics.png")

    if "train/total_loss" in metric_history or "eval/return_mean" in metric_history:
        summary = {}
        if "train/total_loss" in metric_history:
            summary["train/total_loss"] = metric_history["train/total_loss"]
        if "eval/return_mean" in metric_history:
            summary["eval/return_mean"] = metric_history["eval/return_mean"]
        _plot_series_grid(plt, summary, "Training Summary", plots_dir / "summary.png")

    if episode_history:
        _plot_series_grid(plt, episode_history, "Episode Returns", plots_dir / "episode_returns.png")
    return True


class ExperimentLogger:
    def __init__(
        self,
        output_root: str | os.PathLike[str],
        algorithm: str,
        task_id: str,
        seed: int,
        console_interval: int = 1000,
        episode_interval: int = 10,
        plot_interval: int = 10000,
        enable_plots: bool = True,
        run_dir: str | os.PathLike[str] | None = None,
        resume: bool = False,
    ):
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_root_path = Path(output_root).expanduser()
            if not output_root_path.is_absolute():
                output_root_path = (REPO_ROOT / output_root_path).resolve()
            algorithm_root = output_root_path if output_root_path.name == algorithm else output_root_path / algorithm
            self.run_dir = algorithm_root / task_id / f"seed_{seed}_{timestamp}"
        else:
            self.run_dir = Path(run_dir).expanduser()
            if not self.run_dir.is_absolute():
                self.run_dir = (REPO_ROOT / self.run_dir).resolve()
        self.logs_dir = self.run_dir / "logs"
        self.plots_dir = self.run_dir / "plots"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.artifacts_dir = self.run_dir / "artifacts"
        self._ensure_directories()

        self.metrics_path = self.logs_dir / "metrics.jsonl"
        self.episodes_path = self.logs_dir / "episodes.jsonl"
        self.text_log_path = self.logs_dir / "run.log"
        self.console_interval = max(1, int(console_interval))
        self.episode_interval = max(1, int(episode_interval))
        self.plot_interval = max(1, int(plot_interval))
        self.enable_plots = bool(enable_plots)
        self.metric_history: dict[str, list[tuple[int, float]]] = {}
        self.episode_history: dict[str, list[tuple[int, float]]] = {}
        self._matplotlib_warned = False
        if resume:
            self._load_existing_history()

    def _ensure_directories(self) -> None:
        for path in (self.run_dir, self.logs_dir, self.plots_dir, self.checkpoints_dir, self.artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: dict[str, Any]) -> None:
        self._ensure_directories()
        with (self.artifacts_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    def _load_existing_history(self) -> None:
        if self.metrics_path.exists():
            with self.metrics_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    step = int(payload["step"])
                    for key, value in payload.items():
                        if key == "step":
                            continue
                        self.metric_history.setdefault(key, []).append((step, float(value)))
        if self.episodes_path.exists():
            with self.episodes_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    key = f"{payload['split']}/episode_return"
                    self.episode_history.setdefault(key, []).append((int(payload["step"]), float(payload["return"])))

    def log_metrics(
        self,
        step: int,
        metrics: dict[str, float],
        force_console: bool = False,
        force_plot: bool = False,
    ) -> None:
        self._ensure_directories()
        payload = {"step": int(step), **{key: float(value) for key, value in metrics.items()}}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        for key, value in metrics.items():
            self.metric_history.setdefault(key, []).append((int(step), float(value)))

        should_print = force_console or any(key.startswith("eval/") for key in metrics) or step % self.console_interval == 0
        if should_print:
            summary = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            self.info(f"[step {step}] {summary}")

        should_plot = self.enable_plots and (
            force_plot or any(key.startswith("eval/") for key in metrics) or step % self.plot_interval == 0
        )
        if should_plot:
            self.update_plots()

    def log_episode(
        self,
        split: str,
        episode_idx: int,
        step: int,
        episode_return: float,
        episode_length: int,
        force_console: bool = False,
        force_plot: bool = False,
    ) -> None:
        self._ensure_directories()
        payload = {
            "split": split,
            "episode": int(episode_idx),
            "step": int(step),
            "return": float(episode_return),
            "length": int(episode_length),
        }
        with self.episodes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

        key = f"{split}/episode_return"
        self.episode_history.setdefault(key, []).append((int(step), float(episode_return)))

        should_print = force_console or episode_idx % self.episode_interval == 0
        if should_print:
            self.info(f"{split}_episode #{episode_idx} step={step} return={episode_return:.3f}, length={episode_length}")
        if self.enable_plots and force_plot:
            self.update_plots()

    def info(self, message: str) -> None:
        self._ensure_directories()
        print(message)
        with self.text_log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def save_json(self, name: str, payload: dict[str, Any]) -> None:
        self._ensure_directories()
        with (self.artifacts_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def update_plots(self) -> None:
        if not self.enable_plots:
            return
        self._ensure_directories()
        success = write_metric_plots(self.metric_history, self.episode_history, self.plots_dir)
        if not success and not self._matplotlib_warned:
            self._matplotlib_warned = True
            self.info("Plot generation skipped because matplotlib is not available in the current environment.")

    def finalize(self) -> None:
        self.update_plots()
