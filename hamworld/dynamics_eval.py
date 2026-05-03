from __future__ import annotations

import copy
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = REPO_ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _repo_relative_path(path: str | Path | None) -> str:
    if path is None:
        return ""
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        try:
            return resolved.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return resolved.as_posix()
    return resolved.as_posix()


TASK_ORDER = ["reacher_easy", "finger_spin", "cheetah_run", "cartpole_swingup"]
TASK_LABELS = {
    "reacher_easy": "Reacher",
    "finger_spin": "Finger",
    "cheetah_run": "Cheetah",
    "cartpole_swingup": "Cartpole",
}
MODE_ORDER = ["teacher_forced", "imagined"]
MODE_LABELS = {
    "teacher_forced": "Teacher Forced",
    "imagined": "Imagined",
}
MODE_COLORS = {
    "teacher_forced": "#2563eb",
    "imagined": "#dc2626",
}
TRACE_VECTOR_KEYS = [
    "q",
    "p",
    "c",
    "control",
    "action",
    "dH_dq",
    "dH_dp",
    "dq_net",
    "dp_net",
]
TRACE_SCALAR_KEYS = ["H", "reward"]
TRACE_EXTRA_KEYS = ["valid_mask", "episode_lengths", "episode_returns"]
TRACE_KEYS = set(TRACE_VECTOR_KEYS + TRACE_SCALAR_KEYS + TRACE_EXTRA_KEYS)
SUMMARY_FIELDS = [
    "task",
    "seed",
    "mode",
    "num_episodes",
    "num_valid_steps",
    "mean_episode_length",
    "mean_episode_return",
    "mean_reward",
    "mean_abs_energy_from_start",
    "mean_abs_delta_H",
    "mean_control_norm",
    "mean_action_norm",
    "energy_control_corr",
    "alpha",
    "q_dim",
    "p_dim",
    "c_dim",
    "trace_path",
]


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#334155",
        "grid.color": "#dbe2ea",
        "grid.linewidth": 0.7,
        "grid.linestyle": "--",
        "grid.alpha": 0.9,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


@dataclass(frozen=True)
class TaskRun:
    task: str
    seed: int
    run_dir: Path
    checkpoint_path: Path
    config_path: Path


@dataclass(frozen=True)
class EpisodeTrajectory:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    episode_return: float
    length: int


@dataclass
class TraceBundle:
    arrays: dict[str, np.ndarray]
    meta: dict[str, Any]
    path: Path | None = None


def task_title(task: str) -> str:
    return TASK_LABELS.get(task, task.replace("_", " ").title())


def trace_filename(task: str, seed: int, mode: str) -> str:
    return f"{task}_seed{seed}_{mode}.npz"


def _task_sort_key(task: str) -> tuple[int, str]:
    if task in TASK_ORDER:
        return TASK_ORDER.index(task), task
    return len(TASK_ORDER), task


def _mode_sort_key(mode: str) -> tuple[int, str]:
    if mode in MODE_ORDER:
        return MODE_ORDER.index(mode), mode
    return len(MODE_ORDER), mode


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    candidates = sorted(checkpoints_dir.glob("checkpoint_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: (_checkpoint_step(path), path.name))


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    if "_" not in stem:
        return -1
    try:
        return int(stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def resolve_task_runs(task_root: str | Path, seed: int, tasks: list[str] | None = None) -> list[TaskRun]:
    base_root = Path(task_root).expanduser().resolve()
    if not base_root.exists():
        raise FileNotFoundError(f"Task root does not exist: {base_root}")

    task_ids = list(tasks) if tasks else [path.name for path in base_root.iterdir() if path.is_dir()]
    resolved: list[TaskRun] = []
    missing_tasks = []

    for task in sorted(task_ids, key=_task_sort_key):
        task_dir = base_root / task
        if not task_dir.exists():
            missing_tasks.append(task)
            continue
        prefix = f"seed_{seed}_"
        candidates = sorted(path for path in task_dir.iterdir() if path.is_dir() and path.name.startswith(prefix))
        if not candidates:
            raise FileNotFoundError(f"No run directories found for task={task} seed={seed} under {task_dir}")

        run_dir = max(
            candidates,
            key=lambda path: (_latest_checkpoint(path) is not None, path.name),
        )
        checkpoint_path = _latest_checkpoint(run_dir)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint_*.pt found under {run_dir / 'checkpoints'}. "
                f"Collector needs checkpoints to run evaluation rollouts."
            )
        config_path = run_dir / "artifacts" / "config.yaml"
        resolved.append(
            TaskRun(
                task=task,
                seed=int(seed),
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
            )
        )

    if missing_tasks:
        raise FileNotFoundError(f"Missing task directories under {base_root}: {', '.join(missing_tasks)}")
    if not resolved:
        raise FileNotFoundError(f"No matching task runs found under {base_root} for seed={seed}")
    return resolved


def _import_collection_dependencies():
    import torch

    from hamworld.agent import HaMWorldAgent
    from hamworld.runtime import make_env, resolve_device
    from hamworld.world_model import infer_checkpoint_step

    return torch, HaMWorldAgent, infer_checkpoint_step, make_env, resolve_device


def _load_agent_and_env(task_run: TaskRun, device_name: str):
    torch, HaMWorldAgent, infer_checkpoint_step, make_env, resolve_device = _import_collection_dependencies()

    payload = torch.load(task_run.checkpoint_path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(payload["config"])
    config.setdefault("experiment", {})
    config["experiment"]["device"] = device_name
    env, env_spec = make_env(config["task"], int(task_run.seed) + 1000)
    agent = HaMWorldAgent(
        config=config,
        obs_dim=int(env_spec.observation_shape[0]),
        action_dim=int(env_spec.action_shape[0]),
        action_low=env_spec.action_low,
        action_high=env_spec.action_high,
    )
    agent_state = payload.get("agent_state")
    if agent_state is not None:
        agent.load_state_dict(agent_state)
    else:
        model_state = payload.get("model")
        if model_state is None:
            raise ValueError(f"Checkpoint does not contain `agent_state` or `model`: {task_run.checkpoint_path}")
        agent.world_model.load_state_dict(model_state)
        agent.reset()
    if agent.current_step is None:
        agent.set_training_step(infer_checkpoint_step(payload, task_run.checkpoint_path))
    agent.world_model.eval()
    return agent, env, resolve_device(device_name)


def collect_eval_episodes(
    task_run: TaskRun,
    num_episodes: int,
    max_steps: int,
    device_name: str = "auto",
) -> tuple[Any, list[EpisodeTrajectory]]:
    agent, env, _ = _load_agent_and_env(task_run, device_name)
    trajectories: list[EpisodeTrajectory] = []

    try:
        for episode_idx in range(int(num_episodes)):
            agent.reset()
            observation = np.asarray(env.reset(seed=int(task_run.seed) + 1000 + episode_idx), dtype=np.float32)
            observations = [observation]
            actions = []
            rewards = []
            episode_return = 0.0

            for _step in range(int(max_steps)):
                action = np.asarray(agent.act(observation, eval_mode=True), dtype=np.float32)
                next_observation, reward, done, _info = env.step(action)
                next_observation = np.asarray(next_observation, dtype=np.float32)
                actions.append(action)
                rewards.append(float(reward))
                observations.append(next_observation)
                episode_return += float(reward)
                observation = next_observation
                if done:
                    break

            if actions:
                trajectories.append(
                    EpisodeTrajectory(
                        observations=np.asarray(observations, dtype=np.float32),
                        actions=np.asarray(actions, dtype=np.float32),
                        rewards=np.asarray(rewards, dtype=np.float32),
                        episode_return=float(episode_return),
                        length=len(actions),
                    )
                )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    if not trajectories:
        raise RuntimeError(f"No evaluation trajectories were collected for {task_run.task} seed={task_run.seed}.")
    return agent.world_model, trajectories


def _tensor_to_numpy(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _trace_single_episode(world_model, trajectory: EpisodeTrajectory, mode: str, device) -> dict[str, np.ndarray]:
    torch, _, _, _ = _import_collection_dependencies()

    observations = torch.as_tensor(trajectory.observations, dtype=torch.float32, device=device)
    actions = torch.as_tensor(trajectory.actions, dtype=torch.float32, device=device)
    rewards = np.asarray(trajectory.rewards, dtype=np.float32)

    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("Expected flat state-based trajectories with shapes observations=[T+1,D], actions=[T,A].")
    if observations.shape[0] != actions.shape[0] + 1:
        raise ValueError("Trajectory observation/action lengths are inconsistent.")

    outputs = {key: [] for key in TRACE_VECTOR_KEYS + ["H"]}
    with torch.no_grad():
        encoded_obs = world_model.encode(observations[:-1])
        memory_state = world_model.reset_memory_state(batch_size=1, device=device)
        current_latent = encoded_obs[0:1]

        for step_idx in range(actions.shape[0]):
            if mode == "teacher_forced":
                current_latent = encoded_obs[step_idx : step_idx + 1]
            next_latent, memory_state, info = world_model.imagine_step(
                current_latent,
                actions[step_idx : step_idx + 1],
                memory_state,
                return_info=True,
            )
            outputs["q"].append(_tensor_to_numpy(info.q.squeeze(0)))
            outputs["p"].append(_tensor_to_numpy(info.p.squeeze(0)))
            outputs["c"].append(_tensor_to_numpy(info.c.squeeze(0)))
            outputs["control"].append(_tensor_to_numpy(info.control.squeeze(0)))
            outputs["action"].append(_tensor_to_numpy(actions[step_idx]))
            outputs["dH_dq"].append(_tensor_to_numpy(info.dH_dq.squeeze(0)))
            outputs["dH_dp"].append(_tensor_to_numpy(info.dH_dp.squeeze(0)))
            outputs["dq_net"].append(_tensor_to_numpy(info.dq_net.squeeze(0)))
            outputs["dp_net"].append(_tensor_to_numpy(info.dp_net.squeeze(0)))
            outputs["H"].append(np.asarray(float(_tensor_to_numpy(info.energy.reshape(-1))[0]), dtype=np.float32))
            if mode == "imagined":
                current_latent = next_latent

    episode_trace = {
        key: np.stack(values, axis=0).astype(np.float32, copy=False) if key != "H" else np.asarray(values, dtype=np.float32)
        for key, values in outputs.items()
    }
    episode_trace["reward"] = rewards.astype(np.float32, copy=False)
    return episode_trace


def _pad_episode_array(length: int, width: int | None = None) -> np.ndarray:
    if width is None:
        return np.full((length,), np.nan, dtype=np.float32)
    return np.full((length, width), np.nan, dtype=np.float32)


def build_trace_bundle(
    task: str,
    seed: int,
    mode: str,
    alpha: float,
    q_dim: int,
    p_dim: int,
    c_dim: int,
    episode_traces: list[dict[str, np.ndarray]],
    episode_lengths: np.ndarray,
    episode_returns: np.ndarray,
    max_steps: int,
) -> TraceBundle:
    if not episode_traces:
        raise ValueError("Cannot build a trace bundle from an empty episode list.")

    num_episodes = len(episode_traces)
    q_dim = int(q_dim)
    p_dim = int(p_dim)
    c_dim = int(c_dim)
    action_dim = int(episode_traces[0]["action"].shape[-1])
    arrays: dict[str, np.ndarray] = {
        "q": np.full((num_episodes, max_steps, q_dim), np.nan, dtype=np.float32),
        "p": np.full((num_episodes, max_steps, p_dim), np.nan, dtype=np.float32),
        "c": np.full((num_episodes, max_steps, c_dim), np.nan, dtype=np.float32),
        "control": np.full((num_episodes, max_steps, p_dim), np.nan, dtype=np.float32),
        "action": np.full((num_episodes, max_steps, action_dim), np.nan, dtype=np.float32),
        "dH_dq": np.full((num_episodes, max_steps, q_dim), np.nan, dtype=np.float32),
        "dH_dp": np.full((num_episodes, max_steps, p_dim), np.nan, dtype=np.float32),
        "dq_net": np.full((num_episodes, max_steps, q_dim), np.nan, dtype=np.float32),
        "dp_net": np.full((num_episodes, max_steps, p_dim), np.nan, dtype=np.float32),
        "H": np.full((num_episodes, max_steps), np.nan, dtype=np.float32),
        "reward": np.full((num_episodes, max_steps), np.nan, dtype=np.float32),
        "valid_mask": np.zeros((num_episodes, max_steps), dtype=bool),
        "episode_lengths": episode_lengths.astype(np.int32, copy=False),
        "episode_returns": episode_returns.astype(np.float32, copy=False),
    }

    for episode_idx, trace in enumerate(episode_traces):
        length = min(int(trace["H"].shape[0]), int(max_steps))
        arrays["valid_mask"][episode_idx, :length] = True
        for key in TRACE_VECTOR_KEYS + TRACE_SCALAR_KEYS:
            arrays[key][episode_idx, :length] = trace[key][:length]

    meta = {
        "task": task,
        "seed": int(seed),
        "mode": mode,
        "alpha": float(alpha),
        "q_dim": int(q_dim),
        "p_dim": int(p_dim),
        "c_dim": int(c_dim),
        "num_episodes": int(num_episodes),
        "max_steps": int(max_steps),
    }
    return TraceBundle(arrays=arrays, meta=meta)


def collect_trace_bundles(
    task_run: TaskRun,
    num_episodes: int,
    max_steps: int,
    device_name: str = "auto",
) -> dict[str, TraceBundle]:
    world_model, trajectories = collect_eval_episodes(task_run, num_episodes=num_episodes, max_steps=max_steps, device_name=device_name)
    _, _, _, resolve_device = _import_collection_dependencies()
    device = resolve_device(device_name)

    episode_lengths = np.asarray([trajectory.length for trajectory in trajectories], dtype=np.int32)
    episode_returns = np.asarray([trajectory.episode_return for trajectory in trajectories], dtype=np.float32)
    bundles: dict[str, TraceBundle] = {}

    for mode in MODE_ORDER:
        episode_traces = [
            _trace_single_episode(world_model, trajectory, mode=mode, device=device)
            for trajectory in trajectories
        ]
        bundles[mode] = build_trace_bundle(
            task=task_run.task,
            seed=task_run.seed,
            mode=mode,
            alpha=float(world_model.hamiltonian_alpha),
            q_dim=int(world_model.q_dim),
            p_dim=int(world_model.p_dim),
            c_dim=int(world_model.c_dim),
            episode_traces=episode_traces,
            episode_lengths=episode_lengths,
            episode_returns=episode_returns,
            max_steps=int(max_steps),
        )

    return bundles


def save_trace_bundle(bundle: TraceBundle, output_path: str | Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(bundle.arrays)
    for key, value in bundle.meta.items():
        payload[key] = np.asarray(value)
    np.savez_compressed(output_path, **payload)
    bundle.path = output_path
    return output_path


def _load_scalar(value: np.ndarray | Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def load_trace_bundle(path: str | Path) -> TraceBundle:
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files if key in TRACE_KEYS}
        meta = {key: _load_scalar(payload[key]) for key in payload.files if key not in TRACE_KEYS}
    return TraceBundle(arrays=arrays, meta=meta, path=path)


def load_trace_directory(traces_dir: str | Path) -> dict[str, dict[str, TraceBundle]]:
    traces_path = Path(traces_dir).expanduser().resolve()
    if not traces_path.exists():
        raise FileNotFoundError(f"Traces directory does not exist: {traces_path}")

    grouped: dict[str, dict[str, TraceBundle]] = {}
    for trace_path in sorted(traces_path.glob("*.npz")):
        bundle = load_trace_bundle(trace_path)
        task = str(bundle.meta.get("task"))
        mode = str(bundle.meta.get("mode"))
        if task in grouped and mode in grouped[task]:
            raise ValueError(
                f"Duplicate trace bundle detected for task={task} mode={mode}: "
                f"{grouped[task][mode].path} and {trace_path}"
            )
        grouped.setdefault(task, {})[mode] = bundle

    if not grouped:
        raise FileNotFoundError(f"No trace files found under {traces_path}")
    return dict(sorted(grouped.items(), key=lambda item: _task_sort_key(item[0])))


def _as_float_array(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def _masked_norm(array: np.ndarray) -> np.ndarray:
    return np.linalg.norm(_as_float_array(array), axis=-1)


def _nanmean_or_nan(array: np.ndarray) -> float:
    if array.size == 0:
        return float("nan")
    with np.errstate(invalid="ignore"):
        value = np.nanmean(array)
    return float(value) if np.isfinite(value) else float("nan")


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std <= 1e-12 or y_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summarize_trace_bundle(bundle: TraceBundle) -> dict[str, Any]:
    valid_mask = np.asarray(bundle.arrays["valid_mask"], dtype=bool)
    h = _as_float_array(bundle.arrays["H"])
    reward = _as_float_array(bundle.arrays["reward"])
    control_norm = _masked_norm(bundle.arrays["control"])
    action_norm = _masked_norm(bundle.arrays["action"])
    lengths = np.asarray(bundle.arrays["episode_lengths"], dtype=np.int32)
    episode_returns = _as_float_array(bundle.arrays["episode_returns"])

    energy_from_start = np.full_like(h, np.nan)
    for episode_idx, length in enumerate(lengths):
        if int(length) <= 0:
            continue
        energy_from_start[episode_idx, :length] = np.abs(h[episode_idx, :length] - h[episode_idx, 0])

    delta_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
    delta_h = np.abs(h[:, 1:] - h[:, :-1])
    delta_h_valid = delta_h[delta_mask]
    control_valid = control_norm[:, :-1][delta_mask]

    row = {
        "task": str(bundle.meta["task"]),
        "seed": int(bundle.meta["seed"]),
        "mode": str(bundle.meta["mode"]),
        "num_episodes": int(lengths.shape[0]),
        "num_valid_steps": int(valid_mask.sum()),
        "mean_episode_length": float(lengths.mean()) if lengths.size else float("nan"),
        "mean_episode_return": _nanmean_or_nan(episode_returns),
        "mean_reward": _nanmean_or_nan(reward[valid_mask]),
        "mean_abs_energy_from_start": _nanmean_or_nan(energy_from_start[valid_mask]),
        "mean_abs_delta_H": _nanmean_or_nan(delta_h_valid),
        "mean_control_norm": _nanmean_or_nan(control_norm[valid_mask]),
        "mean_action_norm": _nanmean_or_nan(action_norm[valid_mask]),
        "energy_control_corr": _safe_pearson(control_valid, delta_h_valid),
        "alpha": float(bundle.meta["alpha"]),
        "q_dim": int(bundle.meta["q_dim"]),
        "p_dim": int(bundle.meta["p_dim"]),
        "c_dim": int(bundle.meta["c_dim"]),
        "trace_path": _repo_relative_path(bundle.path),
    }
    return row


def write_summary_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#334155")


def _ordered_plot_tasks(traces_by_task: dict[str, dict[str, TraceBundle]]) -> list[str]:
    ordered = [task for task in TASK_ORDER if task in traces_by_task]
    extras = sorted((task for task in traces_by_task if task not in TASK_ORDER), key=_task_sort_key)
    return ordered + extras


def _save_figure_pair(fig: plt.Figure, output_stem: str | Path) -> tuple[Path, Path]:
    output_stem = Path(output_stem).expanduser().resolve()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def _compute_energy_drift_curves(bundle: TraceBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = _as_float_array(bundle.arrays["H"])
    valid_mask = np.asarray(bundle.arrays["valid_mask"], dtype=bool)
    drift = np.full_like(h, np.nan)
    for episode_idx, length in enumerate(np.asarray(bundle.arrays["episode_lengths"], dtype=np.int32)):
        if int(length) <= 0:
            continue
        drift[episode_idx, :length] = np.abs(h[episode_idx, :length] - h[episode_idx, 0])
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(drift, axis=0)
        std = np.nanstd(drift, axis=0)
    valid_steps = np.where(np.any(valid_mask, axis=0))[0]
    if valid_steps.size == 0:
        return np.asarray([], dtype=np.int32), np.asarray([]), np.asarray([])
    step_limit = int(valid_steps[-1]) + 1
    steps = np.arange(step_limit, dtype=np.int32)
    return steps, mean[:step_limit], std[:step_limit]


def _compute_scalar_curves(
    bundle: TraceBundle,
    key: str,
    vector_norm: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_mask = np.asarray(bundle.arrays["valid_mask"], dtype=bool)
    values = bundle.arrays[key]
    if vector_norm:
        values = _masked_norm(values)
    else:
        values = _as_float_array(values)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
    valid_steps = np.where(np.any(valid_mask, axis=0))[0]
    if valid_steps.size == 0:
        return np.asarray([], dtype=np.int32), np.asarray([]), np.asarray([])
    step_limit = int(valid_steps[-1]) + 1
    steps = np.arange(step_limit, dtype=np.int32)
    return steps, mean[:step_limit], std[:step_limit]


FREERUN_MODE_LABELS = {
    "no_action": "No action (free)",
    "random_action": "Random action",
}
FREERUN_MODE_COLORS = {
    "no_action": "#2563eb",
    "random_action": "#dc2626",
}
FREERUN_MODE_ORDER = ["no_action", "random_action"]


def _compute_freerun_curves(arrays: dict[str, np.ndarray], subtract_initial: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.asarray(arrays["H"], dtype=np.float32)
    valid_mask = np.asarray(arrays["valid_mask"], dtype=bool)
    h_masked = np.where(valid_mask, h, np.nan)
    if subtract_initial:
        h0 = h_masked[:, :1]
        h_masked = h_masked - h0
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(h_masked, axis=0)
        std = np.nanstd(h_masked, axis=0)
    valid_steps = np.where(np.any(valid_mask, axis=0))[0]
    if valid_steps.size == 0:
        return np.asarray([], dtype=np.int32), np.asarray([]), np.asarray([])
    step_limit = int(valid_steps[-1]) + 1
    steps = np.arange(step_limit, dtype=np.int32)
    return steps, mean[:step_limit], std[:step_limit]


def plot_h_freerun_per_task(
    traces_by_task: dict[str, dict[str, dict]],
    figures_dir: str | Path,
    subtract_initial: bool = True,
    filename_prefix: str = "h_freerun",
    ylabel: str | None = None,
) -> list[Path]:
    figures_dir = Path(figures_dir).expanduser().resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    if ylabel is None:
        ylabel = r"$H_t - H_0$" if subtract_initial else r"$H_t$"
    saved: list[Path] = []
    for task in traces_by_task:
        fig, axis = plt.subplots(1, 1, figsize=(4.35, 3.2), dpi=300)
        legend_handles = []
        legend_labels = []
        modes = [mode for mode in FREERUN_MODE_ORDER if mode in traces_by_task[task]]
        for mode in modes:
            steps, mean, std = _compute_freerun_curves(traces_by_task[task][mode], subtract_initial=subtract_initial)
            if steps.size == 0:
                continue
            color = FREERUN_MODE_COLORS.get(mode, "#475569")
            label = FREERUN_MODE_LABELS.get(mode, mode)
            axis.fill_between(steps, mean - std, mean + std, color=color, alpha=0.12, linewidth=0, zorder=1)
            line = axis.plot(
                steps,
                mean,
                color=color,
                linewidth=2.35,
                label=label,
                zorder=3,
                solid_capstyle="round",
                solid_joinstyle="round",
            )[0]
            legend_handles.append(line)
            legend_labels.append(label)
        _style_axes(axis)
        axis.set_title(task_title(task))
        axis.set_xlabel("Step")
        axis.set_ylabel(ylabel)
        if legend_handles:
            axis.legend(legend_handles, legend_labels, loc="best", frameon=False, fontsize=8.5, handlelength=2.6)
        png_path = (figures_dir / f"{filename_prefix}_{task}.png")
        pdf_path = (figures_dir / f"{filename_prefix}_{task}.pdf")
        fig.subplots_adjust(left=0.17, right=0.97, top=0.90, bottom=0.16)
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
        plt.close(fig)
        saved.extend([png_path, pdf_path])
    return saved


def plot_energy_evolution_per_task(
    traces_by_task: dict[str, dict[str, TraceBundle]],
    figures_dir: str | Path,
    task_filter: list[str] | None = None,
    quantity: str = "drift",
    filename_prefix: str | None = None,
) -> list[Path]:
    figures_dir = Path(figures_dir).expanduser().resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)
    tasks = _ordered_plot_tasks(traces_by_task)
    if task_filter is not None:
        keep = set(task_filter)
        tasks = [task for task in tasks if task in keep]
    saved: list[Path] = []
    for task in tasks:
        fig, axis = plt.subplots(1, 1, figsize=(4.0, 3.0), dpi=300)
        legend_handles = []
        legend_labels = []
        for mode in sorted(traces_by_task[task], key=_mode_sort_key):
            if mode == "imagined":
                continue
            bundle = traces_by_task[task][mode]
            if quantity == "drift":
                steps, mean, std = _compute_energy_drift_curves(bundle)
                clip_low = True
            elif quantity == "absolute":
                steps, mean, std = _compute_scalar_curves(bundle, "H", vector_norm=False)
                clip_low = False
            else:
                raise ValueError(f"Unknown quantity: {quantity}")
            if steps.size == 0:
                continue
            color = MODE_COLORS.get(mode, "#475569")
            label = MODE_LABELS.get(mode, mode)
            line = axis.plot(steps, mean, color=color, linewidth=1.8, label=label)[0]
            lower = np.clip(mean - std, 0.0, None) if clip_low else mean - std
            axis.fill_between(steps, lower, mean + std, color=color, alpha=0.16, linewidth=0)
            legend_handles.append(line)
            legend_labels.append(label)
        _style_axes(axis)
        axis.set_title(task_title(task))
        axis.set_xlabel("Step")
        axis.set_ylabel(r"$|H_t - H_0|$" if quantity == "drift" else r"$H_t$")
        if legend_handles:
            axis.legend(legend_handles, legend_labels, loc="best", frameon=False, fontsize=8)
        prefix = filename_prefix if filename_prefix is not None else ("energy_evolution" if quantity == "drift" else "energy_evolution_abs")
        png_path = (figures_dir / f"{prefix}_{task}.png")
        pdf_path = (figures_dir / f"{prefix}_{task}.pdf")
        fig.subplots_adjust(left=0.18, right=0.96, top=0.90, bottom=0.16)
        fig.savefig(png_path, dpi=300)
        fig.savefig(pdf_path)
        plt.close(fig)
        png_path, pdf_path = png_path, pdf_path
        saved.extend([png_path, pdf_path])
    return saved


def plot_energy_evolution_grid(traces_by_task: dict[str, dict[str, TraceBundle]], output_stem: str | Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), dpi=300)
    legend_handles = []
    legend_labels = []
    tasks = _ordered_plot_tasks(traces_by_task)

    for axis, task in zip(axes.flatten(), tasks):
        for mode in sorted(traces_by_task[task], key=_mode_sort_key):
            bundle = traces_by_task[task][mode]
            steps, mean, std = _compute_energy_drift_curves(bundle)
            if steps.size == 0:
                continue
            color = MODE_COLORS.get(mode, "#475569")
            label = MODE_LABELS.get(mode, mode)
            line = axis.plot(steps, mean, color=color, linewidth=1.8, label=label)[0]
            axis.fill_between(steps, np.clip(mean - std, 0.0, None), mean + std, color=color, alpha=0.16, linewidth=0)
            if label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(label)

        _style_axes(axis)
        axis.set_title(task_title(task))
        axis.set_xlabel("Step")
        axis.set_ylabel(r"$|H_t - H_0|$")

    for axis in axes.flatten()[len(tasks) :]:
        axis.axis("off")

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save_figure_pair(fig, output_stem)


def plot_hamiltonian_control_grid(
    traces_by_task: dict[str, dict[str, TraceBundle]],
    output_stem: str | Path,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 5.8), dpi=300)
    legend_handles = []
    legend_labels = []
    tasks = _ordered_plot_tasks(traces_by_task)

    for col, task in enumerate(tasks[:4]):
        energy_axis = axes[0, col]
        control_axis = axes[1, col]
        for mode in sorted(traces_by_task[task], key=_mode_sort_key):
            bundle = traces_by_task[task][mode]
            color = MODE_COLORS.get(mode, "#475569")
            label = MODE_LABELS.get(mode, mode)

            steps_h, mean_h, std_h = _compute_scalar_curves(bundle, "H", vector_norm=False)
            if steps_h.size:
                line = energy_axis.plot(steps_h, mean_h, color=color, linewidth=1.8, label=label)[0]
                energy_axis.fill_between(steps_h, mean_h - std_h, mean_h + std_h, color=color, alpha=0.16, linewidth=0)
                if label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)

            steps_c, mean_c, std_c = _compute_scalar_curves(bundle, "control", vector_norm=True)
            if steps_c.size:
                control_axis.plot(steps_c, mean_c, color=color, linewidth=1.8, label=label)
                control_axis.fill_between(steps_c, np.clip(mean_c - std_c, 0.0, None), mean_c + std_c, color=color, alpha=0.16, linewidth=0)

        _style_axes(energy_axis)
        _style_axes(control_axis)
        energy_axis.set_title(task_title(task))
        energy_axis.set_xlabel("Step")
        energy_axis.set_ylabel(r"$H_t$")
        control_axis.set_xlabel("Step")
        control_axis.set_ylabel(r"$||control_t||_2$")

    for col in range(len(tasks), 4):
        axes[0, col].axis("off")
        axes[1, col].axis("off")

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save_figure_pair(fig, output_stem)


def _flatten_energy_vs_control(bundle: TraceBundle) -> tuple[np.ndarray, np.ndarray]:
    h = _as_float_array(bundle.arrays["H"])
    valid_mask = np.asarray(bundle.arrays["valid_mask"], dtype=bool)
    control_norm = _masked_norm(bundle.arrays["control"])
    delta_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
    x = control_norm[:, :-1][delta_mask]
    y = np.abs(h[:, 1:] - h[:, :-1])[delta_mask]
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def _sample_points(x: np.ndarray, y: np.ndarray, max_points: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    rng = np.random.default_rng(42)
    indices = rng.choice(x.size, size=max_points, replace=False)
    return x[indices], y[indices]


def _binned_curve(x: np.ndarray, y: np.ndarray, bins: int = 16) -> tuple[np.ndarray, np.ndarray]:
    if x.size < 8:
        return np.asarray([]), np.asarray([])
    edges = np.linspace(float(x.min()), float(x.max()), bins + 1)
    centers = []
    values = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        if right == edges[-1]:
            mask = (x >= left) & (x <= right)
        else:
            mask = (x >= left) & (x < right)
        if mask.sum() < 4:
            continue
        centers.append(0.5 * (left + right))
        values.append(float(np.median(y[mask])))
    return np.asarray(centers, dtype=np.float32), np.asarray(values, dtype=np.float32)


def plot_energy_vs_control_grid(
    traces_by_task: dict[str, dict[str, TraceBundle]],
    output_stem: str | Path,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), dpi=300)
    legend_handles = []
    legend_labels = []
    tasks = _ordered_plot_tasks(traces_by_task)

    for axis, task in zip(axes.flatten(), tasks):
        for mode in sorted(traces_by_task[task], key=_mode_sort_key):
            bundle = traces_by_task[task][mode]
            x, y = _flatten_energy_vs_control(bundle)
            if x.size == 0:
                continue
            x_plot, y_plot = _sample_points(x, y)
            color = MODE_COLORS.get(mode, "#475569")
            label = MODE_LABELS.get(mode, mode)
            axis.scatter(x_plot, y_plot, s=8, alpha=0.18, color=color, rasterized=True)
            centers, values = _binned_curve(x, y)
            if centers.size:
                line = axis.plot(centers, values, color=color, linewidth=2.0, label=label)[0]
                if label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(label)

        _style_axes(axis)
        axis.set_title(task_title(task))
        axis.set_xlabel(r"$||control_t||_2$")
        axis.set_ylabel(r"$|H_{t+1} - H_t|$")

    for axis in axes.flatten()[len(tasks) :]:
        axis.axis("off")

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save_figure_pair(fig, output_stem)


def _pad_feature_dims(lhs: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target_dim = max(lhs.shape[-1], rhs.shape[-1])
    if lhs.shape[-1] < target_dim:
        lhs = np.pad(lhs, ((0, 0), (0, target_dim - lhs.shape[-1])), mode="constant")
    if rhs.shape[-1] < target_dim:
        rhs = np.pad(rhs, ((0, 0), (0, target_dim - rhs.shape[-1])), mode="constant")
    return lhs, rhs


def _flatten_valid_vectors(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    flat = values[valid_mask]
    return np.asarray(flat, dtype=np.float32)


def plot_qp_umap_grid(
    traces_by_task: dict[str, dict[str, TraceBundle]],
    output_stem: str | Path,
    max_points: int = 4000,
) -> tuple[Path, Path]:
    embed_name = "UMAP"
    try:
        from umap import UMAP
        projector = lambda array: UMAP(n_neighbors=30, min_dist=0.2, n_components=2, random_state=42).fit_transform(array)
    except ImportError:
        embed_name = "PCA"

        def projector(array: np.ndarray) -> np.ndarray:
            centered = array - np.mean(array, axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            basis = vt[:2].T
            return centered @ basis

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.5), dpi=300)
    legend_handles = None
    legend_labels = None
    tasks = _ordered_plot_tasks(traces_by_task)

    for axis, task in zip(axes.flatten(), tasks):
        bundle = traces_by_task[task].get("teacher_forced")
        if bundle is None:
            axis.set_title(f"{task_title(task)} (missing)")
            axis.axis("off")
            continue

        valid_mask = np.asarray(bundle.arrays["valid_mask"], dtype=bool)
        q = _flatten_valid_vectors(bundle.arrays["q"], valid_mask)
        p = _flatten_valid_vectors(bundle.arrays["p"], valid_mask)
        if q.size == 0 or p.size == 0:
            axis.set_title(f"{task_title(task)} (empty)")
            axis.axis("off")
            continue

        q, p = _pad_feature_dims(q, p)
        sample_count = min(int(max_points), int(q.shape[0]), int(p.shape[0]))
        rng = np.random.default_rng(42)
        q_indices = rng.choice(q.shape[0], size=sample_count, replace=False)
        p_indices = rng.choice(p.shape[0], size=sample_count, replace=False)
        q_sample = q[q_indices]
        p_sample = p[p_indices]
        combined = np.concatenate([q_sample, p_sample], axis=0)

        embedding = projector(combined)
        q_emb = embedding[:sample_count]
        p_emb = embedding[sample_count:]

        q_artist = axis.scatter(q_emb[:, 0], q_emb[:, 1], s=5, alpha=0.35, color="#ee7733", rasterized=True, label="q")
        p_artist = axis.scatter(p_emb[:, 0], p_emb[:, 1], s=5, alpha=0.35, color="#0077bb", rasterized=True, label="p")
        legend_handles = [q_artist, p_artist]
        legend_labels = ["q", "p"]

        axis.set_title(task_title(task))
        axis.set_xlabel(f"{embed_name}-1")
        axis.set_ylabel(f"{embed_name}-2")
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for axis in axes.flatten()[len(tasks) :]:
        axis.axis("off")

    if legend_handles is not None and legend_labels is not None:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save_figure_pair(fig, output_stem)
