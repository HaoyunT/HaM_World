from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import re
import statistics
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
import torch

from dreamerv3.modules import RSSMState, flatten_time as dreamer_flatten_time, slice_state as dreamer_slice_state
from dreamerv3.world_model import DreamerV3WorldModel
from jepa.world_model import JEPAWorldModel
from hamworld import compare as cmp
from hamworld.export_paper_assets import MAIN_TABLE_TASK_LABELS, SeedRun
from hamworld.world_model import CanonicalDynamicsWorldModel, infer_checkpoint_step
from tdmpc2.world_model import TDMPC2WorldModel


SUPPORTED_ALGOS = ("hamworld", "jepa", "tdmpc2", "dreamerv3")
DEFAULT_HORIZONS = tuple(range(1, 16))
LONG_HORIZON_STYLES = {
    "hamworld": {"label": "HaM-World", "color": "#2563eb"},
    "jepa": {"label": "JEPA", "color": "#f59e0b"},
    "tdmpc2": {"label": "TD-MPC2", "color": "#16a34a"},
    "dreamerv3": {"label": "DreamerV3", "color": "#dc2626"},
}
PLOT_ALGORITHM_ORDER = ("hamworld", "tdmpc2", "dreamerv3", "jepa")


def _summary_horizon(results: dict[str, dict[str, dict[str, Any]]]) -> int:
    horizons = sorted(
        {
            horizon
            for task_results in results.values()
            for item in task_results.values()
            for horizon in item["mse_mean"].keys()
        }
    )
    return horizons[-1] if horizons else DEFAULT_HORIZONS[-1]


@dataclass
class SeedEvaluation:
    algorithm: str
    task: str
    seed: int
    run_dir: Path
    checkpoint_path: Path
    mse: dict[int, float]
    stability: dict[int, float]
    episode_count: int
    valid_episode_count: dict[int, int]
    stability_rollouts: int
    stability_noise_scale: float
    stability_noise_std_mean: float


class WorldModelAdapter:
    algorithm: str

    def __init__(self, world_model: torch.nn.Module, device: torch.device):
        self.world_model = world_model
        self.device = device
        self.world_model.eval()

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def prepare_sequences(self, observations: torch.Tensor, actions: torch.Tensor):
        encoded_obs = self.encode(observations.reshape(-1, observations.shape[-1])).view(observations.shape[0], observations.shape[1], -1)
        context = self.prepare_context(encoded_obs[:, :-1], actions)
        return encoded_obs, context

    def prepare_context(self, encoded_obs: torch.Tensor, actions: torch.Tensor):
        raise NotImplementedError

    def select_context(self, context, starts: torch.Tensor):
        raise NotImplementedError

    def repeat_context(self, context, repeats: int):
        raise NotImplementedError

    def imagine_step(self, latent: torch.Tensor, action: torch.Tensor, context):
        raise NotImplementedError

    def apply_noise(self, latent: torch.Tensor, context, noise_std: float | torch.Tensor):
        if isinstance(noise_std, torch.Tensor):
            if torch.all(noise_std <= 0):
                return latent, context
            return latent + torch.randn_like(latent) * noise_std, context
        if noise_std <= 0.0:
            return latent, context
        return latent + torch.randn_like(latent) * noise_std, context


class HaMWorldAdapter(WorldModelAdapter):
    algorithm = "hamworld"

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.world_model.encode(observations)

    def prepare_context(self, encoded_obs: torch.Tensor, actions: torch.Tensor) -> list[torch.Tensor]:
        if encoded_obs.ndim == 2:
            encoded_obs = encoded_obs.unsqueeze(0)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        batch_size = encoded_obs.shape[0]
        state = self.world_model.reset_memory_state(batch_size=batch_size, device=self.device)
        snapshots: list[list[torch.Tensor]] = []
        for index in range(actions.shape[1]):
            snapshots.append([layer_state.clone() for layer_state in state])
            state = self.world_model.advance_memory(encoded_obs[:, index], actions[:, index], state)
        if not snapshots:
            return []
        layers = []
        for layer in range(len(snapshots[0])):
            layers.append(torch.stack([snapshot[layer] for snapshot in snapshots], dim=1))
        return layers

    def select_context(self, context: list[torch.Tensor], starts: torch.Tensor) -> list[torch.Tensor]:
        return [layer_state.index_select(0, starts) for layer_state in context]

    def repeat_context(self, context: list[torch.Tensor], repeats: int) -> list[torch.Tensor]:
        return [layer_state.repeat_interleave(repeats, dim=0) for layer_state in context]

    def imagine_step(self, latent: torch.Tensor, action: torch.Tensor, context: list[torch.Tensor]):
        next_latent, next_context = self.world_model.imagine_step(latent, action, context)
        return next_latent, next_context

    def apply_noise(self, latent: torch.Tensor, context, noise_std: float | torch.Tensor):
        noisy, context = super().apply_noise(latent, context, noise_std)
        return self.world_model._clamp_latent_norm(noisy), context


class JEPAAdapter(WorldModelAdapter):
    algorithm = "jepa"

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.world_model.encode(observations)

    def prepare_context(self, encoded_obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if encoded_obs.ndim == 2:
            encoded_obs = encoded_obs.unsqueeze(0)
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        hidden = self.world_model.reset_prior_state(batch_size=encoded_obs.shape[0], device=self.device)
        snapshots = []
        for index in range(actions.shape[1]):
            snapshots.append(hidden.clone())
            _, hidden = self.world_model.imagine_step(encoded_obs[:, index], actions[:, index], hidden)
        return torch.stack(snapshots, dim=1) if snapshots else torch.empty(0, 0, hidden.shape[-1], device=self.device)

    def select_context(self, context: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
        return context.index_select(0, starts)

    def repeat_context(self, context: torch.Tensor, repeats: int) -> torch.Tensor:
        return context.repeat_interleave(repeats, dim=0)

    def imagine_step(self, latent: torch.Tensor, action: torch.Tensor, context: torch.Tensor):
        next_latent, next_context = self.world_model.imagine_step(latent, action, context)
        return next_latent, next_context


class TDMPC2Adapter(WorldModelAdapter):
    algorithm = "tdmpc2"

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.world_model.encode(observations)

    def prepare_context(self, encoded_obs: torch.Tensor, actions: torch.Tensor):
        del encoded_obs, actions
        return None

    def select_context(self, context, starts: torch.Tensor):
        del context, starts
        return None

    def repeat_context(self, context, repeats: int):
        del context, repeats
        return None

    def imagine_step(self, latent: torch.Tensor, action: torch.Tensor, context):
        del context
        return self.world_model.imagine_step(latent, action), None


class DreamerV3Adapter(WorldModelAdapter):
    algorithm = "dreamerv3"

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        return self.world_model.encode(observations)

    def prepare_sequences(self, observations: torch.Tensor, actions: torch.Tensor):
        posts, _ = self.world_model.observe_sequence(observations, actions, deterministic=True)
        features = self.world_model.feature(posts)
        contexts = dreamer_slice_state(posts, stop=-1)
        return features, contexts

    def prepare_context(self, encoded_obs: torch.Tensor, actions: torch.Tensor):
        del encoded_obs, actions
        raise NotImplementedError("DreamerV3 uses prepare_sequences directly.")

    def select_context(self, context: RSSMState, starts: torch.Tensor) -> RSSMState:
        return RSSMState(
            deter=context.deter.index_select(0, starts),
            stoch=context.stoch.index_select(0, starts),
            logits=context.logits.index_select(0, starts),
        )

    def repeat_context(self, context: RSSMState, repeats: int) -> RSSMState:
        return RSSMState(
            deter=context.deter.repeat_interleave(repeats, dim=0),
            stoch=context.stoch.repeat_interleave(repeats, dim=0),
            logits=context.logits.repeat_interleave(repeats, dim=0),
        )

    def imagine_step(self, latent: torch.Tensor, action: torch.Tensor, context: RSSMState):
        del latent
        next_state = self.world_model.imagine_step(context, action, deterministic=True)
        next_feature = self.world_model.feature(next_state)
        return next_feature, next_state

    def apply_noise(self, latent: torch.Tensor, context: RSSMState, noise_std: float | torch.Tensor):
        if isinstance(noise_std, torch.Tensor):
            if torch.all(noise_std <= 0):
                return latent, context
            deter_noise = torch.randn_like(context.deter) * noise_std
        else:
            if noise_std <= 0.0:
                return latent, context
            deter_noise = torch.randn_like(context.deter) * noise_std
        noisy_state = RSSMState(
            deter=context.deter + deter_noise,
            stoch=context.stoch,
            logits=context.logits,
        )
        return self.world_model.feature(noisy_state), noisy_state


def _world_model_class(algorithm: str):
    if algorithm == "hamworld":
        return CanonicalDynamicsWorldModel, HaMWorldAdapter
    if algorithm == "jepa":
        return JEPAWorldModel, JEPAAdapter
    if algorithm == "tdmpc2":
        return TDMPC2WorldModel, TDMPC2Adapter
    if algorithm == "dreamerv3":
        return DreamerV3WorldModel, DreamerV3Adapter
    raise ValueError(f"Unsupported long-horizon algorithm: {algorithm}")


def _resolve_device(raw_device: str | None = None) -> torch.device:
    if raw_device is None or raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def _resolve_device_list(raw_device: str | None = None) -> list[str]:
    if raw_device is None or raw_device == "auto":
        if torch.cuda.is_available():
            return ["cuda:0"]
        return ["cpu"]
    devices = [item.strip() for item in str(raw_device).split(",") if item.strip()]
    return devices or ["cpu"]


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None
    candidates = sorted(checkpoints_dir.glob("checkpoint_*.pt"), key=lambda path: (_checkpoint_step(path), path.name))
    return candidates[-1] if candidates else None


def _to_serializable_metrics(metrics: dict[int, float]) -> dict[str, float]:
    return {str(int(key)): float(value) for key, value in sorted(metrics.items())}


def _from_serializable_metrics(metrics: dict[str, float]) -> dict[int, float]:
    return {int(key): float(value) for key, value in metrics.items()}


def _cache_path(run_dir: Path) -> Path:
    return run_dir / "analysis" / "long_horizon_eval.json"


def _load_cached_result(
    run_dir: Path,
    checkpoint_path: Path,
    horizons: tuple[int, ...],
    batch_size: int,
    stability_rollouts: int,
    stability_noise_scale: float,
) -> SeedEvaluation | None:
    cache_path = _cache_path(run_dir)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    expected_meta = {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
        "horizons": list(horizons),
        "batch_size": int(batch_size),
        "stability_rollouts": int(stability_rollouts),
        "stability_noise_scale": float(stability_noise_scale),
    }
    meta = payload.get("meta", {})
    if meta != expected_meta:
        return None

    return SeedEvaluation(
        algorithm=str(payload["algorithm"]),
        task=str(payload["task"]),
        seed=int(payload["seed"]),
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        mse=_from_serializable_metrics(payload["mse"]),
        stability=_from_serializable_metrics(payload["stability"]),
        episode_count=int(payload["episode_count"]),
        valid_episode_count={int(key): int(value) for key, value in payload["valid_episode_count"].items()},
        stability_rollouts=int(payload["stability_rollouts"]),
        stability_noise_scale=float(payload["stability_noise_scale"]),
        stability_noise_std_mean=float(payload.get("stability_noise_std_mean", 0.0)),
    )


def _save_cached_result(result: SeedEvaluation, horizons: tuple[int, ...], batch_size: int) -> None:
    checkpoint_path = result.checkpoint_path.resolve()
    cache_path = _cache_path(result.run_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns,
            "horizons": list(horizons),
            "batch_size": int(batch_size),
            "stability_rollouts": int(result.stability_rollouts),
            "stability_noise_scale": float(result.stability_noise_scale),
        },
        "algorithm": result.algorithm,
        "task": result.task,
        "seed": result.seed,
        "mse": _to_serializable_metrics(result.mse),
        "stability": _to_serializable_metrics(result.stability),
        "episode_count": int(result.episode_count),
        "valid_episode_count": {str(key): int(value) for key, value in sorted(result.valid_episode_count.items())},
        "stability_rollouts": int(result.stability_rollouts),
        "stability_noise_scale": float(result.stability_noise_scale),
        "stability_noise_std_mean": float(result.stability_noise_std_mean),
    }
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_model_and_dataset(checkpoint_path: Path, device: torch.device) -> tuple[WorldModelAdapter, list[dict[str, np.ndarray]], dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    algorithm = config["experiment"]["algorithm"]
    if algorithm not in SUPPORTED_ALGOS:
        raise ValueError(f"Long-horizon evaluation only supports {SUPPORTED_ALGOS}, got {algorithm}.")

    buffer_state = payload.get("buffer_state", {})
    episodes = list(buffer_state.get("episodes", []))
    if not episodes:
        raise ValueError(f"No replay episodes found in checkpoint: {checkpoint_path}")

    obs_dim = int(episodes[0]["observations"].shape[-1])
    action_dim = int(episodes[0]["actions"].shape[-1])
    model_cls, adapter_cls = _world_model_class(algorithm)
    world_model = model_cls(config, obs_dim, action_dim).to(device)
    state_dict = payload.get("model")
    if state_dict is None:
        agent_state = payload.get("agent_state", {})
        state_dict = agent_state.get("model")
    if state_dict is None:
        raise ValueError(f"Checkpoint does not contain a world-model state_dict: {checkpoint_path}")
    world_model.load_state_dict(state_dict)
    world_model.set_schedule_step(infer_checkpoint_step(payload, checkpoint_path))
    world_model.eval()
    return adapter_cls(world_model, device), episodes, payload


def _rollout_prediction(
    adapter: WorldModelAdapter,
    encoded_obs: torch.Tensor,
    actions: torch.Tensor,
    context,
    starts: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    current = encoded_obs.index_select(0, starts)
    rollout_context = adapter.select_context(context, starts)
    for delta in range(horizon):
        step_actions = actions.index_select(0, starts + delta)
        current, rollout_context = adapter.imagine_step(current, step_actions, rollout_context)
    return current


def _rollout_stability(
    adapter: WorldModelAdapter,
    encoded_obs: torch.Tensor,
    actions: torch.Tensor,
    context,
    starts: torch.Tensor,
    horizon: int,
    repeats: int,
    noise_std: float,
) -> torch.Tensor:
    current = encoded_obs.index_select(0, starts)
    rollout_context = adapter.select_context(context, starts)
    current = current.repeat_interleave(repeats, dim=0)
    rollout_context = adapter.repeat_context(rollout_context, repeats)
    for delta in range(horizon):
        step_actions = actions.index_select(0, starts + delta).repeat_interleave(repeats, dim=0)
        current, rollout_context = adapter.imagine_step(current, step_actions, rollout_context)
        current, rollout_context = adapter.apply_noise(current, rollout_context, noise_std)
    final = current.view(starts.shape[0], repeats, -1)
    return final.var(dim=1, unbiased=False).sum(dim=-1)


def _slice_episode_context(context, start: int, stop: int):
    if context is None:
        return None
    if isinstance(context, RSSMState):
        return RSSMState(
            deter=context.deter[start:stop],
            stoch=context.stoch[start:stop],
            logits=context.logits[start:stop],
        )
    if isinstance(context, list):
        return [layer[start:stop] for layer in context]
    return context[start:stop]


def _flatten_start_context(context, start_count: int):
    if context is None:
        return None
    if isinstance(context, RSSMState):
        return dreamer_flatten_time(dreamer_slice_state(context, stop=start_count))
    if isinstance(context, list):
        return [layer[:, :start_count].reshape(-1, layer.shape[-1]) for layer in context]
    return context[:, :start_count].reshape(-1, context.shape[-1])


def _repeat_context(context, repeats: int):
    if context is None:
        return None
    if isinstance(context, RSSMState):
        return RSSMState(
            deter=context.deter.repeat_interleave(repeats, dim=0),
            stoch=context.stoch.repeat_interleave(repeats, dim=0),
            logits=context.logits.repeat_interleave(repeats, dim=0),
        )
    if isinstance(context, list):
        return [layer.repeat_interleave(repeats, dim=0) for layer in context]
    return context.repeat_interleave(repeats, dim=0)


def _episode_chunk_size(valid_start_count: int, batch_size: int, repeats: int = 1) -> int:
    denom = max(1, valid_start_count * repeats)
    return max(1, batch_size // denom)


def _evaluate_chunk_mse(
    adapter: WorldModelAdapter,
    encoded_obs: torch.Tensor,
    actions: torch.Tensor,
    context,
    horizon: int,
) -> torch.Tensor:
    batch_episodes = encoded_obs.shape[0]
    start_count = int(actions.shape[1]) - int(horizon) + 1
    current = encoded_obs[:, :start_count].reshape(batch_episodes * start_count, -1)
    rollout_context = _flatten_start_context(context, start_count)
    for delta in range(horizon):
        step_actions = actions[:, delta : delta + start_count].reshape(batch_episodes * start_count, -1)
        current, rollout_context = adapter.imagine_step(current, step_actions, rollout_context)
    pred = current.view(batch_episodes, start_count, -1)
    target = encoded_obs[:, horizon : horizon + start_count]
    return (pred - target).pow(2).sum(dim=-1).mean(dim=1)


def _evaluate_chunk_stability(
    adapter: WorldModelAdapter,
    encoded_obs: torch.Tensor,
    actions: torch.Tensor,
    context,
    horizon: int,
    repeats: int,
    noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_episodes = encoded_obs.shape[0]
    start_count = int(actions.shape[1]) - int(horizon) + 1
    current = encoded_obs[:, :start_count].unsqueeze(2).repeat(1, 1, repeats, 1).reshape(batch_episodes * start_count * repeats, -1)
    rollout_context = _repeat_context(_flatten_start_context(context, start_count), repeats)

    episode_noise = encoded_obs.std(dim=(1, 2), unbiased=False) * noise_scale
    if noise_scale > 0.0:
        episode_noise = torch.clamp(episode_noise, min=1e-6)
    expanded_noise = episode_noise[:, None, None].expand(batch_episodes, start_count, repeats).reshape(-1, 1)

    for delta in range(horizon):
        step_actions = (
            actions[:, delta : delta + start_count]
            .unsqueeze(2)
            .repeat(1, 1, repeats, 1)
            .reshape(batch_episodes * start_count * repeats, -1)
        )
        current, rollout_context = adapter.imagine_step(current, step_actions, rollout_context)
        current, rollout_context = adapter.apply_noise(current, rollout_context, expanded_noise)

    final = current.view(batch_episodes, start_count, repeats, -1)
    episode_scores = final.var(dim=2, unbiased=False).sum(dim=-1).mean(dim=1)
    return episode_scores, episode_noise


def _evaluate_episode_mse(
    adapter: WorldModelAdapter,
    encoded_obs: torch.Tensor,
    actions: torch.Tensor,
    context,
    horizon: int,
    batch_size: int,
) -> float | None:
    valid_count = int(actions.shape[0]) - int(horizon) + 1
    if valid_count <= 0:
        return None
    starts = torch.arange(valid_count, device=actions.device, dtype=torch.long)
    errors = []
    for chunk in starts.split(batch_size):
        pred = _rollout_prediction(adapter, encoded_obs, actions, context, chunk, horizon)
        target = encoded_obs.index_select(0, chunk + horizon)
        errors.append((pred - target).pow(2).sum(dim=-1))
    if not errors:
        return None
    return float(torch.cat(errors, dim=0).mean().item())


def _evaluate_episode_stability(
    adapter: WorldModelAdapter,
    encoded_obs: torch.Tensor,
    actions: torch.Tensor,
    context,
    horizon: int,
    batch_size: int,
    repeats: int,
    noise_scale: float,
) -> tuple[float | None, float]:
    valid_count = int(actions.shape[0]) - int(horizon) + 1
    if valid_count <= 0:
        return None, 0.0
    starts = torch.arange(valid_count, device=actions.device, dtype=torch.long)
    latent_std = float(encoded_obs.std(unbiased=False).item())
    noise_std = max(1e-6, latent_std * noise_scale) if noise_scale > 0.0 else 0.0
    stability_scores = []
    stability_batch = max(1, batch_size // max(1, repeats))
    for chunk in starts.split(stability_batch):
        stability_scores.append(_rollout_stability(adapter, encoded_obs, actions, context, chunk, horizon, repeats, noise_std))
    if not stability_scores:
        return None, noise_std
    return float(torch.cat(stability_scores, dim=0).mean().item()), noise_std


def evaluate_model(
    model: WorldModelAdapter,
    dataset: list[dict[str, np.ndarray]],
    horizons: tuple[int, ...],
    batch_size: int = 2048,
    stability_rollouts: int = 5,
    stability_noise_scale: float = 0.01,
) -> tuple[dict[int, float], dict[int, float], dict[int, int], float]:
    horizons = tuple(sorted(int(horizon) for horizon in horizons))
    mse_by_horizon: dict[int, list[float]] = {horizon: [] for horizon in horizons}
    stability_by_horizon: dict[int, list[float]] = {horizon: [] for horizon in horizons}
    valid_episode_count = {horizon: 0 for horizon in horizons}
    noise_stds: list[float] = []
    episodes_by_length: dict[int, list[dict[str, np.ndarray]]] = {}
    for episode in dataset:
        actions = episode.get("actions")
        observations = episode.get("observations")
        if actions is None or observations is None:
            continue
        if actions.ndim != 2 or observations.ndim != 2:
            continue
        if actions.shape[0] <= 0 or observations.shape[0] != actions.shape[0] + 1:
            continue
        episodes_by_length.setdefault(int(actions.shape[0]), []).append(episode)

    with torch.no_grad():
        for episode_length, episodes in sorted(episodes_by_length.items()):
            obs_dim = int(episodes[0]["observations"].shape[-1])
            observations_all = torch.as_tensor(
                np.stack([episode["observations"] for episode in episodes]),
                dtype=torch.float32,
                device=model.device,
            )
            actions_all = torch.as_tensor(
                np.stack([episode["actions"] for episode in episodes]),
                dtype=torch.float32,
                device=model.device,
            )

            encoded_all, context_all = model.prepare_sequences(observations_all, actions_all)

            for horizon in horizons:
                valid_start_count = episode_length - horizon + 1
                if valid_start_count <= 0:
                    continue
                chunk_size = _episode_chunk_size(valid_start_count, batch_size, repeats=1)
                stability_chunk_size = _episode_chunk_size(valid_start_count, batch_size, repeats=stability_rollouts)
                for start in range(0, observations_all.shape[0], chunk_size):
                    stop = min(observations_all.shape[0], start + chunk_size)
                    encoded_chunk = encoded_all[start:stop]
                    actions_chunk = actions_all[start:stop]
                    context_chunk = _slice_episode_context(context_all, start, stop)
                    mse_scores = _evaluate_chunk_mse(model, encoded_chunk, actions_chunk, context_chunk, horizon)
                    mse_by_horizon[horizon].extend(float(value) for value in mse_scores.cpu().tolist())
                    valid_episode_count[horizon] += int(mse_scores.shape[0])
                for start in range(0, observations_all.shape[0], stability_chunk_size):
                    stop = min(observations_all.shape[0], start + stability_chunk_size)
                    encoded_chunk = encoded_all[start:stop]
                    actions_chunk = actions_all[start:stop]
                    context_chunk = _slice_episode_context(context_all, start, stop)
                    stability_scores, episode_noise = _evaluate_chunk_stability(
                        model,
                        encoded_chunk,
                        actions_chunk,
                        context_chunk,
                        horizon,
                        stability_rollouts,
                        stability_noise_scale,
                    )
                    stability_by_horizon[horizon].extend(float(value) for value in stability_scores.cpu().tolist())
                    noise_stds.extend(float(value) for value in episode_noise.cpu().tolist())

    mse = {
        horizon: (statistics.fmean(values) if values else float("nan"))
        for horizon, values in mse_by_horizon.items()
    }
    stability = {
        horizon: (statistics.fmean(values) if values else float("nan"))
        for horizon, values in stability_by_horizon.items()
    }
    noise_std_mean = statistics.fmean(noise_stds) if noise_stds else 0.0
    return mse, stability, valid_episode_count, noise_std_mean


def evaluate_seed_run(
    run: SeedRun,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    device: torch.device | None = None,
    batch_size: int = 2048,
    stability_rollouts: int = 5,
    stability_noise_scale: float = 0.01,
) -> SeedEvaluation:
    checkpoint_path = _latest_checkpoint(run.run_dir)
    if checkpoint_path is None:
        raise ValueError(f"No checkpoint found for run: {run.run_dir}")

    cached = _load_cached_result(
        run.run_dir,
        checkpoint_path,
        horizons,
        batch_size,
        stability_rollouts,
        stability_noise_scale,
    )
    if cached is not None:
        return cached

    resolved_device = device or _resolve_device("auto")
    adapter, dataset, _ = _load_model_and_dataset(checkpoint_path, resolved_device)
    mse, stability, valid_episode_count, noise_std_mean = evaluate_model(
        adapter,
        dataset,
        horizons,
        batch_size=batch_size,
        stability_rollouts=stability_rollouts,
        stability_noise_scale=stability_noise_scale,
    )
    result = SeedEvaluation(
        algorithm=run.algorithm,
        task=run.task,
        seed=run.seed,
        run_dir=run.run_dir,
        checkpoint_path=checkpoint_path,
        mse=mse,
        stability=stability,
        episode_count=len(dataset),
        valid_episode_count=valid_episode_count,
        stability_rollouts=stability_rollouts,
        stability_noise_scale=stability_noise_scale,
        stability_noise_std_mean=noise_std_mean,
    )
    _save_cached_result(result, horizons, batch_size)
    return result


def _evaluate_seed_run_worker(spec: dict[str, Any]) -> SeedEvaluation:
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    if torch.cuda.is_available():
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    run_dir = Path(spec["run_dir"]).expanduser().resolve()
    checkpoint_path = _latest_checkpoint(run_dir)
    if checkpoint_path is None:
        raise ValueError(f"No checkpoint found for run: {run_dir}")

    horizons = tuple(int(horizon) for horizon in spec["horizons"])
    batch_size = int(spec["batch_size"])
    stability_rollouts = int(spec["stability_rollouts"])
    stability_noise_scale = float(spec["stability_noise_scale"])

    cached = _load_cached_result(
        run_dir,
        checkpoint_path,
        horizons,
        batch_size,
        stability_rollouts,
        stability_noise_scale,
    )
    if cached is not None:
        return cached

    resolved_device = _resolve_device(str(spec["device"]))
    adapter, dataset, _ = _load_model_and_dataset(checkpoint_path, resolved_device)
    mse, stability, valid_episode_count, noise_std_mean = evaluate_model(
        adapter,
        dataset,
        horizons,
        batch_size=batch_size,
        stability_rollouts=stability_rollouts,
        stability_noise_scale=stability_noise_scale,
    )
    result = SeedEvaluation(
        algorithm=str(spec["algorithm"]),
        task=str(spec["task"]),
        seed=int(spec["seed"]),
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        mse=mse,
        stability=stability,
        episode_count=len(dataset),
        valid_episode_count=valid_episode_count,
        stability_rollouts=stability_rollouts,
        stability_noise_scale=stability_noise_scale,
        stability_noise_std_mean=noise_std_mean,
    )
    _save_cached_result(result, horizons, batch_size)
    return result


def aggregate_results(all_seeds_results: list[SeedEvaluation]) -> dict[str, dict[str, dict[str, Any]]]:
    aggregated: dict[str, dict[str, dict[str, Any]]] = {}
    tasks = sorted({result.task for result in all_seeds_results})
    for task in tasks:
        aggregated[task] = {}
        algorithms = sorted({result.algorithm for result in all_seeds_results if result.task == task})
        for algorithm in algorithms:
            matching = [result for result in all_seeds_results if result.task == task and result.algorithm == algorithm]
            horizons = sorted({horizon for result in matching for horizon in result.mse})
            mse_mean = {}
            mse_std = {}
            stability_mean = {}
            stability_std = {}
            for horizon in horizons:
                mse_values = [result.mse[horizon] for result in matching if horizon in result.mse and not math.isnan(result.mse[horizon])]
                stability_values = [result.stability[horizon] for result in matching if horizon in result.stability and not math.isnan(result.stability[horizon])]
                mse_mean[horizon] = statistics.fmean(mse_values) if mse_values else float("nan")
                mse_std[horizon] = statistics.pstdev(mse_values) if len(mse_values) > 1 else (0.0 if mse_values else float("nan"))
                stability_mean[horizon] = statistics.fmean(stability_values) if stability_values else float("nan")
                stability_std[horizon] = statistics.pstdev(stability_values) if len(stability_values) > 1 else (0.0 if stability_values else float("nan"))
            aggregated[task][algorithm] = {
                "label": LONG_HORIZON_STYLES.get(algorithm, {"label": algorithm})["label"],
                "color": LONG_HORIZON_STYLES.get(algorithm, {"color": "#4b5563"})["color"],
                "seed_results": matching,
                "mse_mean": mse_mean,
                "mse_std": mse_std,
                "stability_mean": stability_mean,
                "stability_std": stability_std,
            }
    return aggregated


def _configure_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle="--", linewidth=0.8, color="#dbe2ea", alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#334155", labelsize=10)


def _task_title(task: str) -> str:
    return MAIN_TABLE_TASK_LABELS.get(task, task.replace("_", " ").title())


def _save_figure(fig: plt.Figure, output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220 if output_format == "png" else None, bbox_inches="tight")
    plt.close(fig)


def _plot_task_mse_curve(ax: plt.Axes, task: str, task_results: dict[str, dict[str, Any]]) -> None:
    horizon_ticks: list[int] = []
    for algorithm in PLOT_ALGORITHM_ORDER:
        item = task_results.get(algorithm)
        if item is None:
            continue
        horizons = sorted(item["mse_mean"])
        horizon_ticks = horizons
        means = np.asarray([item["mse_mean"][horizon] for horizon in horizons], dtype=np.float64)
        stds = np.asarray([item["mse_std"][horizon] for horizon in horizons], dtype=np.float64)
        lower = np.clip(means - stds, 1e-12, None)
        upper = np.clip(means + stds, 1e-12, None)
        ax.plot(
            horizons,
            means,
            marker="o",
            linewidth=2.3,
            markersize=5.5,
            color=item["color"],
            label=item["label"],
        )
        ax.fill_between(horizons, lower, upper, color=item["color"], alpha=0.16, linewidth=0)
    _configure_axes(ax)
    ax.set_yscale("log")
    if horizon_ticks:
        ax.set_xticks(horizon_ticks)
    ax.set_xlabel("rollout horizon k", fontsize=11, color="#0f172a")
    ax.set_ylabel("latent MSE", fontsize=11, color="#0f172a")
    ax.set_title(f"Long-horizon consistency on {_task_title(task)}", fontsize=13, color="#0f172a")
    ax.legend(loc="upper right", fontsize=9, frameon=False)


def _plot_task_stability_curve(ax: plt.Axes, task: str, task_results: dict[str, dict[str, Any]]) -> None:
    horizon_ticks: list[int] = []
    for algorithm in PLOT_ALGORITHM_ORDER:
        item = task_results.get(algorithm)
        if item is None:
            continue
        horizons = sorted(item["stability_mean"])
        horizon_ticks = horizons
        means = np.asarray([item["stability_mean"][horizon] for horizon in horizons], dtype=np.float64)
        stds = np.asarray([item["stability_std"][horizon] for horizon in horizons], dtype=np.float64)
        lower = np.clip(means - stds, 1e-12, None)
        upper = np.clip(means + stds, 1e-12, None)
        ax.plot(
            horizons,
            means,
            marker="o",
            linewidth=2.3,
            markersize=5.5,
            color=item["color"],
            label=item["label"],
        )
        ax.fill_between(horizons, lower, upper, color=item["color"], alpha=0.16, linewidth=0)
    _configure_axes(ax)
    ax.set_yscale("log")
    if horizon_ticks:
        ax.set_xticks(horizon_ticks)
    ax.set_xlabel("rollout horizon k", fontsize=11, color="#0f172a")
    ax.set_ylabel("latent variance", fontsize=11, color="#0f172a")
    ax.set_title(f"Imagined rollout stability on {_task_title(task)}", fontsize=13, color="#0f172a")
    ax.legend(loc="upper left", fontsize=9, frameon=False)


def plot_mse_curves(
    results: dict[str, dict[str, dict[str, Any]]],
    by_task_root: Path,
    figures_root: Path,
    formats: list[str],
) -> list[Path]:
    exported: list[Path] = []
    ordered_tasks = [task for task in MAIN_TABLE_TASK_LABELS if task in results]
    for task in ordered_tasks:
        for fmt in formats:
            fig, ax = plt.subplots(figsize=(6.6, 4.2))
            _plot_task_mse_curve(ax, task, results[task])
            path = by_task_root / task / f"long_horizon_consistency.{fmt}"
            _save_figure(fig, path, fmt)
            exported.append(path)

    if ordered_tasks:
        cols = 2
        rows = math.ceil(len(ordered_tasks) / cols)
        for fmt in formats:
            fig, axes = plt.subplots(rows, cols, figsize=(13.2, 4.5 * rows))
            axes_array = np.atleast_1d(axes).reshape(rows, cols)
            for index, task in enumerate(ordered_tasks):
                row = index // cols
                col = index % cols
                _plot_task_mse_curve(axes_array[row, col], task, results[task])
            for index in range(len(ordered_tasks), rows * cols):
                row = index // cols
                col = index % cols
                axes_array[row, col].axis("off")
            fig.suptitle("Long-Horizon Consistency", fontsize=18, color="#0f172a", y=0.98)
            fig.subplots_adjust(hspace=0.3, wspace=0.22)
            path = figures_root / f"long_horizon_consistency_grid.{fmt}"
            _save_figure(fig, path, fmt)
            exported.append(path)
    return exported


def _plot_grouped_task_bars(
    results: dict[str, dict[str, dict[str, Any]]],
    metric_key: str,
    y_label: str,
    title: str,
    output_path: Path,
    output_format: str,
) -> None:
    summary_horizon = _summary_horizon(results)
    tasks = [task for task in MAIN_TABLE_TASK_LABELS if task in results]
    if not tasks:
        return
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    x = np.arange(len(tasks), dtype=np.float64)
    algorithms = [algorithm for algorithm in PLOT_ALGORITHM_ORDER if any(algorithm in results[task] for task in tasks)]
    width = 0.18 if len(algorithms) >= 4 else 0.22
    offsets = (np.arange(len(algorithms), dtype=np.float64) - (len(algorithms) - 1) / 2.0) * width
    for index, algorithm in enumerate(algorithms):
        color = LONG_HORIZON_STYLES[algorithm]["color"]
        label = LONG_HORIZON_STYLES[algorithm]["label"]
        means = []
        stds = []
        for task in tasks:
            item = results[task].get(algorithm)
            if item is None:
                means.append(float("nan"))
                stds.append(float("nan"))
                continue
            means.append(float(item[f"{metric_key}_mean"].get(summary_horizon, float("nan"))))
            stds.append(float(item[f"{metric_key}_std"].get(summary_horizon, float("nan"))))
        ax.bar(x + offsets[index], means, width=width * 0.92, yerr=stds, capsize=3, color=color, edgecolor="#ffffff", linewidth=1.0, label=label)
    _configure_axes(ax)
    ax.set_yscale("log")
    ax.set_xticks(x, [_task_title(task) for task in tasks])
    ax.set_ylabel(y_label, fontsize=11, color="#0f172a")
    ax.set_title(f"{title} (k={summary_horizon})", fontsize=15, color="#0f172a")
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
    _save_figure(fig, output_path, output_format)


def _plot_relative_summary_curve(
    results: dict[str, dict[str, dict[str, Any]]],
    figures_root: Path,
    formats: list[str],
) -> list[Path]:
    tasks = [task for task in MAIN_TABLE_TASK_LABELS if task in results]
    if not tasks:
        return []
    exported: list[Path] = []
    for fmt in formats:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        horizon_template = []
        for algorithm in PLOT_ALGORITHM_ORDER:
            rel_means = []
            rel_stds = []
            for horizon in DEFAULT_HORIZONS:
                task_values = []
                for task in tasks:
                    task_items = results[task]
                    if algorithm not in task_items:
                        continue
                    available = [item["mse_mean"].get(horizon, float("nan")) for item in task_items.values()]
                    available = [value for value in available if not math.isnan(value)]
                    if not available:
                        continue
                    best = min(available)
                    current = task_items[algorithm]["mse_mean"].get(horizon, float("nan"))
                    if math.isnan(current) or best <= 0:
                        continue
                    task_values.append(current / best)
                if task_values:
                    rel_means.append(statistics.fmean(task_values))
                    rel_stds.append(statistics.pstdev(task_values) if len(task_values) > 1 else 0.0)
                    horizon_template.append(horizon)
                else:
                    rel_means.append(float("nan"))
                    rel_stds.append(float("nan"))
            means = np.asarray(rel_means, dtype=np.float64)
            stds = np.asarray(rel_stds, dtype=np.float64)
            horizons = np.asarray(DEFAULT_HORIZONS, dtype=np.int64)
            valid = ~np.isnan(means)
            if not np.any(valid):
                continue
            horizons = horizons[valid]
            means = means[valid]
            stds = stds[valid]
            ax.plot(horizons, means, marker="o", linewidth=2.2, markersize=4.8, color=LONG_HORIZON_STYLES[algorithm]["color"], label=LONG_HORIZON_STYLES[algorithm]["label"])
            ax.fill_between(horizons, np.clip(means - stds, 1.0, None), means + stds, color=LONG_HORIZON_STYLES[algorithm]["color"], alpha=0.14, linewidth=0)
        _configure_axes(ax)
        ax.set_xticks(list(DEFAULT_HORIZONS))
        ax.set_xlabel("rollout horizon k", fontsize=11, color="#0f172a")
        ax.set_ylabel("relative latent MSE (task-best = 1.0)", fontsize=11, color="#0f172a")
        ax.set_title("Cross-Task Relative Long-Horizon Error", fontsize=15, color="#0f172a")
        ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
        path = figures_root / f"long_horizon_consistency_relative.{fmt}"
        _save_figure(fig, path, fmt)
        exported.append(path)
    return exported


def plot_stability_curves(
    results: dict[str, dict[str, dict[str, Any]]],
    by_task_root: Path,
    figures_root: Path,
    formats: list[str],
) -> list[Path]:
    exported: list[Path] = []
    ordered_tasks = [task for task in MAIN_TABLE_TASK_LABELS if task in results]
    for task in ordered_tasks:
        for fmt in formats:
            fig, ax = plt.subplots(figsize=(6.6, 4.2))
            _plot_task_stability_curve(ax, task, results[task])
            path = by_task_root / task / f"long_horizon_stability.{fmt}"
            _save_figure(fig, path, fmt)
            exported.append(path)

    if ordered_tasks:
        cols = 2
        rows = math.ceil(len(ordered_tasks) / cols)
        for fmt in formats:
            fig, axes = plt.subplots(rows, cols, figsize=(12.4, 4.4 * rows))
            axes_array = np.atleast_1d(axes).reshape(rows, cols)
            for index, task in enumerate(ordered_tasks):
                row = index // cols
                col = index % cols
                _plot_task_stability_curve(axes_array[row, col], task, results[task])
            for index in range(len(ordered_tasks), rows * cols):
                row = index // cols
                col = index % cols
                axes_array[row, col].axis("off")
            fig.suptitle("Imagined Rollout Stability", fontsize=18, color="#0f172a", y=0.98)
            fig.subplots_adjust(hspace=0.3, wspace=0.22)
            path = figures_root / f"long_horizon_stability_grid.{fmt}"
            _save_figure(fig, path, fmt)
            exported.append(path)
    return exported


def generate_table(results: dict[str, dict[str, dict[str, Any]]], summary_root: Path) -> list[Path]:
    summary_horizon = _summary_horizon(results)
    rows = []
    for task, task_results in results.items():
        for algorithm, item in task_results.items():
            rows.append(
                {
                    "task": task,
                    "algorithm": algorithm,
                    "label": item["label"],
                    f"mse{summary_horizon}_mean": item["mse_mean"].get(summary_horizon, float("nan")),
                    f"mse{summary_horizon}_std": item["mse_std"].get(summary_horizon, float("nan")),
                    f"stability{summary_horizon}_mean": item["stability_mean"].get(summary_horizon, float("nan")),
                    f"stability{summary_horizon}_std": item["stability_std"].get(summary_horizon, float("nan")),
                    "num_seeds": len(item["seed_results"]),
                }
            )

    csv_path = summary_root / f"long_horizon_mse{summary_horizon}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "algorithm",
                "label",
                f"mse{summary_horizon}_mean",
                f"mse{summary_horizon}_std",
                f"stability{summary_horizon}_mean",
                f"stability{summary_horizon}_std",
                "num_seeds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    tex_lines = [
        "\\begin{tabular}{llc}",
        "\\toprule",
        f"Task & Method & MSE@{summary_horizon} \\\\",
        "\\midrule",
    ]
    for task in MAIN_TABLE_TASK_LABELS:
        if task not in results:
            continue
        task_rows = [row for row in rows if row["task"] == task and not math.isnan(float(row[f"mse{summary_horizon}_mean"]))]
        best = min((float(row[f"mse{summary_horizon}_mean"]) for row in task_rows), default=None)
        for row in task_rows:
            value = f"{float(row[f'mse{summary_horizon}_mean']):.4f} $\\pm$ {float(row[f'mse{summary_horizon}_std']):.4f}"
            if best is not None and math.isclose(float(row[f"mse{summary_horizon}_mean"]), best, rel_tol=1e-9, abs_tol=1e-12):
                value = f"\\textbf{{{value}}}"
            tex_lines.append(f"{MAIN_TABLE_TASK_LABELS.get(task, task)} & {row['label']} & {value} \\\\")
        tex_lines.append("\\midrule")
    if tex_lines[-1] == "\\midrule":
        tex_lines.pop()
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    tex_path = summary_root / f"long_horizon_mse{summary_horizon}.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")

    json_rows = []
    for row in rows:
        json_row = dict(row)
        for key in (
            f"mse{summary_horizon}_mean",
            f"mse{summary_horizon}_std",
            f"stability{summary_horizon}_mean",
            f"stability{summary_horizon}_std",
        ):
            json_row[key] = float(json_row[key])
        json_rows.append(json_row)
    json_path = summary_root / "long_horizon_metrics.json"
    json_path.write_text(json.dumps(json_rows, indent=2, sort_keys=True), encoding="utf-8")
    return [csv_path, tex_path, json_path]


def export_campaign_long_horizon(
    seed_runs: list[SeedRun],
    summary_root: Path,
    figures_root: Path,
    by_task_root: Path,
    tasks: list[str],
    algorithms: list[str],
    formats: list[str],
    device: str | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    batch_size: int = 2048,
    stability_rollouts: int = 5,
    stability_noise_scale: float = 0.01,
    workers: int = 4,
) -> list[Path]:
    supported_algorithms = [algorithm for algorithm in algorithms if algorithm in SUPPORTED_ALGOS]
    selected_runs = [
        run
        for run in seed_runs
        if run.algorithm in supported_algorithms and run.task in tasks
    ]
    if not selected_runs:
        return []

    resolved_devices = _resolve_device_list(device)
    first_device = _resolve_device(resolved_devices[0])
    multi_gpu = len(resolved_devices) > 1 and all(item.startswith("cuda") for item in resolved_devices)
    if first_device.type == "cuda" and not multi_gpu:
        workers = 1
    requested_workers = int(workers)
    worker_cap = len(resolved_devices) if multi_gpu else (os.cpu_count() or 1)
    max_workers = max(1, min(requested_workers, len(selected_runs), worker_cap))

    run_specs = [
        {
            "run_dir": str(run.run_dir),
            "algorithm": run.algorithm,
            "task": run.task,
            "seed": run.seed,
            "horizons": list(horizons),
            "device": resolved_devices[index % len(resolved_devices)],
            "batch_size": int(batch_size),
            "stability_rollouts": int(stability_rollouts),
            "stability_noise_scale": float(stability_noise_scale),
        }
        for index, run in enumerate(selected_runs)
    ]

    seed_results: list[SeedEvaluation] = []
    if max_workers <= 1:
        for spec in run_specs:
            seed_results.append(_evaluate_seed_run_worker(spec))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_evaluate_seed_run_worker, spec) for spec in run_specs]
            for future in as_completed(futures):
                seed_results.append(future.result())

    aggregated = aggregate_results(seed_results)
    summary_horizon = _summary_horizon(aggregated)
    exported: list[Path] = []
    exported.extend(generate_table(aggregated, summary_root))
    exported.extend(plot_mse_curves(aggregated, by_task_root, figures_root, formats))
    exported.extend(plot_stability_curves(aggregated, by_task_root, figures_root, formats))
    exported.extend(_plot_relative_summary_curve(aggregated, figures_root, formats))
    for fmt in formats:
        mse_grouped = figures_root / f"long_horizon_mse{summary_horizon}_grouped.{fmt}"
        _plot_grouped_task_bars(
            aggregated,
            metric_key="mse",
            y_label=f"MSE@{summary_horizon}",
            title="Long-Horizon Error",
            output_path=mse_grouped,
            output_format=fmt,
        )
        exported.append(mse_grouped)
        stability_grouped = figures_root / f"long_horizon_stability{summary_horizon}_grouped.{fmt}"
        _plot_grouped_task_bars(
            aggregated,
            metric_key="stability",
            y_label=f"Latent variance at k={summary_horizon}",
            title="Imagined Rollout Stability",
            output_path=stability_grouped,
            output_format=fmt,
        )
        exported.append(stability_grouped)
    return exported
