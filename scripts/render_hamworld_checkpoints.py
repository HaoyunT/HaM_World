from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import (
    ANALYSIS_ROLLOUT_CHECKPOINT_CACHE_ROOT,
    ANALYSIS_ROLLOUT_MANIFEST,
    ANALYSIS_ROLLOUT_RENDER_ROOT,
    ensure_repo_on_path,
    resolve_repo_path,
)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
ensure_repo_on_path()
from hamworld.agent import HaMWorldAgent
from hamworld.runtime import make_env
from hamworld.world_model import infer_checkpoint_step

DEFAULT_MANIFEST = ANALYSIS_ROLLOUT_MANIFEST
DEFAULT_OUTPUT_DIR = ANALYSIS_ROLLOUT_RENDER_ROOT
DEFAULT_CACHE_DIR = ANALYSIS_ROLLOUT_CHECKPOINT_CACHE_ROOT
TASKS = ("cartpole_swingup", "cheetah_run", "finger_spin", "reacher_easy")


@dataclass
class RunRecord:
    task: str
    seed: int
    final_step: int
    final_eval: float
    best_eval: float
    auc: float
    run_dir: str

    @property
    def local_run_dir(self) -> Path:
        return resolve_repo_path(self.run_dir)

    @property
    def local_checkpoint_path(self) -> Path:
        return self.local_run_dir / "checkpoints" / f"checkpoint_{self.final_step}.pt"


def parse_manifest(manifest_path: Path) -> dict[str, RunRecord]:
    records: dict[str, RunRecord] = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["algorithm"] != "hamworld":
                continue
            task = row["task"]
            if task not in TASKS:
                continue
            record = RunRecord(
                task=task,
                seed=int(row["seed"]),
                final_step=int(row["final_step"]),
                final_eval=float(row["final_eval"]),
                best_eval=float(row["best_eval"]),
                auc=float(row["auc"]),
                run_dir=row["run_dir"],
            )
            current = records.get(task)
            if current is None or (record.final_eval, record.best_eval, record.auc, -record.seed) > (
                current.final_eval,
                current.best_eval,
                current.auc,
                -current.seed,
            ):
                records[task] = record
    missing = [task for task in TASKS if task not in records]
    if missing:
        raise ValueError(f"Missing hamworld runs for tasks: {', '.join(missing)}")
    return records


def ensure_checkpoint(record: RunRecord, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if record.local_checkpoint_path.exists() and record.local_checkpoint_path.stat().st_size > 1024:
        return record.local_checkpoint_path

    dest = cache_dir / f"{record.task}_seed{record.seed}_checkpoint_{record.final_step}.pt"
    if dest.exists() and dest.stat().st_size > 1024:
        return dest
    raise FileNotFoundError(
        f"Missing checkpoint for task={record.task} seed={record.seed}: "
        f"looked in local run dir {record.local_checkpoint_path} and cache {dest}"
    )


def build_agent(checkpoint_path: Path) -> tuple[HaMWorldAgent, dict[str, Any]]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = state["config"]
    config.setdefault("experiment", {})
    config["experiment"]["device"] = "cpu"

    env, env_spec = make_env(config["task"], int(config["experiment"].get("seed", 0)) + 5000)
    del env

    agent = HaMWorldAgent(
        config=config,
        obs_dim=int(env_spec.observation_shape[0]),
        action_dim=int(env_spec.action_shape[0]),
        action_low=env_spec.action_low,
        action_high=env_spec.action_high,
    )
    agent_state = state.get("agent_state")
    if agent_state is not None:
        agent.load_state_dict(agent_state)
    else:
        model_state = state.get("model")
        if model_state is None:
            raise ValueError(f"Checkpoint missing both agent_state and model: {checkpoint_path}")
        agent.world_model.load_state_dict(model_state)
    if agent.current_step is None:
        agent.set_training_step(infer_checkpoint_step(state, checkpoint_path))
    return agent, config


def render_episode(
    task_config: dict[str, Any],
    agent: HaMWorldAgent,
    eval_seed: int,
    image_height: int,
    image_width: int,
    camera_id: int,
) -> dict[str, Any]:
    env, _env_spec = make_env(task_config, eval_seed)
    if hasattr(agent, "reset"):
        agent.reset()
    obs = env.reset()

    total_return = 0.0
    frames: list[np.ndarray] = []
    rewards: list[float] = []

    # capture the initial frame as well, so the trajectory starts from t=0
    frames.append(env.env.physics.render(height=image_height, width=image_width, camera_id=camera_id))
    rewards.append(0.0)

    while True:
        action = agent.act(obs, eval_mode=True)
        obs, reward, done, _info = env.step(action)
        total_return += float(reward)
        frames.append(env.env.physics.render(height=image_height, width=image_width, camera_id=camera_id))
        rewards.append(float(reward))
        if done:
            break

    return {
        "frames": frames,
        "rewards": rewards,
        "episode_return": total_return,
        "decision_steps": len(frames) - 1,
    }


def select_frame_indices(total: int, num: int) -> list[int]:
    if total <= 0:
        return []
    if num >= total:
        return list(range(total))
    # evenly spaced including first and last
    return [int(round(i * (total - 1) / (num - 1))) for i in range(num)]


def select_success_indices(rewards: list[float], min_gap_frac: float = 0.3) -> list[int]:
    """Start, peak single-step reward (kept away from endpoints), end."""
    n = len(rewards)
    if n <= 3:
        return list(range(n))
    margin = max(1, int(round(n * min_gap_frac)))
    lo, hi = margin, n - 1 - margin
    if hi <= lo:
        return [0, n // 2, n - 1]
    # argmax restricted to the interior window so the middle frame is visually distinct
    interior = rewards[lo : hi + 1]
    peak = lo + int(np.argmax(interior))
    return [0, peak, n - 1]


def make_grid(frames: list[np.ndarray], pad: int = 4) -> np.ndarray:
    h, w = frames[0].shape[:2]
    n = len(frames)
    grid = np.full((h, w * n + pad * (n - 1), 3), 255, dtype=np.uint8)
    for i, fr in enumerate(frames):
        x = i * (w + pad)
        grid[:, x : x + w] = fr
    return grid


def save_image(frame: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render rollout screenshot frames from local HaM-World checkpoints.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--eval-seed-offset", type=int, default=5000)
    parser.add_argument("--tasks", nargs="*", default=list(TASKS), help="Optional subset of task ids to render.")
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--num-frames", type=int, default=8,
                        help="Frames per task to sample evenly across the rollout.")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="dm_control camera id; kept identical across tasks.")
    parser.add_argument("--selection-mode", choices=("even", "success"), default="even",
                        help="even: uniform sampling; success: start / peak-reward / end.")
    args = parser.parse_args()

    selected_tasks = [task for task in args.tasks if task in TASKS]
    if not selected_tasks:
        raise ValueError("No valid tasks selected.")
    records = parse_manifest(args.manifest)
    summary: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_task_strips: list[np.ndarray] = []

    for task in selected_tasks:
        record = records[task]
        checkpoint = ensure_checkpoint(record, args.cache_dir)
        agent, config = build_agent(checkpoint)
        eval_seed = record.seed + args.eval_seed_offset
        episode = render_episode(
            config["task"], agent, eval_seed,
            args.image_height, args.image_width, args.camera_id,
        )
        frames = episode["frames"]
        if args.selection_mode == "success":
            idxs = select_success_indices(episode["rewards"])
        else:
            idxs = select_frame_indices(len(frames), args.num_frames)
        sampled = [frames[i] for i in idxs]

        task_dir = args.output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        for k, (i, fr) in enumerate(zip(idxs, sampled)):
            p = task_dir / f"{task}_seed{record.seed}_t{i:03d}.png"
            save_image(fr, p)
            frame_paths.append(str(p.resolve()))

        strip = make_grid(sampled)
        strip_path = args.output_dir / f"{task}_seed{record.seed}_strip.png"
        save_image(strip, strip_path)
        per_task_strips.append(strip)

        item = {
            "task": task,
            "seed": record.seed,
            "final_eval": record.final_eval,
            "best_eval": record.best_eval,
            "auc": record.auc,
            "checkpoint": str(checkpoint.resolve()),
            "strip_path": str(strip_path.resolve()),
            "frame_paths": frame_paths,
            "frame_indices": idxs,
            "camera_id": args.camera_id,
            "eval_seed": eval_seed,
            "episode_return": episode["episode_return"],
            "decision_steps": episode["decision_steps"],
        }
        summary.append(item)
        print(
            f"[render] task={task} seed={record.seed} ret={episode['episode_return']:.2f} "
            f"frames={len(idxs)}/{len(frames)} strip={strip_path}"
        )

    # 4xN overview: stack per-task strips vertically (all use same camera_id)
    h = max(s.shape[0] for s in per_task_strips)
    w = max(s.shape[1] for s in per_task_strips)
    pad = 6
    overview = np.full((h * len(per_task_strips) + pad * (len(per_task_strips) - 1), w, 3), 255, dtype=np.uint8)
    for i, s in enumerate(per_task_strips):
        y = i * (h + pad)
        overview[y : y + s.shape[0], : s.shape[1]] = s
    overview_path = args.output_dir / "overview_4tasks.png"
    save_image(overview, overview_path)
    print(f"[render] overview {overview_path}")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
