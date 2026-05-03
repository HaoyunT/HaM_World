#!/usr/bin/env python3
"""Evaluate the learned Hamiltonian H on free-running env rollouts.

Two scenarios per task:
  - no_action     : action is all zeros (free dynamics, no external force)
  - random_action : action sampled uniformly in [action_low, action_high]

For each (task, mode) we run --num-episodes rollouts of length --max-steps,
encode every observation through the learned encoder, and read H from the
learned Hamiltonian head. Outputs npz traces and per-task mean +/- std plots.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import FINAL_ROOT, ensure_repo_on_path

MPLCONFIGDIR = FINAL_ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
ensure_repo_on_path()

from hamworld.dynamics_eval import (  # noqa: E402
    _load_agent_and_env,
    plot_h_freerun_per_task,
    resolve_task_runs,
)


FREERUN_MODES = ("no_action", "random_action")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True, help="Root containing per-task HaM-World run dirs.")
    parser.add_argument("--seed", type=int, required=True, help="Run seed (e.g. 7).")
    parser.add_argument("--tasks", nargs="*", default=["finger_spin", "cheetah_run"], help="Task ids to evaluate.")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes per (task, mode).")
    parser.add_argument("--max-steps", type=int, default=200, help="Steps per episode.")
    parser.add_argument("--output-dir", required=True, help="Where to save npz traces.")
    parser.add_argument("--figures-dir", default=None, help="Where to save the per-task H figures (defaults to <output-dir>/figures).")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, cpu, auto.")
    parser.add_argument("--rng-seed", type=int, default=20260426, help="Numpy RNG seed for random-action sampling.")
    parser.add_argument("--action-repeat", type=int, default=1, help="Override env action_repeat for rollouts (default 1 so --max-steps env steps actually run before dm-control internal time limit).")
    return parser.parse_args()


def _sample_action(mode: str, action_low: np.ndarray, action_high: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if mode == "no_action":
        return np.zeros_like(action_low, dtype=np.float32)
    if mode == "random_action":
        return rng.uniform(action_low, action_high).astype(np.float32)
    raise ValueError(f"Unknown mode: {mode}")


def _rollout_one_mode(task_run, mode: str, num_episodes: int, max_steps: int, device_name: str, rng: np.random.Generator, action_repeat: int | None = None):
    import torch

    agent, env, device = _load_agent_and_env(task_run, device_name)
    if action_repeat is not None and hasattr(env, "action_repeat"):
        env.action_repeat = int(action_repeat)
    if hasattr(env, "max_episode_steps"):
        env.max_episode_steps = max(int(env.max_episode_steps), max_steps + 1)
    world_model = agent.world_model
    q_dim = int(world_model.q_dim)
    p_dim = int(world_model.p_dim)

    action_low = np.asarray(env.action_low, dtype=np.float32)
    action_high = np.asarray(env.action_high, dtype=np.float32)

    H_buf = np.full((num_episodes, max_steps), np.nan, dtype=np.float32)
    Q_buf = np.full((num_episodes, max_steps, q_dim), np.nan, dtype=np.float32)
    P_buf = np.full((num_episodes, max_steps, p_dim), np.nan, dtype=np.float32)
    valid = np.zeros((num_episodes, max_steps), dtype=bool)

    try:
        for ep in range(num_episodes):
            agent.reset()
            obs = np.asarray(env.reset(seed=int(task_run.seed) + 5000 + ep), dtype=np.float32)
            for step_idx in range(max_steps):
                with torch.no_grad():
                    z = world_model.encode(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                    q, p, _c = world_model.split_latent(z)
                    h_val = float(world_model.energy_head(q, p).reshape(-1)[0].item())
                H_buf[ep, step_idx] = h_val
                Q_buf[ep, step_idx] = q.squeeze(0).detach().cpu().numpy()
                P_buf[ep, step_idx] = p.squeeze(0).detach().cpu().numpy()
                valid[ep, step_idx] = True

                action = _sample_action(mode, action_low, action_high, rng)
                next_obs, _reward, done, _info = env.step(action)
                if done:
                    break
                obs = np.asarray(next_obs, dtype=np.float32)
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()

    meta = {
        "task": task_run.task,
        "seed": int(task_run.seed),
        "mode": mode,
        "num_episodes": int(num_episodes),
        "max_steps": int(max_steps),
        "q_dim": q_dim,
        "p_dim": p_dim,
        "checkpoint": str(task_run.checkpoint_path),
    }
    return {"H": H_buf, "Q": Q_buf, "P": P_buf, "valid_mask": valid}, meta


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir).expanduser().resolve() if args.figures_dir else output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    task_runs = resolve_task_runs(args.task_root, seed=args.seed, tasks=args.tasks)
    rng = np.random.default_rng(int(args.rng_seed))

    traces_by_task: dict[str, dict[str, dict]] = {}
    for task_run in task_runs:
        traces_by_task.setdefault(task_run.task, {})
        for mode in FREERUN_MODES:
            print(f"[freerun] task={task_run.task} seed={task_run.seed} mode={mode} ckpt={task_run.checkpoint_path}")
            arrays, meta = _rollout_one_mode(
                task_run, mode, args.num_episodes, args.max_steps, args.device, rng, action_repeat=args.action_repeat
            )
            out_path = traces_dir / f"h_freerun_{task_run.task}_seed{task_run.seed}_{mode}.npz"
            np.savez(out_path, **arrays, **{f"meta_{k}": np.asarray(v) for k, v in meta.items()})
            print(f"  -> {out_path}")
            traces_by_task[task_run.task][mode] = arrays

    paths = plot_h_freerun_per_task(traces_by_task, figures_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
