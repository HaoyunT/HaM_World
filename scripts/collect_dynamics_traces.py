#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import ensure_repo_on_path, repo_relative_path

ensure_repo_on_path()

from hamworld.dynamics_eval import (
    MODE_ORDER,
    collect_trace_bundles,
    resolve_task_runs,
    save_trace_bundle,
    summarize_trace_bundle,
    trace_filename,
    write_summary_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect teacher-forced and imagined dynamics traces from HaM-World checkpoints.")
    parser.add_argument("--task-root", required=True, help="Root directory containing per-task run folders, e.g. exp/.../runs/hamworld")
    parser.add_argument("--seed", required=True, type=int, help="Seed to collect, e.g. 7")
    parser.add_argument("--output-dir", required=True, help="Output directory for traces/ and dynamics_summary.csv")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task ids to collect. Defaults to every task under --task-root.")
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of eval episodes to roll out per task.")
    parser.add_argument("--max-steps", type=int, default=200, help="Maximum steps per eval episode.")
    parser.add_argument("--device", default="auto", help="Torch device for loading/running the checkpoint, e.g. cuda, cuda:0, cpu, auto.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    task_runs = resolve_task_runs(args.task_root, seed=args.seed, tasks=args.tasks)
    for task_run in task_runs:
        print(f"[collect] task={task_run.task} seed={task_run.seed} checkpoint={task_run.checkpoint_path}")
        bundles = collect_trace_bundles(
            task_run,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            device_name=args.device,
        )
        for mode in MODE_ORDER:
            bundle = bundles[mode]
            trace_path = save_trace_bundle(bundle, traces_dir / trace_filename(task_run.task, task_run.seed, mode))
            row = summarize_trace_bundle(bundle)
            row["trace_path"] = repo_relative_path(trace_path)
            summary_rows.append(row)
            print(f"  -> {trace_path}")

    task_order = {task: index for index, task in enumerate(["reacher_easy", "finger_spin", "cheetah_run", "cartpole_swingup"])}
    mode_order = {mode: index for index, mode in enumerate(MODE_ORDER)}
    summary_rows.sort(key=lambda row: (task_order.get(str(row["task"]), 999), mode_order.get(str(row["mode"]), 999)))
    summary_path = write_summary_csv(summary_rows, output_dir / "dynamics_summary.csv")
    print(f"[summary] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
