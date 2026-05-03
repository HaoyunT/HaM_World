from __future__ import annotations

import argparse
from pathlib import Path

from common import RESULTS_MECHANISM_ROOT, RUNS_MAIN_ROOT, run_repo_python


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recollect Hamiltonian free-run traces from paper main runs.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_MECHANISM_ROOT / "freerun" / "recollected_seed7",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_root = RUNS_MAIN_ROOT / "model_based" / "runs" / "hamworld"
    run_repo_python(
        "scripts/eval_h_freerun.py",
        [
            "--task-root",
            str(task_root),
            "--seed",
            str(args.seed),
            "--output-dir",
            str(args.output_dir),
            "--num-episodes",
            str(args.num_episodes),
            "--max-steps",
            str(args.max_steps),
            "--device",
            str(args.device),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
