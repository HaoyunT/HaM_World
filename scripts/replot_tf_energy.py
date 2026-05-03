from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import (
    RESULTS_APPENDIX_FIG_ROOT,
    RESULTS_MECHANISM_FIG_ROOT,
    RESULTS_MECHANISM_TRACE_EP10_ROOT,
    RESULTS_MECHANISM_TRACE_QUICK_ROOT,
    ensure_repo_on_path,
)

ensure_repo_on_path()
from hamworld.dynamics_eval import load_trace_directory, plot_energy_evolution_per_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replot teacher-forced Hamiltonian energy figures from paper mechanism traces.")
    parser.add_argument("--bundle", choices=["quick_seed7", "ep10_seed7"], default="ep10_seed7")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    traces_dir = (RESULTS_MECHANISM_TRACE_QUICK_ROOT if args.bundle == "quick_seed7" else RESULTS_MECHANISM_TRACE_EP10_ROOT) / "traces"
    grouped = load_trace_directory(traces_dir)
    tasks = ["finger_spin", "cheetah_run"]
    saved_abs = plot_energy_evolution_per_task(
        grouped,
        RESULTS_MECHANISM_FIG_ROOT,
        task_filter=tasks,
        quantity="absolute",
        filename_prefix="energy_evolution_abs",
    )
    saved_delta = plot_energy_evolution_per_task(
        grouped,
        RESULTS_APPENDIX_FIG_ROOT,
        task_filter=tasks,
        quantity="drift",
        filename_prefix="energy_evolution",
    )
    for path in [*saved_abs, *saved_delta]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
