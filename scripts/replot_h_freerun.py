from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import (
    ANALYSIS_MECHANISM_FREERUN_MANIFEST,
    RESULTS_APPENDIX_FIG_ROOT,
    RESULTS_MECHANISM_FIG_ROOT,
    ensure_repo_on_path,
    read_csv_rows,
    resolve_repo_path,
)

ensure_repo_on_path()
from hamworld.dynamics_eval import plot_h_freerun_per_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replot Hamiltonian free-run figures from final trace manifests.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ANALYSIS_MECHANISM_FREERUN_MANIFEST,
    )
    return parser.parse_args()


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    return {
        "H": np.asarray(payload["H"]),
        "Q": np.asarray(payload["Q"]),
        "P": np.asarray(payload["P"]),
        "valid_mask": np.asarray(payload["valid_mask"]),
    }


def _group_rows(rows: list[dict[str, str]], paper_group: str) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    grouped: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for row in rows:
        if row["paper_group"] != paper_group:
            continue
        grouped.setdefault(row["task"], {})[row["condition"]] = _load_arrays(resolve_repo_path(row["trace_path"]))
    return grouped


def main() -> int:
    args = parse_args()
    rows = read_csv_rows(args.manifest)

    mechanism_group = _group_rows(rows, "mechanism_main_abs_h")
    appendix_group = _group_rows(rows, "appendix_delta_h")

    saved_main = plot_h_freerun_per_task(
        mechanism_group,
        RESULTS_MECHANISM_FIG_ROOT,
        subtract_initial=False,
        filename_prefix="h_freerun_abs",
        ylabel=r"$H_t$",
    )
    saved_appendix = plot_h_freerun_per_task(
        appendix_group,
        RESULTS_APPENDIX_FIG_ROOT,
        subtract_initial=True,
        filename_prefix="h_freerun",
        ylabel=r"$H_t - H_0$",
    )
    for path in [*saved_main, *saved_appendix]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
