from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import ANALYSIS_MECHANISM_FREERUN_MANIFEST, RESULTS_MECHANISM_FIG_ROOT, read_csv_rows, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replot simple q/p phase portrait from paper free-run traces.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ANALYSIS_MECHANISM_FREERUN_MANIFEST,
    )
    return parser.parse_args()


def _load_qp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    q = np.asarray(payload["Q"], dtype=np.float32)
    p = np.asarray(payload["P"], dtype=np.float32)
    valid = np.asarray(payload["valid_mask"], dtype=bool)
    episode = 0
    steps = np.where(valid[episode])[0]
    if steps.size == 0:
        return np.asarray([]), np.asarray([])
    q_series = q[episode, steps, 0]
    p_series = p[episode, steps, 0]
    return q_series, p_series


def main() -> int:
    args = parse_args()
    rows = [row for row in read_csv_rows(args.manifest) if row["paper_group"] == "phase_portrait"]
    rows.sort(key=lambda row: row["task"])
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2), dpi=250)
    titles = {"finger_spin": "Finger Spin", "cheetah_run": "Cheetah Run"}
    for ax, row in zip(axes, rows):
        q_series, p_series = _load_qp(resolve_repo_path(row["trace_path"]))
        if q_series.size == 0:
            continue
        color = "#1f4ea8" if row["task"] == "cheetah_run" else "#c3324b"
        ax.plot(q_series, p_series, color=color, linewidth=1.4, alpha=0.95)
        ax.scatter(q_series[0], p_series[0], color="#111827", s=10, zorder=3)
        ax.set_title(titles.get(row["task"], row["task"]))
        ax.set_xlabel("q[0]")
        ax.set_ylabel("p[0]")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.5)
    out_stem = RESULTS_MECHANISM_FIG_ROOT / "phase_qp_no_action_combined"
    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"))
    plt.close(fig)
    print(out_stem.with_suffix(".pdf"))
    print(out_stem.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
