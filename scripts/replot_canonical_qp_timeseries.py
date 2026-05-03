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

from common import ANALYSIS_FIG_ROOT, ANALYSIS_MECHANISM_FREERUN_MANIFEST, read_csv_rows, resolve_repo_path


TASK_ORDER = ["finger_spin", "cheetah_run"]
TASK_LABELS = {
    "finger_spin": "Finger Spin",
    "cheetah_run": "Cheetah Run",
}
PANEL_COLORS = {
    ("finger_spin", "q"): "#cf425f",
    ("finger_spin", "p"): "#9b5de5",
    ("cheetah_run", "q"): "#1b8a89",
    ("cheetah_run", "p"): "#3569d4",
}
OUT_STEM_SUMMARY = ANALYSIS_FIG_ROOT / "canonical_qp_no_action_pc_grid"
OUT_STEM_ALL = ANALYSIS_FIG_ROOT / "canonical_qp_no_action_pc_grid_all_episodes"
NUM_REPRESENTATIVE_EPISODES = 4


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 11.8,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11.0,
        "axes.labelcolor": "#334155",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.5,
        "figure.dpi": 220,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replot canonical q/p timeseries under no-action zero-damp freerun traces.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ANALYSIS_MECHANISM_FREERUN_MANIFEST,
    )
    parser.add_argument(
        "--style",
        choices=("summary", "all"),
        default="summary",
        help="summary: representative episodes + IQR + median; all: all episode traces with a bold mean curve.",
    )
    return parser.parse_args()


def _pc1_scores(latent: np.ndarray) -> np.ndarray:
    flattened = latent.reshape(-1, latent.shape[-1])
    centered = flattened - flattened.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    scores = centered @ axis
    scores = scores.reshape(latent.shape[0], latent.shape[1])
    median_delta = float(np.median(scores[:, -1] - scores[:, 0]))
    if median_delta < 0:
        scores = -scores
    return scores


def _load_task_scores(trace_path: Path) -> dict[str, np.ndarray]:
    payload = np.load(trace_path, allow_pickle=True)
    q = np.asarray(payload["Q"], dtype=np.float32)
    p = np.asarray(payload["P"], dtype=np.float32)
    valid = np.asarray(payload["valid_mask"], dtype=bool)
    if not valid.all():
        raise ValueError(f"Expected dense valid_mask for {trace_path}")
    return {"q": _pc1_scores(q), "p": _pc1_scores(p)}


def _select_representative_indices(series: np.ndarray, num_keep: int = NUM_REPRESENTATIVE_EPISODES) -> np.ndarray:
    num_episodes = series.shape[0]
    if num_episodes <= num_keep:
        return np.arange(num_episodes, dtype=np.int32)

    features = np.column_stack(
        [
            series[:, 0],
            series[:, -1],
            series.mean(axis=1),
            series.std(axis=1),
            series.max(axis=1),
            series.min(axis=1),
        ]
    )
    features = (features - features.mean(axis=0, keepdims=True)) / (features.std(axis=0, keepdims=True) + 1e-8)

    center = np.median(features, axis=0)
    selected = [int(np.argmin(np.linalg.norm(features - center, axis=1)))]
    while len(selected) < num_keep:
        remaining = [idx for idx in range(num_episodes) if idx not in selected]
        distances = []
        for idx in remaining:
            distance = min(np.linalg.norm(features[idx] - features[chosen]) for chosen in selected)
            distances.append(distance)
        selected.append(int(remaining[int(np.argmax(distances))]))
    return np.asarray(sorted(selected), dtype=np.int32)


def _plot_panel(ax: plt.Axes, series: np.ndarray, *, color: str, title: str) -> None:
    x = np.arange(series.shape[1], dtype=np.int32)
    q25 = np.percentile(series, 25.0, axis=0)
    q50 = np.percentile(series, 50.0, axis=0)
    q75 = np.percentile(series, 75.0, axis=0)
    representative = series[_select_representative_indices(series)]

    ax.set_facecolor("#fbfcfe")
    for line in representative:
        ax.plot(x, line, color=color, linewidth=2.1, alpha=0.30, solid_capstyle="round")
    ax.fill_between(x, q25, q75, color=color, alpha=0.14, linewidth=0)
    ax.plot(x, q50, color=color, linewidth=4.5, solid_capstyle="round")
    ax.scatter([x[0]], [q50[0]], s=46, facecolor="white", edgecolor=color, linewidth=1.7, zorder=3)
    ax.scatter([x[-1]], [q50[-1]], s=50, color=color, edgecolor="white", linewidth=1.0, zorder=3)

    ax.set_title(title, loc="left", color="#0f172a", pad=7)
    ax.set_xlim(x[0], x[-1])
    ax.grid(True, axis="both", linestyle="--", linewidth=0.7, color="#d8e1ec", alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="x", which="major", pad=3)
    ax.tick_params(axis="y", which="major", pad=3)


def _plot_panel_all(ax: plt.Axes, series: np.ndarray, *, color: str, title: str) -> None:
    x = np.arange(series.shape[1], dtype=np.int32)

    ax.set_facecolor("#fbfcfe")
    for line in series:
        ax.plot(x, line, color=color, linewidth=2.2, alpha=0.44, solid_capstyle="round")

    ax.set_title(title, loc="left", color="#0f172a", pad=7)
    ax.set_xlim(x[0], x[-1])
    ax.grid(True, axis="both", linestyle="--", linewidth=0.7, color="#d8e1ec", alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="x", which="major", pad=3)
    ax.tick_params(axis="y", which="major", pad=3)


def main() -> int:
    args = parse_args()
    manifest_rows = [
        row
        for row in read_csv_rows(args.manifest)
        if row["paper_group"] == "phase_portrait" and row["condition"] == "no_action"
    ]
    row_by_task = {row["task"]: row for row in manifest_rows}

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.3))
    for row_idx, task in enumerate(TASK_ORDER):
        if task not in row_by_task:
            continue
        scores = _load_task_scores(resolve_repo_path(row_by_task[task]["trace_path"]))
        for col_idx, component in enumerate(("q", "p")):
            ax = axes[row_idx, col_idx]
            plot_fn = _plot_panel if args.style == "summary" else _plot_panel_all
            plot_fn(
                ax,
                scores[component],
                color=PANEL_COLORS[(task, component)],
                title=f"{TASK_LABELS[task]} · {component.upper()} PC-1 across 10 episodes",
            )
            ax.set_xlabel("Step")
            ax.set_ylabel("PC-1 score")

    fig.suptitle("Canonical latent trajectories under no-action zero-damp freerun", fontsize=13.2, color="#0f172a", y=0.975)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), pad=0.6, w_pad=1.0, h_pad=1.2)
    out_stem = OUT_STEM_SUMMARY if args.style == "summary" else OUT_STEM_ALL
    fig.savefig(out_stem.with_suffix(".png"))
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)
    print(out_stem.with_suffix(".png"))
    print(out_stem.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
