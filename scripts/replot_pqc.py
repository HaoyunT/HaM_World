from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import gaussian_kde

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import RESULTS_MECHANISM_FIG_ROOT, RESULTS_MECHANISM_TRACE_EP10_ROOT, RESULTS_MECHANISM_TRACE_QUICK_ROOT


MAX_POINTS_PER_COMPONENT = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replot 2-task P/Q/C UMAP figure from paper mechanism traces.")
    parser.add_argument("--bundle", choices=["quick_seed7", "ep10_seed7"], default="ep10_seed7")
    return parser.parse_args()


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11.5,
        "axes.labelcolor": "#334155",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 1.0,
        "xtick.color": "#64748b",
        "ytick.color": "#64748b",
        "legend.fontsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfcfe",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def _load_trace(task: str, bundle: str) -> dict[str, np.ndarray]:
    traces_dir = (RESULTS_MECHANISM_TRACE_QUICK_ROOT if bundle == "quick_seed7" else RESULTS_MECHANISM_TRACE_EP10_ROOT) / "traces"
    payload = np.load(traces_dir / f"{task}_seed7_teacher_forced.npz", allow_pickle=True)
    return {
        "valid_mask": np.asarray(payload["valid_mask"]),
        "q": np.asarray(payload["q"]),
        "p": np.asarray(payload["p"]),
        "c": np.asarray(payload["c"]),
    }


def _project(array: np.ndarray) -> np.ndarray:
    try:
        from umap import UMAP

        return UMAP(n_neighbors=35, min_dist=0.12, random_state=0, n_components=2).fit_transform(array)
    except Exception:
        centered = array - np.mean(array, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:2].T
        return centered @ basis


def _pad(values: np.ndarray, target_dim: int) -> np.ndarray:
    if values.shape[1] >= target_dim:
        return values[:, :target_dim]
    return np.concatenate([values, np.zeros((values.shape[0], target_dim - values.shape[1]))], axis=1)


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=0, keepdims=True)
    std = np.std(values, axis=0, keepdims=True)
    return (values - mean) / np.clip(std, 1e-6, None)


def _prepare_component(values: np.ndarray, target_dim: int) -> np.ndarray:
    standardized = _standardize(values)
    padded = _pad(standardized, target_dim)
    return padded / np.sqrt(float(values.shape[1]))


def _subsample_even(values: np.ndarray, count: int) -> np.ndarray:
    if values.shape[0] <= count:
        return values
    indices = np.linspace(0, values.shape[0] - 1, count, dtype=np.int32)
    return values[indices]


def _draw_density_contours(ax: plt.Axes, points: np.ndarray, color: str) -> None:
    if points.shape[0] < 32:
        return
    x = points[:, 0]
    y = points[:, 1]
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except Exception:
        return

    x_lo, x_hi = np.quantile(x, [0.01, 0.99])
    y_lo, y_hi = np.quantile(y, [0.01, 0.99])
    x_pad = max(1e-3, 0.12 * (x_hi - x_lo))
    y_pad = max(1e-3, 0.12 * (y_hi - y_lo))
    xx, yy = np.meshgrid(
        np.linspace(x_lo - x_pad, x_hi + x_pad, 160),
        np.linspace(y_lo - y_pad, y_hi + y_pad, 160),
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    levels = np.quantile(zz[zz > 0], [0.72, 0.86, 0.95])
    ax.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=[0.9, 1.15, 1.45], alpha=0.7, zorder=3)


def _set_view_limits(ax: plt.Axes, embedding: np.ndarray) -> None:
    x_lo, x_hi = np.quantile(embedding[:, 0], [0.01, 0.99])
    y_lo, y_hi = np.quantile(embedding[:, 1], [0.01, 0.99])
    x_pad = max(1e-3, 0.10 * (x_hi - x_lo))
    y_pad = max(1e-3, 0.10 * (y_hi - y_lo))
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)


def _save_pair(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"))


def main() -> int:
    args = parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), sharex=False, sharey=False, dpi=220)
    colors = {"q": "#2b6cb0", "p": "#dd6b20", "c": "#2f855a"}
    titles = {"finger_spin": "Finger Spin", "cheetah_run": "Cheetah Run"}
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6.5, markerfacecolor=colors[key], markeredgecolor="white", markeredgewidth=0.6, label=fr"${key}$")
        for key in ("q", "p", "c")
    ]

    for ax, task in zip(axes, ["finger_spin", "cheetah_run"]):
        trace = _load_trace(task, args.bundle)
        mask = trace["valid_mask"].reshape(-1)
        q = trace["q"].reshape(-1, trace["q"].shape[-1])[mask]
        p = trace["p"].reshape(-1, trace["p"].shape[-1])[mask]
        c = trace["c"].reshape(-1, trace["c"].shape[-1])[mask]
        count = min(MAX_POINTS_PER_COMPONENT, q.shape[0], p.shape[0], c.shape[0])
        q = _subsample_even(q, count)
        p = _subsample_even(p, count)
        c = _subsample_even(c, count)
        target_dim = max(q.shape[1], p.shape[1], c.shape[1])
        stacked = np.concatenate(
            [
                _prepare_component(q, target_dim),
                _prepare_component(p, target_dim),
                _prepare_component(c, target_dim),
            ],
            axis=0,
        )
        labels = (["q"] * count) + (["p"] * count) + (["c"] * count)
        embedding = _project(stacked)
        for key in ("q", "p", "c"):
            selected = np.array([label == key for label in labels])
            points = embedding[selected]
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=4,
                alpha=0.20,
                color=colors[key],
                label=fr"${key}$",
                linewidths=0.0,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )
            _draw_density_contours(ax, points, colors[key])
        _set_view_limits(ax, embedding)
        ax.set_title(titles[task], loc="left", pad=8, color="#0f172a")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Embedding 1", labelpad=8)
        ax.set_ylabel("Embedding 2", labelpad=8)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color("#cbd5e1")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(legend_handles, ["q", "p", "c"], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02), handletextpad=0.35, columnspacing=1.2)
    fig.tight_layout(pad=0.7, w_pad=1.2, rect=(0, 0, 1, 0.92))
    out_stem = RESULTS_MECHANISM_FIG_ROOT / "pqc"
    _save_pair(fig, out_stem)
    plt.close(fig)
    print(out_stem.with_suffix(".pdf"))
    print(out_stem.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
