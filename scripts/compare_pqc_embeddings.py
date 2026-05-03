from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from umap import UMAP

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import ANALYSIS_FIG_ROOT, ANALYSIS_MECHANISM_TRACE_EP10_ROOT


TRACE_ROOT = ANALYSIS_MECHANISM_TRACE_EP10_ROOT / "traces"
OUT_STEM = ANALYSIS_FIG_ROOT / "pqc_embedding_method_sweep"
OUT_CSV = ANALYSIS_FIG_ROOT / "pqc_embedding_method_sweep_metrics.csv"
TASKS = ["finger_spin", "cheetah_run"]
TASK_LABELS = {"finger_spin": "Finger Spin", "cheetah_run": "Cheetah Run"}
COLORS = {"q": "#2b6cb0", "p": "#dd6b20", "c": "#2f855a"}
MAX_POINTS_PER_COMPONENT = 900


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10.5,
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


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    supervised: bool = False


METHODS = [
    MethodSpec("pca", "PCA"),
    MethodSpec("tsne", "t-SNE"),
    MethodSpec("umap_tight", "UMAP Tight"),
    MethodSpec("umap_balanced", "UMAP Balanced"),
    MethodSpec("sup_umap", "Supervised UMAP", supervised=True),
]


def _load_task_arrays(task: str) -> dict[str, np.ndarray]:
    payload = np.load(TRACE_ROOT / f"{task}_seed7_teacher_forced.npz", allow_pickle=True)
    valid = np.asarray(payload["valid_mask"], dtype=bool).reshape(-1)
    data = {}
    for key in ("q", "p", "c"):
        values = np.asarray(payload[key], dtype=np.float32)
        data[key] = values.reshape(-1, values.shape[-1])[valid]
    return data


def _subsample_even(values: np.ndarray, count: int) -> np.ndarray:
    if values.shape[0] <= count:
        return values
    indices = np.linspace(0, values.shape[0] - 1, count, dtype=np.int32)
    return values[indices]


def _pad(values: np.ndarray, target_dim: int) -> np.ndarray:
    if values.shape[1] >= target_dim:
        return values[:, :target_dim]
    return np.concatenate([values, np.zeros((values.shape[0], target_dim - values.shape[1]), dtype=values.dtype)], axis=1)


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = np.mean(values, axis=0, keepdims=True)
    std = np.std(values, axis=0, keepdims=True)
    return (values - mean) / np.clip(std, 1e-6, None)


def _prepare_task_dataset(task: str) -> tuple[np.ndarray, np.ndarray]:
    arrays = _load_task_arrays(task)
    count = min(MAX_POINTS_PER_COMPONENT, *(array.shape[0] for array in arrays.values()))
    sampled = {key: _subsample_even(array, count) for key, array in arrays.items()}
    target_dim = max(array.shape[1] for array in sampled.values())

    parts = []
    labels = []
    for label_idx, key in enumerate(("q", "p", "c")):
        values = _pad(_standardize(sampled[key]), target_dim) / np.sqrt(float(sampled[key].shape[1]))
        parts.append(values)
        labels.append(np.full(values.shape[0], label_idx, dtype=np.int32))
    return np.concatenate(parts, axis=0), np.concatenate(labels, axis=0)


def _embed(method: MethodSpec, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
    if method.key == "pca":
        return PCA(n_components=2, random_state=0).fit_transform(values)
    if method.key == "tsne":
        return TSNE(n_components=2, perplexity=35, init="pca", learning_rate="auto", random_state=0).fit_transform(values)
    if method.key == "umap_tight":
        return UMAP(n_neighbors=15, min_dist=0.03, random_state=0, n_components=2).fit_transform(values)
    if method.key == "umap_balanced":
        return UMAP(n_neighbors=35, min_dist=0.12, random_state=0, n_components=2).fit_transform(values)
    if method.key == "sup_umap":
        return UMAP(
            n_neighbors=30,
            min_dist=0.08,
            random_state=0,
            n_components=2,
            target_metric="categorical",
            target_weight=0.35,
        ).fit_transform(values, labels)
    raise ValueError(f"Unknown method: {method.key}")


def _write_metrics(rows: list[dict[str, str]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task", "method", "silhouette"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    fig, axes = plt.subplots(len(METHODS), len(TASKS), figsize=(9.4, 11.8), dpi=220)
    metrics_rows: list[dict[str, str]] = []

    for col_idx, task in enumerate(TASKS):
        values, labels = _prepare_task_dataset(task)
        key_labels = np.array(["q", "p", "c"])[labels]
        for row_idx, method in enumerate(METHODS):
            ax = axes[row_idx, col_idx]
            embedding = _embed(method, values, labels)
            silhouette = float(silhouette_score(embedding, labels))
            metrics_rows.append({"task": task, "method": method.key, "silhouette": f"{silhouette:.4f}"})

            for key in ("q", "p", "c"):
                selected = key_labels == key
                ax.scatter(
                    embedding[selected, 0],
                    embedding[selected, 1],
                    s=6,
                    alpha=0.56,
                    color=COLORS[key],
                    edgecolors="none",
                    rasterized=True,
                )

            title = f"{TASK_LABELS[task]} · {method.label}"
            if method.supervised:
                title += " *"
            ax.set_title(f"{title}\nSilhouette = {silhouette:.3f}", loc="left", pad=7, color="#0f172a")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("Embedding 1")
            ax.set_ylabel("Embedding 2")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#cbd5e1")
            ax.spines["bottom"].set_color("#cbd5e1")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=7, markerfacecolor=COLORS[key], markeredgecolor="none", label=key)
        for key in ("q", "p", "c")
    ]
    fig.legend(handles, ["q", "p", "c"], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("P/Q/C embedding method sweep on teacher-forced latent traces", fontsize=15, color="#0f172a", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985), pad=0.8, h_pad=1.0, w_pad=0.8)

    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_STEM.with_suffix(".png"))
    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    plt.close(fig)

    _write_metrics(metrics_rows)
    print(OUT_STEM.with_suffix(".png"))
    print(OUT_STEM.with_suffix(".pdf"))
    print(OUT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
