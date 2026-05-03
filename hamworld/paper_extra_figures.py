from __future__ import annotations

import os
import statistics
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = REPO_ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hamworld.export_paper_assets import MAIN_TABLE_TASK_LABELS, SeedRun


MECHANISM_GROUPS = [
    (
        "Physics / Consistency Losses",
        [
            ("train/roll_loss", "Rollout Loss", "#2563eb"),
            ("train/sa_loss", "Small-Action Loss", "#f59e0b"),
            ("train/energy_loss", "Energy Loss", "#16a34a"),
            ("train/hamiltonian_loss", "Hamiltonian Loss", "#dc2626"),
        ],
        True,
    ),
    (
        "Latent Norms",
        [
            ("train/q_norm", "q Norm", "#2563eb"),
            ("train/p_norm", "p Norm", "#f59e0b"),
            ("train/c_norm", "c Norm", "#16a34a"),
        ],
        False,
    ),
    (
        "Delta Norms",
        [
            ("train/delta_q_norm", "Delta q", "#2563eb"),
            ("train/delta_p_norm", "Delta p", "#f59e0b"),
            ("train/delta_c_norm", "Delta c", "#16a34a"),
        ],
        False,
    ),
    (
        "Control / Energy Signals",
        [
            ("train/control_norm", "Control Norm", "#2563eb"),
            ("train/energy_drift", "Energy Drift", "#16a34a"),
            ("train/hamiltonian_grad_norm", "Ham. Grad Norm", "#7c3aed"),
        ],
        False,
    ),
]


def _configure_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, linestyle="--", linewidth=0.8, color="#dbe2ea", alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#334155", labelsize=9)


def _downsample_series(
    steps: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    max_points: int = 900,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(steps) <= max_points:
        return steps, mean, std
    indices = np.linspace(0, len(steps) - 1, max_points, dtype=int)
    return steps[indices], mean[indices], std[indices]


def _aggregate_metric_series(task_runs: list[SeedRun], metric_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    step_to_values: dict[int, list[float]] = {}
    for run in task_runs:
        history = run.run_data.metric_history.get(metric_key, [])
        for step, value in history:
            step_to_values.setdefault(int(step), []).append(float(value))

    if not step_to_values:
        return np.asarray([]), np.asarray([]), np.asarray([])

    steps = []
    means = []
    stds = []
    for step in sorted(step_to_values):
        values = step_to_values[step]
        steps.append(step)
        means.append(statistics.fmean(values))
        stds.append(statistics.pstdev(values) if len(values) > 1 else 0.0)

    return _downsample_series(
        np.asarray(steps, dtype=np.float64),
        np.asarray(means, dtype=np.float64),
        np.asarray(stds, dtype=np.float64),
    )


def _task_title(task: str) -> str:
    return MAIN_TABLE_TASK_LABELS.get(task, task.replace("_", " ").title())


def _save_figure(fig: plt.Figure, output_path: Path, output_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220 if output_format == "png" else None, bbox_inches="tight")
    plt.close(fig)


def _render_task_mechanism_figure(task: str, task_runs: list[SeedRun], output_path: Path, output_format: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.8))
    axes_flat = axes.flatten()
    for ax, (panel_title, metrics, use_log) in zip(axes_flat, MECHANISM_GROUPS):
        for metric_key, label, color in metrics:
            steps, means, stds = _aggregate_metric_series(task_runs, metric_key)
            if len(steps) == 0:
                continue
            lower = means - stds
            upper = means + stds
            if use_log:
                lower = np.clip(lower, 1e-12, None)
                upper = np.clip(upper, 1e-12, None)
            ax.plot(steps, means, linewidth=2.1, color=color, label=label)
            ax.fill_between(steps, lower, upper, color=color, alpha=0.15, linewidth=0)
        _configure_axes(ax)
        if use_log:
            ax.set_yscale("log")
        ax.set_title(panel_title, fontsize=12, color="#0f172a")
        ax.set_xlabel("environment step", fontsize=10, color="#0f172a")
        ax.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle(f"HaM-World Mechanism Diagnostics on {_task_title(task)}", fontsize=17, color="#0f172a", y=0.98)
    fig.text(
        0.5,
        0.01,
        "Curves show seed-mean metrics across completed HaM-World runs; shaded region = mean ± std.",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    fig.subplots_adjust(hspace=0.3, wspace=0.24)
    _save_figure(fig, output_path, output_format)


def export_hamworld_mechanism_figures(
    seed_runs: list[SeedRun],
    by_task_root: Path,
    tasks: list[str],
    formats: list[str],
) -> list[Path]:
    pgm_runs = [run for run in seed_runs if run.algorithm == "hamworld"]
    exported: list[Path] = []
    for task in tasks:
        task_runs = [run for run in pgm_runs if run.task == task]
        if not task_runs:
            continue
        for fmt in formats:
            path = by_task_root / task / f"hamworld_mechanism.{fmt}"
            _render_task_mechanism_figure(task, task_runs, path, fmt)
            exported.append(path)
    return exported
