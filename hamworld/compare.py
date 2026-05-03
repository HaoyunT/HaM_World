from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MPLCONFIGDIR = REPO_ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ALGORITHM_STYLES = {
    "hamworld": {"label": "HaM-World", "color": "#2563eb"},
    "hamworld_old": {"label": "HaM-World (Old)", "color": "#94a3b8"},
    "hamworld_new": {"label": "HaM-World (New)", "color": "#2563eb"},
    "hamworld_cnorm": {"label": "HaM-World (c-norm)", "color": "#2563eb"},
    "jepa": {"label": "JEPA", "color": "#dc2626"},
    "td_jepa": {"label": "TD-JEPA", "color": "#7c3aed"},
    "td_jepa_taskwise": {"label": "TD-JEPA (Taskwise)", "color": "#7c3aed"},
    "td_jepa_pretrain_finetune": {"label": "TD-JEPA (P+F)", "color": "#8b5cf6"},
    "dreamerv3": {"label": "DreamerV3", "color": "#059669"},
    "tdmpc2": {"label": "TD-MPC2", "color": "#d97706"},
    "ppo": {"label": "PPO", "color": "#0891b2"},
    "sac": {"label": "SAC", "color": "#ea580c"},
    "ppo_low": {"label": "PPO", "color": "#0f766e"},
    "sac_low": {"label": "SAC", "color": "#c2410c"},
    "dreamerv3_rerun": {"label": "DreamerV3 (Rerun)", "color": "#10b981"},
}


VARIANT_COLORS = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#0891b2",
    "#be123c",
    "#65a30d",
    "#7c3aed",
    "#4b5563",
]

MAX_RENDER_POINTS = 1200


@dataclass
class RunData:
    algorithm_id: str
    algorithm: str
    color: str
    task: str
    run_dir: Path
    train_returns: list[tuple[int, float]]
    eval_returns: list[tuple[int, float]]
    metric_history: dict[str, list[tuple[int, float]]]
    source: str | None = None


METRIC_PLOT_ORDER = [
    "train/latent_mse_k1",
    "train/three_step_latent_mse",
    "train/latent_mse_k3",
    "train/latent_mse_k5",
    "train/latent_mse_k10",
    "train/latent_mse_k15",
    "train/rollout_latent_drift",
    "train/rollout_drift_k3",
    "train/rollout_drift_k5",
    "train/rollout_drift_k10",
    "train/rollout_drift_k15",
    "train/reward_pred_mae",
    "train/value_pred_mae",
    "train/energy_drift",
    "train/small_action_fraction",
    "train/energy_control_corr",
    "train/delta_q_norm",
    "train/delta_p_norm",
    "train/delta_c_norm",
    "train/planner_latency_ms",
    "train/control_norm",
    "train/one_step_latent_mse",
    "train/reward_pred_mse",
    "train/value_pred_mse",
    "train/hamiltonian_loss",
    "train/decouple_loss",
    "train/c_sparse_loss",
    "train/value_ce_loss",
    "train/value_slow_loss",
]

METRIC_LABELS = {
    "train/latent_mse_k1": "Latent MSE k=1",
    "train/one_step_latent_mse": "1-Step Latent MSE",
    "train/three_step_latent_mse": "3-Step Latent MSE",
    "train/latent_mse_k3": "Latent MSE k=3",
    "train/latent_mse_k5": "Latent MSE k=5",
    "train/latent_mse_k10": "Latent MSE k=10",
    "train/latent_mse_k15": "Latent MSE k=15",
    "train/rollout_latent_drift": "Rollout Drift",
    "train/rollout_drift_k3": "Rollout Drift k=3",
    "train/rollout_drift_k5": "Rollout Drift k=5",
    "train/rollout_drift_k10": "Rollout Drift k=10",
    "train/rollout_drift_k15": "Rollout Drift k=15",
    "train/reward_pred_mae": "Reward Pred MAE",
    "train/reward_pred_mse": "Reward Pred MSE",
    "train/value_pred_mae": "Value Pred MAE",
    "train/value_pred_mse": "Value Pred MSE",
    "train/energy_drift": "Energy Drift",
    "train/small_action_fraction": "Small-Action Fraction",
    "train/energy_control_corr": "Energy-Control Corr",
    "train/delta_q_norm": "Delta q Norm",
    "train/delta_p_norm": "Delta p Norm",
    "train/delta_c_norm": "Delta c Norm",
    "train/planner_latency_ms": "Planner Latency (ms)",
    "train/control_norm": "Control Norm",
    "train/hamiltonian_loss": "Hamiltonian Loss",
    "train/decouple_loss": "Q-P Decouple",
    "train/c_sparse_loss": "C Sparse",
    "train/value_ce_loss": "Value CE",
    "train/value_slow_loss": "Value Slow",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the latest world-model RL runs with pure-SVG output.")
    parser.add_argument("--outputs", default="outputs", type=str, help="Root outputs directory.")
    parser.add_argument("--task", default=None, type=str, help="Optional task id to compare.")
    parser.add_argument("--train-window", default=10, type=int, help="Smoothing window for train episode return.")
    parser.add_argument("--max-step", default=None, type=int, help="Optional max environment step to include in the plot.")
    parser.add_argument(
        "--experiment-root",
        action="append",
        default=[],
        nargs="+",
        help="Variant output root(s), e.g. outputs_qpc_select outputs_planner_ablation.",
    )
    parser.add_argument(
        "--compare-output",
        default=None,
        type=str,
        help="Output directory for experiment-root compare plots. Defaults to <outputs>/compare.",
    )
    parser.add_argument(
        "--run-selection",
        default="latest",
        choices=("latest", "max-step"),
        help="How to choose among multiple seed_* run directories.",
    )
    return parser.parse_args()


def _episode_max_step(run_dir: Path) -> int:
    episodes_rows = _load_jsonl(run_dir / "logs" / "episodes.jsonl")
    return max((int(row["step"]) for row in episodes_rows if "step" in row), default=-1)


def _select_run_dir(task_root: Path, run_selection: str = "latest") -> Path | None:
    candidates = [path for path in task_root.iterdir() if path.is_dir() and path.name.startswith("seed_")]
    if not candidates:
        return None
    if run_selection == "max-step":
        return sorted(candidates, key=lambda path: (_episode_max_step(path), path.name))[-1]
    return sorted(candidates, key=lambda path: path.name)[-1]


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _moving_average(points: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    if window <= 1 or len(points) <= 1:
        return points
    smoothed: list[tuple[int, float]] = []
    values: list[float] = []
    total = 0.0
    for step, value in points:
        values.append(value)
        total += value
        if len(values) > window:
            total -= values.pop(0)
        smoothed.append((step, total / len(values)))
    return smoothed


def _downsample_points(points: list[tuple[int, float]], max_points: int = MAX_RENDER_POINTS) -> list[tuple[int, float]]:
    if len(points) <= max_points:
        return points
    if max_points <= 2:
        return [points[0], points[-1]]
    last_index = len(points) - 1
    indices = [0]
    for slot in range(1, max_points - 1):
        indices.append(round(slot * last_index / (max_points - 1)))
    indices.append(last_index)
    deduped = sorted(set(indices))
    return [points[index] for index in deduped]


def _load_episode_returns(
    rows: list[dict],
    split: str,
    max_step: int | None = None,
    average_by_step: bool = False,
) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for row in rows:
        if row.get("split") != split:
            continue
        step = int(row["step"])
        if max_step is not None and step > max_step:
            continue
        points.append((step, float(row["return"])))

    if not average_by_step:
        return points

    grouped: dict[int, list[float]] = {}
    for step, value in points:
        grouped.setdefault(step, []).append(value)
    return [(step, sum(values) / len(values)) for step, values in sorted(grouped.items())]


def _load_metric_history(
    rows: list[dict],
    train_window: int,
    max_step: int | None = None,
) -> dict[str, list[tuple[int, float]]]:
    history: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        if "step" not in row:
            continue
        step = int(row["step"])
        if max_step is not None and step > max_step:
            continue
        for key, value in row.items():
            if key == "step":
                continue
            history.setdefault(key, []).append((step, float(value)))

    for key, values in list(history.items()):
        if key.startswith("train/"):
            history[key] = _moving_average(values, train_window)
    return history


def _load_run_dir(
    run_dir: Path,
    algorithm_id: str,
    label: str,
    color: str,
    task: str,
    train_window: int,
    max_step: int | None = None,
    source: str | None = None,
) -> RunData | None:
    episodes_rows = _load_jsonl(run_dir / "logs" / "episodes.jsonl")
    metrics_rows = _load_jsonl(run_dir / "logs" / "metrics.jsonl")
    train_returns = _load_episode_returns(episodes_rows, "train", max_step=max_step)
    train_returns = _moving_average(train_returns, train_window)
    eval_returns = _load_episode_returns(episodes_rows, "eval", max_step=max_step, average_by_step=True)
    metric_history = _load_metric_history(metrics_rows, train_window, max_step=max_step)
    if not train_returns and not eval_returns and not metric_history:
        return None

    return RunData(
        algorithm_id=algorithm_id,
        algorithm=label,
        color=color,
        task=task,
        run_dir=run_dir,
        train_returns=train_returns,
        eval_returns=eval_returns,
        metric_history=metric_history,
        source=source,
    )


def _load_run(
    outputs_root: Path,
    algorithm_id: str,
    task: str,
    train_window: int,
    max_step: int | None = None,
    run_selection: str = "latest",
) -> RunData | None:
    algorithm_root = outputs_root / algorithm_id
    task_root = algorithm_root / task
    if not task_root.exists():
        return None
    run_dir = _select_run_dir(task_root, run_selection)
    if run_dir is None:
        return None

    style = ALGORITHM_STYLES.get(algorithm_id, {"label": algorithm_id, "color": "#4b5563"})
    return _load_run_dir(
        run_dir=run_dir,
        algorithm_id=algorithm_id,
        label=style["label"],
        color=style["color"],
        task=task,
        train_window=train_window,
        max_step=max_step,
    )


def _scale_points(
    points: list[tuple[int, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: float,
    height: float,
    left: float,
    top: float,
) -> str:
    if not points:
        return ""
    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)
    coords = []
    for x, y in points:
        px = left + (x - x_min) / x_span * width
        py = top + height - (y - y_min) / y_span * height
        coords.append(f"{px:.2f},{py:.2f}")
    return " ".join(coords)


def _escape_text(text: str) -> str:
    return html.escape(text, quote=False)


def _summary_text(run: RunData) -> str:
    final_train = run.train_returns[-1][1] if run.train_returns else float("nan")
    best_train = max((value for _, value in run.train_returns), default=float("nan"))
    final_eval = run.eval_returns[-1][1] if run.eval_returns else float("nan")
    best_eval = max((value for _, value in run.eval_returns), default=float("nan"))
    source = f"  root={run.source.split('/')[0]}" if run.source else ""
    return (
        f"{run.algorithm}{source}  run={run.run_dir.name}  "
        f"final_train={final_train:.3f}  best_train={best_train:.3f}  "
        f"final_eval={final_eval:.3f}  best_eval={best_eval:.3f}"
    )


def _legend_height(runs: list[RunData]) -> float:
    return 22 * max(1, math.ceil(len(runs) / 4))


def _legend(runs: list[RunData], left: float, top: float) -> str:
    parts: list[str] = []
    col_width = 285
    for index, run in enumerate(runs):
        row = index // 4
        col = index % 4
        x = left + col * col_width
        y = top + row * 22
        parts.append(f'<rect x="{x}" y="{y - 12}" width="16" height="16" rx="4" fill="{run.color}"/>')
        parts.append(f'<text x="{x + 24}" y="{y + 1}" font-size="14" fill="#1f2937">{_escape_text(run.algorithm)}</text>')
    return "".join(parts)


def _panel_svg(
    title: str,
    runs: list[RunData],
    series_name: str,
    x_label: str,
    y_label: str,
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    all_points = [point for run in runs for point in getattr(run, series_name)]
    if not all_points:
        return f'<text x="{left}" y="{top + 20}" font-size="16">{title}: no data</text>'

    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if math.isclose(y_min, y_max):
        margin = 1.0 if y_min == 0 else abs(y_min) * 0.1
        y_min -= margin
        y_max += margin
    else:
        pad = (y_max - y_min) * 0.1
        y_min -= pad
        y_max += pad

    plot_left = left + 60
    plot_top = top + 35
    plot_width = width - 90
    plot_height = height - 80

    y_ticks = []
    for index in range(5):
        frac = index / 4.0
        value = y_max - frac * (y_max - y_min)
        y = plot_top + frac * plot_height
        y_ticks.append(
            f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1"/>'
            f'<text x="{plot_left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" fill="#4b5563">{value:.2f}</text>'
        )

    x_ticks = []
    for index in range(5):
        frac = index / 4.0
        value = x_min + frac * (x_max - x_min)
        x = plot_left + frac * plot_width
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_top + plot_height}" stroke="#f3f4f6" stroke-width="1"/>'
            f'<text x="{x:.2f}" y="{plot_top + plot_height + 18}" text-anchor="middle" font-size="11" fill="#4b5563">{int(value)}</text>'
        )

    polylines = []
    markers = []
    for run in runs:
        points = _downsample_points(getattr(run, series_name))
        poly = _scale_points(points, x_min, x_max, y_min, y_max, plot_width, plot_height, plot_left, plot_top)
        if poly:
            last_x, last_y = points[-1]
            marker_x = plot_left + (last_x - x_min) / max(1e-6, x_max - x_min) * plot_width
            marker_y = plot_top + plot_height - (last_y - y_min) / max(1e-6, y_max - y_min) * plot_height
            polylines.append(
                f'<polyline fill="none" stroke="{run.color}" stroke-width="3.2" '
                f'stroke-linecap="round" stroke-linejoin="round" points="{poly}"/>'
            )
            markers.append(f'<circle cx="{marker_x:.2f}" cy="{marker_y:.2f}" r="4.5" fill="{run.color}" stroke="#ffffff" stroke-width="1.5"/>')

    panel = [
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="16" fill="#ffffff" stroke="#d7dee8"/>',
        f'<text x="{left + 18}" y="{top + 24}" font-size="18" font-weight="700" fill="#111827">{_escape_text(title)}</text>',
        *y_ticks,
        *x_ticks,
        f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" stroke="#6b7280" stroke-width="1.5"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" stroke="#6b7280" stroke-width="1.5"/>',
        *polylines,
        *markers,
        f'<text x="{plot_left + plot_width / 2:.2f}" y="{top + height - 10}" text-anchor="middle" font-size="12" fill="#374151">{_escape_text(x_label)}</text>',
        f'<text x="{left + 15}" y="{plot_top + plot_height / 2:.2f}" text-anchor="middle" font-size="12" fill="#374151" transform="rotate(-90 {left + 15},{plot_top + plot_height / 2:.2f})">{_escape_text(y_label)}</text>',
    ]
    return "".join(panel)


def _summary_lines(runs: list[RunData], start_y: float) -> str:
    lines = []
    for index, run in enumerate(runs):
        y = start_y + index * 16
        text = _summary_text(run)
        lines.append(f'<text x="40" y="{y}" font-size="13" fill="#4b5563">{_escape_text(text)}</text>')
    return "".join(lines)


def _metric_panel_svg(
    title: str,
    metric_key: str,
    runs: list[RunData],
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    plot_runs = [run for run in runs if run.metric_history.get(metric_key)]
    if len(plot_runs) < 1:
        return ""
    all_points = [point for run in plot_runs for point in run.metric_history[metric_key]]
    if not all_points:
        return ""

    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if math.isclose(y_min, y_max):
        margin = 1.0 if y_min == 0 else abs(y_min) * 0.1
        y_min -= margin
        y_max += margin
    else:
        pad = (y_max - y_min) * 0.12
        y_min -= pad
        y_max += pad

    plot_left = left + 56
    plot_top = top + 34
    plot_width = width - 80
    plot_height = height - 66

    y_ticks = []
    for index in range(4):
        frac = index / 3.0
        value = y_max - frac * (y_max - y_min)
        y = plot_top + frac * plot_height
        y_ticks.append(
            f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_left + plot_width}" y2="{y:.2f}" stroke="#edf2f7" stroke-width="1"/>'
            f'<text x="{plot_left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="10.5" fill="#64748b">{value:.2f}</text>'
        )

    x_ticks = []
    for index in range(4):
        frac = index / 3.0
        value = x_min + frac * (x_max - x_min)
        x = plot_left + frac * plot_width
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_top + plot_height}" stroke="#f8fafc" stroke-width="1"/>'
            f'<text x="{x:.2f}" y="{plot_top + plot_height + 16}" text-anchor="middle" font-size="10.5" fill="#64748b">{int(value)}</text>'
        )

    polylines = []
    markers = []
    for run in plot_runs:
        points = _downsample_points(run.metric_history[metric_key])
        poly = _scale_points(points, x_min, x_max, y_min, y_max, plot_width, plot_height, plot_left, plot_top)
        if poly:
            last_x, last_y = points[-1]
            marker_x = plot_left + (last_x - x_min) / max(1e-6, x_max - x_min) * plot_width
            marker_y = plot_top + plot_height - (last_y - y_min) / max(1e-6, y_max - y_min) * plot_height
            polylines.append(
                f'<polyline fill="none" stroke="{run.color}" stroke-width="2.8" '
                f'stroke-linecap="round" stroke-linejoin="round" points="{poly}"/>'
            )
            markers.append(f'<circle cx="{marker_x:.2f}" cy="{marker_y:.2f}" r="3.8" fill="{run.color}" stroke="#ffffff" stroke-width="1.2"/>')

    return "".join(
        [
            f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="16" fill="#ffffff" stroke="#d7dee8"/>',
            f'<text x="{left + 16}" y="{top + 22}" font-size="16" font-weight="700" fill="#111827">{_escape_text(title)}</text>',
            *y_ticks,
            *x_ticks,
            f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" stroke="#94a3b8" stroke-width="1.2"/>',
            f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" stroke="#94a3b8" stroke-width="1.2"/>',
            *polylines,
            *markers,
        ]
    )


def render_series_compare_svg(
    task: str,
    runs: list[RunData],
    output_path: Path,
    series_name: str,
    panel_title: str,
    y_label: str,
    max_step: int | None = None,
    title: str | None = None,
    note: str | None = None,
) -> Path | None:
    plot_runs = [run for run in runs if getattr(run, series_name)]
    if not plot_runs:
        return None

    width = 1280
    header_y = 42
    legend_top = 74
    note_y = legend_top + _legend_height(runs) + 8
    summary_start_y = note_y + 18
    header_height = summary_start_y + 16 * len(runs) + 18
    panel_top = header_height

    panel = _panel_svg(
        title=panel_title,
        runs=plot_runs,
        series_name=series_name,
        x_label="environment step",
        y_label=y_label,
        left=40,
        top=panel_top,
        width=1200,
        height=340,
    )
    height = panel_top + 340 + 50

    step_note = note or (f"visualized up to step {max_step}" if max_step is not None else "latest matched runs")
    title_text = title or f"World Model Compare: {task}"
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f4f7fb"/>
<text x="40" y="{header_y}" font-size="28" font-weight="800" fill="#111827">{_escape_text(title_text)}</text>
{_legend(runs, 42, legend_top)}
<text x="40" y="{note_y}" font-size="13" fill="#4b5563">{_escape_text(step_note)}</text>
{_summary_lines(runs, summary_start_y)}
{panel}
</svg>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def _styled_legend_handles(runs: list[RunData]) -> list[Line2D]:
    return [
        Line2D([0], [0], color=run.color, lw=2.4, marker="o", markersize=5.5, label=run.algorithm)
        for run in runs
    ]


def _style_plot_axis(ax: plt.Axes, y_label: str) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(True, color="#e5e7eb", linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#475569", labelsize=10)
    ax.set_xlabel("environment step", fontsize=11, color="#334155")
    ax.set_ylabel(y_label, fontsize=11, color="#334155")


def render_series_compare_png(
    task: str,
    runs: list[RunData],
    output_path: Path,
    series_name: str,
    panel_title: str,
    y_label: str,
    max_step: int | None = None,
    title: str | None = None,
    note: str | None = None,
) -> Path | None:
    del task
    plot_runs = [run for run in runs if getattr(run, series_name)]
    if not plot_runs:
        return None

    summary_lines = [_summary_text(run) for run in runs]
    fig_height = 5.8 + 0.22 * max(1, len(summary_lines))
    fig, ax = plt.subplots(figsize=(12.8, fig_height), dpi=170)
    fig.patch.set_facecolor("#f4f7fb")

    for run in plot_runs:
        points = _downsample_points(getattr(run, series_name))
        xs = [step for step, _ in points]
        ys = [value for _, value in points]
        ax.plot(xs, ys, color=run.color, linewidth=2.4, solid_capstyle="round")
        ax.scatter([xs[-1]], [ys[-1]], s=36, color=run.color, edgecolors="#ffffff", linewidths=1.2, zorder=3)

    _style_plot_axis(ax, y_label)
    ax.set_title(panel_title, loc="left", fontsize=15, fontweight="bold", color="#111827", pad=12)

    title_text = title or "World Model Compare"
    step_note = note or (f"visualized up to step {max_step}" if max_step is not None else "latest matched runs")
    plot_top = max(0.53, 0.84 - 0.035 * len(summary_lines))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=plot_top)
    fig.suptitle(title_text, x=0.08, y=0.985, ha="left", fontsize=21, fontweight="bold", color="#111827")
    fig.legend(
        handles=_styled_legend_handles(runs),
        loc="upper left",
        bbox_to_anchor=(0.08, 0.93),
        ncol=min(4, max(1, len(runs))),
        frameon=False,
        fontsize=10,
    )
    fig.text(0.08, 0.885, step_note, fontsize=10.5, color="#475569", ha="left", va="top")
    for index, text in enumerate(summary_lines):
        y = 0.855 - index * 0.028
        fig.text(0.08, y, text, fontsize=8.2, color="#64748b", family="monospace", ha="left", va="top")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_metric_compare_svg(
    task: str,
    runs: list[RunData],
    output_path: Path,
    max_step: int | None = None,
    title: str | None = None,
    note: str | None = None,
) -> Path | None:
    metric_keys = [
        key
        for key in METRIC_PLOT_ORDER
        if any(run.metric_history.get(key) for run in runs)
    ]
    if not metric_keys:
        return None

    width = 1280
    cols = 2
    panel_width = 580
    panel_height = 240
    rows = math.ceil(len(metric_keys) / cols)
    title_y = 40
    legend_top = 68
    note_y = legend_top + _legend_height(runs) + 8
    top_offset = note_y + 26
    height = top_offset + rows * (panel_height + 24) + 24

    panels: list[str] = []
    for index, metric_key in enumerate(metric_keys):
        row = index // cols
        col = index % cols
        left = 40 + col * 610
        top = top_offset + row * (panel_height + 24)
        label = METRIC_LABELS.get(metric_key, metric_key.replace("train/", "").replace("_", " "))
        panel = _metric_panel_svg(label, metric_key, runs, left, top, panel_width, panel_height)
        if panel:
            panels.append(panel)

    if not panels:
        return None

    title_text = title or f"World Model Metrics Compare: {task}"
    step_note = note or (f"visualized up to step {max_step}" if max_step is not None else "latest matched runs")
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f4f7fb"/>
<text x="40" y="{title_y}" font-size="28" font-weight="800" fill="#111827">{_escape_text(title_text)}</text>
{_legend(runs, 42, legend_top)}
<text x="40" y="{note_y}" font-size="13" fill="#4b5563">{_escape_text(step_note)}</text>
{"".join(panels)}
</svg>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def render_metric_compare_png(
    task: str,
    runs: list[RunData],
    output_path: Path,
    max_step: int | None = None,
    title: str | None = None,
    note: str | None = None,
) -> Path | None:
    del task
    metric_keys = [
        key
        for key in METRIC_PLOT_ORDER
        if any(run.metric_history.get(key) for run in runs)
    ]
    if not metric_keys:
        return None

    cols = 2
    rows = math.ceil(len(metric_keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13.0, 2.9 * rows + 1.8), dpi=170)
    fig.patch.set_facecolor("#f4f7fb")
    axes_list = axes.flat if hasattr(axes, "flat") else [axes]

    for axis_index, ax in enumerate(axes_list):
        if axis_index >= len(metric_keys):
            ax.axis("off")
            continue
        metric_key = metric_keys[axis_index]
        label = METRIC_LABELS.get(metric_key, metric_key.replace("train/", "").replace("_", " "))
        plotted = False
        for run in runs:
            if not run.metric_history.get(metric_key):
                continue
            points = _downsample_points(run.metric_history[metric_key])
            xs = [step for step, _ in points]
            ys = [value for _, value in points]
            ax.plot(xs, ys, color=run.color, linewidth=2.0, solid_capstyle="round")
            ax.scatter([xs[-1]], [ys[-1]], s=28, color=run.color, edgecolors="#ffffff", linewidths=1.0, zorder=3)
            plotted = True
        if not plotted:
            ax.axis("off")
            continue
        _style_plot_axis(ax, "")
        ax.set_title(label, loc="left", fontsize=13, fontweight="bold", color="#111827", pad=8)

    title_text = title or "World Model Metrics Compare"
    step_note = note or (f"visualized up to step {max_step}" if max_step is not None else "latest matched runs")
    fig.suptitle(title_text, x=0.06, y=0.985, ha="left", fontsize=21, fontweight="bold", color="#111827")
    fig.legend(
        handles=_styled_legend_handles(runs),
        loc="upper left",
        bbox_to_anchor=(0.06, 0.94),
        ncol=min(4, max(1, len(runs))),
        frameon=False,
        fontsize=10,
    )
    fig.text(0.06, 0.89, step_note, fontsize=10.5, color="#475569", ha="left", va="top")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.06, top=0.84, hspace=0.34, wspace=0.16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_series_compare(
    task: str,
    runs: list[RunData],
    output_path: Path,
    series_name: str,
    panel_title: str,
    y_label: str,
    output_format: str = "svg",
    max_step: int | None = None,
    title: str | None = None,
    note: str | None = None,
) -> Path | None:
    if output_format == "png":
        return render_series_compare_png(
            task,
            runs,
            output_path,
            series_name=series_name,
            panel_title=panel_title,
            y_label=y_label,
            max_step=max_step,
            title=title,
            note=note,
        )
    return render_series_compare_svg(
        task,
        runs,
        output_path,
        series_name=series_name,
        panel_title=panel_title,
        y_label=y_label,
        max_step=max_step,
        title=title,
        note=note,
    )


def render_metric_compare(
    task: str,
    runs: list[RunData],
    output_path: Path,
    output_format: str = "svg",
    max_step: int | None = None,
    title: str | None = None,
    note: str | None = None,
) -> Path | None:
    if output_format == "png":
        return render_metric_compare_png(
            task,
            runs,
            output_path,
            max_step=max_step,
            title=title,
            note=note,
        )
    return render_metric_compare_svg(
        task,
        runs,
        output_path,
        max_step=max_step,
        title=title,
        note=note,
    )


def compare_task(
    outputs_root: Path,
    task: str,
    train_window: int,
    max_step: int | None = None,
    run_selection: str = "latest",
    compare_output_root: Path | None = None,
) -> Path | None:
    runs = []
    for algorithm_id in ALGORITHM_STYLES:
        run = _load_run(outputs_root, algorithm_id, task, train_window, max_step=max_step, run_selection=run_selection)
        if run is not None:
            runs.append(run)

    if len(runs) < 2:
        return None

    output_dir = (compare_output_root if compare_output_root is not None else outputs_root / "compare") / task
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"world_model_compare_smoothed_{timestamp}.svg"
    render_series_compare_svg(
        task,
        runs,
        output_path,
        series_name="train_returns",
        panel_title="Train Episode Return (Smoothed)",
        y_label="train episode return",
        max_step=max_step,
    )
    render_series_compare_svg(
        task,
        runs,
        output_dir / f"world_model_compare_eval_{timestamp}.svg",
        series_name="eval_returns",
        panel_title="Eval Episode Return (Mean)",
        y_label="mean eval episode return",
        max_step=max_step,
        title=f"World Model Eval Compare: {task}",
    )
    render_metric_compare_svg(
        task,
        runs,
        output_dir / f"world_model_compare_metrics_{timestamp}.svg",
        max_step=max_step,
    )
    return output_path


def _flatten_experiment_roots(experiment_roots: list[list[str]]) -> list[Path]:
    roots: list[Path] = []
    for group in experiment_roots:
        for path in group:
            resolved = Path(path).resolve()
            if resolved.exists():
                roots.append(resolved)
            else:
                print(f"Skipping missing experiment root: {resolved}")
    return roots


def _format_variant_label(variant_id: str) -> str:
    label = re.sub(r"q(\d+)_p(\d+)_c(\d+)", r"q\1/p\2/c\3", variant_id)
    label = re.sub(r"_pt(\d+)", r" pt\1", label)
    label = label.replace("_no_", " no ")
    return label.replace("_", " ")


def _iter_variant_task_roots(experiment_root: Path) -> list[tuple[str, str, str, Path]]:
    task_roots: list[tuple[str, str, str, Path]] = []
    for variant_root in sorted((path for path in experiment_root.iterdir() if path.is_dir()), key=lambda path: path.name):
        if variant_root.name == "compare":
            continue
        for algorithm_root in sorted((path for path in variant_root.iterdir() if path.is_dir()), key=lambda path: path.name):
            for task_root in sorted((path for path in algorithm_root.iterdir() if path.is_dir()), key=lambda path: path.name):
                if any(path.is_dir() and path.name.startswith("seed_") for path in task_root.iterdir()):
                    task_roots.append((variant_root.name, algorithm_root.name, task_root.name, task_root))
    return task_roots


def _experiment_task_ids(experiment_roots: list[Path]) -> list[str]:
    tasks = set()
    for experiment_root in experiment_roots:
        for _, _, task, _ in _iter_variant_task_roots(experiment_root):
            tasks.add(task)
    return sorted(tasks)


def _load_experiment_runs(
    experiment_roots: list[Path],
    task: str,
    train_window: int,
    max_step: int | None,
    run_selection: str,
) -> list[RunData]:
    runs: list[RunData] = []
    for experiment_root in experiment_roots:
        for variant_id, algorithm_id, task_id, task_root in _iter_variant_task_roots(experiment_root):
            if task_id != task:
                continue
            run_dir = _select_run_dir(task_root, run_selection)
            if run_dir is None:
                continue
            color = VARIANT_COLORS[len(runs) % len(VARIANT_COLORS)]
            label = _format_variant_label(variant_id)
            if algorithm_id != "hamworld":
                label = f"{label} {algorithm_id}"
            source = f"{experiment_root.name}/{variant_id}/{algorithm_id}"
            run = _load_run_dir(
                run_dir=run_dir,
                algorithm_id=f"{experiment_root.name}:{variant_id}:{algorithm_id}",
                label=label,
                color=color,
                task=task,
                train_window=train_window,
                max_step=max_step,
                source=source,
            )
            if run is not None:
                runs.append(run)
    return runs


def compare_experiment_task(
    experiment_roots: list[Path],
    compare_output_root: Path,
    task: str,
    train_window: int,
    max_step: int | None = None,
    run_selection: str = "latest",
) -> Path | None:
    runs = _load_experiment_runs(experiment_roots, task, train_window, max_step, run_selection)
    if len(runs) < 2:
        return None

    output_dir = compare_output_root / "experiments" / task
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"hamworld_experiment_compare_smoothed_{timestamp}.svg"
    run_note = "max-step run per variant" if run_selection == "max-step" else "latest run per variant"
    step_note = f"{run_note}; visualized up to step {max_step}" if max_step is not None else run_note
    render_series_compare_svg(
        task,
        runs,
        output_path,
        series_name="train_returns",
        panel_title="Train Episode Return (Smoothed)",
        y_label="train episode return",
        max_step=max_step,
        title=f"HaM-World Experiment Compare: {task}",
        note=step_note,
    )
    render_series_compare_svg(
        task,
        runs,
        output_dir / f"hamworld_experiment_compare_eval_{timestamp}.svg",
        series_name="eval_returns",
        panel_title="Eval Episode Return (Mean)",
        y_label="mean eval episode return",
        max_step=max_step,
        title=f"HaM-World Eval Compare: {task}",
        note=step_note,
    )
    render_metric_compare_svg(
        task,
        runs,
        output_dir / f"hamworld_experiment_compare_metrics_{timestamp}.svg",
        max_step=max_step,
        title=f"HaM-World Metric Compare: {task}",
        note=step_note,
    )
    return output_path


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs).resolve()
    compare_output_root = Path(args.compare_output).resolve() if args.compare_output else outputs_root / "compare"
    experiment_roots = _flatten_experiment_roots(args.experiment_root)
    if experiment_roots:
        experiment_tasks = _experiment_task_ids(experiment_roots)
        if args.task:
            experiment_tasks = [task for task in experiment_tasks if task == args.task]

        if not experiment_tasks:
            print("No variant experiment tasks were found under the provided experiment roots.")
            return

        generated = []
        skipped = []
        for task in experiment_tasks:
            output_path = compare_experiment_task(
                experiment_roots,
                compare_output_root,
                task,
                args.train_window,
                max_step=args.max_step,
                run_selection=args.run_selection,
            )
            if output_path is not None:
                generated.append(output_path)
            else:
                skipped.append(task)

        if not generated:
            print("Variant experiment tasks were found, but fewer than two comparable runs were available per task.")
            return

        print("Generated experiment compare plots:")
        for path in generated:
            print(path)
        if skipped:
            print("Skipped experiment tasks with fewer than two runs:")
            for task in skipped:
                print(task)
        return

    task_sets = []
    for algorithm_id in ALGORITHM_STYLES:
        algorithm_root = outputs_root / algorithm_id
        if algorithm_root.exists():
            task_sets.append({path.name for path in algorithm_root.iterdir() if path.is_dir()})
    common_tasks = sorted(set.union(*task_sets)) if task_sets else []
    if args.task:
        common_tasks = [task for task in common_tasks if task == args.task]

    if not common_tasks:
        print("No comparable tasks were found across the supported algorithm output folders.")
        return

    generated = []
    for task in common_tasks:
        output_path = compare_task(
            outputs_root,
            task,
            args.train_window,
            max_step=args.max_step,
            run_selection=args.run_selection,
            compare_output_root=compare_output_root,
        )
        if output_path is not None:
            generated.append(output_path)

    if not generated:
        print("Tasks were found, but fewer than two comparable runs were available per task.")
        return

    print("Generated compare plots:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
