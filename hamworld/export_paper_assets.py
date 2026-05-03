from __future__ import annotations

import csv
import html
import json
import math
import numpy as np
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = REPO_ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hamworld import compare as cmp


DEFAULT_ALGOS = ["hamworld", "tdmpc2", "dreamerv3", "jepa"]
DEFAULT_TASKS = ["reacher_easy", "finger_spin", "cheetah_run", "cartpole_swingup"]
MAIN_TABLE_TASK_LABELS = {
    "reacher_easy": "Reacher",
    "finger_spin": "Finger",
    "cheetah_run": "Cheetah",
    "cartpole_swingup": "Cartpole",
}
WM_KEYS = [
    ("train/one_step_latent_mse", "1-step MSE"),
    ("train/three_step_latent_mse", "3-step MSE"),
    ("train/rollout_latent_drift", "Rollout Drift"),
    ("train/reward_pred_mae", "Reward Err."),
    ("train/value_pred_mae", "Value Err."),
]


@dataclass
class SeedRun:
    algorithm: str
    task: str
    seed: int
    run_dir: Path
    run_data: cmp.RunData
    final_eval: float
    best_eval: float
    final_step: int
    auc: float


@dataclass
class AggregateSeries:
    algorithm: str
    color: str
    mean_points: list[tuple[int, float]]
    std_low_points: list[tuple[int, float]]
    std_high_points: list[tuple[int, float]]
    p05_points: list[tuple[int, float]]
    p95_points: list[tuple[int, float]]


def parse_seed(run_name: str) -> int:
    match = re.search(r"seed_(\d+)_", run_name)
    if not match:
        raise ValueError(f"Unable to parse seed from run directory '{run_name}'.")
    return int(match.group(1))


def load_run_metrics(metrics_path: Path) -> tuple[list[tuple[int, float]], dict[str, list[tuple[int, float]]]]:
    eval_returns: list[tuple[int, float]] = []
    metric_history: dict[str, list[tuple[int, float]]] = {}
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            step = int(row["step"])
            if "eval/return_mean" in row:
                eval_returns.append((step, float(row["eval/return_mean"])))
            else:
                for key, value in row.items():
                    if key == "step":
                        continue
                    metric_history.setdefault(key, []).append((step, float(value)))
    return eval_returns, metric_history


def trapezoid_auc(points: list[tuple[int, float]]) -> float:
    if len(points) < 2:
        return float("nan")
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
        area += (x2 - x1) * (y1 + y2) * 0.5
    span = points[-1][0] - points[0][0]
    return area / span if span > 0 else float("nan")


def select_seed_runs(
    runs_root: Path,
    algorithms: list[str],
    tasks: list[str],
    train_window: int,
    min_step: int,
    max_step: int | None = None,
) -> list[SeedRun]:
    selected: list[SeedRun] = []
    for algorithm in algorithms:
        for task in tasks:
            task_root = runs_root / algorithm / task
            if not task_root.exists():
                continue
            best_per_seed: dict[int, tuple[int, str, Path]] = {}
            for run_dir in sorted(path for path in task_root.iterdir() if path.is_dir() and path.name.startswith("seed_")):
                metrics_path = run_dir / "logs" / "metrics.jsonl"
                if not metrics_path.exists():
                    continue
                eval_returns, _ = load_run_metrics(metrics_path)
                if max_step is not None:
                    eval_returns = [(step, value) for step, value in eval_returns if step <= max_step]
                if not eval_returns:
                    continue
                final_step = int(eval_returns[-1][0])
                if final_step < min_step:
                    continue
                seed = parse_seed(run_dir.name)
                key = (final_step, run_dir.name)
                if seed not in best_per_seed or key > (best_per_seed[seed][0], best_per_seed[seed][1]):
                    best_per_seed[seed] = (final_step, run_dir.name, run_dir)

            for seed in sorted(best_per_seed):
                _, _, run_dir = best_per_seed[seed]
                run_data = cmp._load_run_dir(
                    run_dir=run_dir,
                    algorithm_id=algorithm,
                    label=cmp.ALGORITHM_STYLES.get(algorithm, {"label": algorithm})["label"],
                    color=cmp.ALGORITHM_STYLES.get(algorithm, {"color": "#4b5563"})["color"],
                    task=task,
                    train_window=train_window,
                    max_step=max_step,
                )
                if run_data is None or not run_data.eval_returns:
                    continue
                final_eval = run_data.eval_returns[-1][1]
                best_eval = max(value for _, value in run_data.eval_returns)
                auc = trapezoid_auc(run_data.eval_returns)
                selected.append(
                    SeedRun(
                        algorithm=algorithm,
                        task=task,
                        seed=seed,
                        run_dir=run_dir,
                        run_data=run_data,
                        final_eval=final_eval,
                        best_eval=best_eval,
                        final_step=run_data.eval_returns[-1][0],
                        auc=auc,
                    )
                )
    return selected


def mean_std_text(values: list[float]) -> str:
    if not values:
        return "--"
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_main_results_table(seed_runs: list[SeedRun], summary_root: Path, tasks: list[str], algorithms: list[str]) -> None:
    csv_rows: list[dict[str, object]] = []
    for algorithm in algorithms:
        row: dict[str, object] = {"algorithm": algorithm}
        auc_values: list[float] = []
        for task in tasks:
            values = [run.final_eval for run in seed_runs if run.algorithm == algorithm and run.task == task]
            auc_task = [run.auc for run in seed_runs if run.algorithm == algorithm and run.task == task and not math.isnan(run.auc)]
            row[f"{task}_final_mean"] = statistics.fmean(values) if values else ""
            row[f"{task}_final_std"] = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else "")
            row[f"{task}_n"] = len(values)
            if auc_task:
                auc_values.extend(auc_task)
        row["auc_mean"] = statistics.fmean(auc_values) if auc_values else ""
        row["auc_std"] = statistics.pstdev(auc_values) if len(auc_values) > 1 else (0.0 if auc_values else "")
        csv_rows.append(row)

    fieldnames = ["algorithm"] + [f"{task}_{suffix}" for task in tasks for suffix in ("final_mean", "final_std", "n")] + ["auc_mean", "auc_std"]
    write_csv(summary_root / "paper_main_results.csv", csv_rows, fieldnames)

    tex_lines = [
        "\\begin{tabular}{l" + "c" * (len(tasks) + 1) + "}",
        "\\toprule",
        "Method & " + " & ".join(MAIN_TABLE_TASK_LABELS.get(task, task) for task in tasks) + " & AUC \\\\",
        "\\midrule",
    ]
    for algorithm in algorithms:
        cells = []
        for task in tasks:
            vals = [run.final_eval for run in seed_runs if run.algorithm == algorithm and run.task == task]
            cells.append(mean_std_text(vals))
        auc_vals = [run.auc for run in seed_runs if run.algorithm == algorithm and not math.isnan(run.auc)]
        cells.append(mean_std_text(auc_vals))
        tex_lines.append(f"{cmp.ALGORITHM_STYLES.get(algorithm, {'label': algorithm})['label']} & " + " & ".join(cells) + " \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (summary_root / "paper_main_results.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def export_wm_metrics_table(seed_runs: list[SeedRun], summary_root: Path, tasks: list[str], algorithms: list[str], tail_window: int) -> None:
    rows: list[dict[str, object]] = []
    for algorithm in algorithms:
        for task in tasks:
            matching = [run for run in seed_runs if run.algorithm == algorithm and run.task == task]
            row: dict[str, object] = {"algorithm": algorithm, "task": task, "n": len(matching)}
            for metric_key, _ in WM_KEYS:
                values: list[float] = []
                for run in matching:
                    history = run.run_data.metric_history.get(metric_key, [])
                    if history:
                        tail = history[-tail_window:] if len(history) > tail_window else history
                        values.append(statistics.fmean(value for _, value in tail))
                row[f"{metric_key}_mean"] = statistics.fmean(values) if values else ""
                row[f"{metric_key}_std"] = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else "")
            rows.append(row)

    fieldnames = ["algorithm", "task", "n"] + [f"{metric_key}_{suffix}" for metric_key, _ in WM_KEYS for suffix in ("mean", "std")]
    write_csv(summary_root / "paper_wm_metrics.csv", rows, fieldnames)

    tex_lines = [
        "\\begin{tabular}{ll" + "c" * len(WM_KEYS) + "}",
        "\\toprule",
        "Method & Task & " + " & ".join(label for _, label in WM_KEYS) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = []
        for metric_key, _ in WM_KEYS:
            value = row[f"{metric_key}_mean"]
            cells.append("--" if value == "" else f"{float(value):.3f}")
        tex_lines.append(
            f"{cmp.ALGORITHM_STYLES.get(row['algorithm'], {'label': row['algorithm']})['label']} & "
            f"{MAIN_TABLE_TASK_LABELS.get(str(row['task']), str(row['task']))} & "
            + " & ".join(cells)
            + " \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (summary_root / "paper_wm_metrics.tex").write_text("\n".join(tex_lines), encoding="utf-8")


def aggregate_eval_seed_runs(seed_runs: list[SeedRun]) -> dict[str, dict[int, list[float]]]:
    grouped: dict[str, dict[int, list[float]]] = {}
    for run in seed_runs:
        algo_steps = grouped.setdefault(run.algorithm, {})
        for step, value in run.run_data.eval_returns:
            algo_steps.setdefault(int(step), []).append(float(value))
    return grouped


def aggregate_eval_seed_series(seed_runs: list[SeedRun], algorithms: list[str]) -> list[AggregateSeries]:
    grouped = aggregate_eval_seed_runs(seed_runs)
    series: list[AggregateSeries] = []
    for algorithm in algorithms:
        step_map = grouped.get(algorithm, {})
        if not step_map:
            continue
        mean_points = []
        std_low_points = []
        std_high_points = []
        p05_points = []
        p95_points = []
        for step, values in sorted(step_map.items()):
            values_arr = np.asarray(values, dtype=np.float64)
            mean = float(values_arr.mean())
            std = float(values_arr.std(ddof=0))
            p05 = float(np.quantile(values_arr, 0.05))
            p95 = float(np.quantile(values_arr, 0.95))
            mean_points.append((step, mean))
            std_low_points.append((step, mean - std))
            std_high_points.append((step, mean + std))
            p05_points.append((step, p05))
            p95_points.append((step, p95))
        series.append(
            AggregateSeries(
                algorithm=cmp.ALGORITHM_STYLES.get(algorithm, {"label": algorithm})["label"],
                color=cmp.ALGORITHM_STYLES.get(algorithm, {"color": "#4b5563"})["color"],
                mean_points=mean_points,
                std_low_points=std_low_points,
                std_high_points=std_high_points,
                p05_points=p05_points,
                p95_points=p95_points,
            )
        )
    return series


def _scale_coords(
    points: list[tuple[int, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: float,
    height: float,
    left: float,
    top: float,
) -> list[tuple[float, float]]:
    if not points:
        return []
    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)
    coords: list[tuple[float, float]] = []
    for x, y in points:
        px = left + (x - x_min) / x_span * width
        py = top + height - (y - y_min) / y_span * height
        coords.append((px, py))
    return coords


def _coords_to_str(coords: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)


def _band_polygon(
    low_points: list[tuple[int, float]],
    high_points: list[tuple[int, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: float,
    height: float,
    left: float,
    top: float,
) -> str:
    low_coords = _scale_coords(low_points, x_min, x_max, y_min, y_max, width, height, left, top)
    high_coords = _scale_coords(high_points, x_min, x_max, y_min, y_max, width, height, left, top)
    if not low_coords or not high_coords:
        return ""
    coords = low_coords + list(reversed(high_coords))
    return _coords_to_str(coords)


def _aggregate_panel_svg(
    title: str,
    series: list[AggregateSeries],
    left: float,
    top: float,
    width: float,
    height: float,
) -> str:
    all_points = [point for item in series for point in item.mean_points]
    if not all_points:
        return f'<text x="{left}" y="{top + 20}" font-size="16">{title}: no data</text>'

    xs = [point[0] for point in all_points]
    ys = []
    for item in series:
        ys.extend(value for _, value in item.p05_points)
        ys.extend(value for _, value in item.p95_points)
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

    bands = []
    lines = []
    markers = []
    for item in series:
        p_band = _band_polygon(item.p05_points, item.p95_points, x_min, x_max, y_min, y_max, plot_width, plot_height, plot_left, plot_top)
        if p_band:
            bands.append(f'<polygon fill="{item.color}" fill-opacity="0.05" stroke="none" points="{p_band}"/>')
        std_band = _band_polygon(item.std_low_points, item.std_high_points, x_min, x_max, y_min, y_max, plot_width, plot_height, plot_left, plot_top)
        if std_band:
            bands.append(f'<polygon fill="{item.color}" fill-opacity="0.12" stroke="none" points="{std_band}"/>')

        mean_coords = _scale_coords(item.mean_points, x_min, x_max, y_min, y_max, plot_width, plot_height, plot_left, plot_top)
        if mean_coords:
            lines.append(
                f'<polyline fill="none" stroke="{item.color}" stroke-width="3.0" '
                f'stroke-linecap="round" stroke-linejoin="round" points="{_coords_to_str(mean_coords)}"/>'
            )
            mx, my = mean_coords[-1]
            markers.append(f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="4.5" fill="{item.color}" stroke="#ffffff" stroke-width="1.5"/>')

    panel = [
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="16" fill="#ffffff" stroke="#d7dee8"/>',
        f'<text x="{left + 18}" y="{top + 24}" font-size="18" font-weight="700" fill="#111827">{html.escape(title, quote=False)}</text>',
        *y_ticks,
        *x_ticks,
        f'<line x1="{plot_left}" y1="{plot_top + plot_height}" x2="{plot_left + plot_width}" y2="{plot_top + plot_height}" stroke="#6b7280" stroke-width="1.5"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_height}" stroke="#6b7280" stroke-width="1.5"/>',
        *bands,
        *lines,
        *markers,
        f'<text x="{plot_left + plot_width / 2:.2f}" y="{top + height - 10}" text-anchor="middle" font-size="12" fill="#374151">environment step</text>',
        f'<text x="{left + 15}" y="{plot_top + plot_height / 2:.2f}" text-anchor="middle" font-size="12" fill="#374151" transform="rotate(-90 {left + 15},{plot_top + plot_height / 2:.2f})">mean eval episode return</text>',
    ]
    return "".join(panel)


def render_eval_grid_svg(task_to_runs: dict[str, list[SeedRun]], output_path: Path, algorithms: list[str]) -> None:
    width = 1280
    cols = 2
    panel_width = 580
    panel_height = 260
    rows = math.ceil(len(task_to_runs) / cols)
    title_y = 40
    legend_top = 72
    note_y = legend_top + 30
    top_offset = note_y + 24
    height = top_offset + rows * (panel_height + 28) + 24

    legend_runs = [
        cmp.RunData(
            algorithm_id=algo,
            algorithm=cmp.ALGORITHM_STYLES.get(algo, {"label": algo})["label"],
            color=cmp.ALGORITHM_STYLES.get(algo, {"color": "#4b5563"})["color"],
            task="",
            run_dir=Path("."),
            train_returns=[],
            eval_returns=[],
            metric_history={},
        )
        for algo in algorithms
    ]

    panels: list[str] = []
    ordered_tasks = [task for task in DEFAULT_TASKS if task in task_to_runs]
    for index, task in enumerate(ordered_tasks):
        row = index // cols
        col = index % cols
        left = 40 + col * 610
        top = top_offset + row * (panel_height + 28)
        series = aggregate_eval_seed_series(task_to_runs[task], algorithms)
        if series:
            panels.append(
                _aggregate_panel_svg(
                    title=f"{MAIN_TABLE_TASK_LABELS.get(task, task)} Eval Curves",
                    series=series,
                    left=left,
                    top=top,
                    width=panel_width,
                    height=panel_height,
                )
            )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f4f7fb"/>
<text x="40" y="{title_y}" font-size="28" font-weight="800" fill="#111827">Main Results: Eval Learning Curves</text>
{cmp._legend(legend_runs, 42, legend_top)}
<text x="40" y="{note_y}" font-size="13" fill="#4b5563">Curves show seed-mean eval return over completed runs only; dark band = mean ± std, light band = p05–p95.</text>
{"".join(panels)}
</svg>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def _plot_aggregate_series(ax: plt.Axes, series: list[AggregateSeries]) -> None:
    for item in series:
        if not item.mean_points:
            continue
        px = [step for step, _ in item.p05_points]
        p05 = [value for _, value in item.p05_points]
        p95 = [value for _, value in item.p95_points]
        std_low = [value for _, value in item.std_low_points]
        std_high = [value for _, value in item.std_high_points]
        mx = [step for step, _ in item.mean_points]
        mean = [value for _, value in item.mean_points]
        ax.fill_between(px, p05, p95, color=item.color, alpha=0.05, linewidth=0)
        ax.fill_between(px, std_low, std_high, color=item.color, alpha=0.12, linewidth=0)
        ax.plot(mx, mean, color=item.color, linewidth=2.4, solid_capstyle="round", label=item.algorithm)
        ax.scatter([mx[-1]], [mean[-1]], s=28, color=item.color, edgecolors="#ffffff", linewidths=1.0, zorder=3)

    ax.set_facecolor("#ffffff")
    ax.grid(True, color="#e5e7eb", linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(colors="#475569", labelsize=10)
    ax.set_xlabel("environment step", fontsize=11, color="#334155")
    ax.set_ylabel("mean eval episode return", fontsize=11, color="#334155")


def render_eval_grid_png(task_to_runs: dict[str, list[SeedRun]], output_path: Path, algorithms: list[str]) -> None:
    ordered_tasks = [task for task in DEFAULT_TASKS if task in task_to_runs]
    if not ordered_tasks:
        return

    cols = 2
    rows = math.ceil(len(ordered_tasks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(13.2, 3.5 * rows + 1.4), dpi=180)
    fig.patch.set_facecolor("#f4f7fb")
    axes_list = axes.flat if hasattr(axes, "flat") else [axes]

    legend_handles = []
    legend_labels = []
    for axis_index, ax in enumerate(axes_list):
        if axis_index >= len(ordered_tasks):
            ax.axis("off")
            continue
        task = ordered_tasks[axis_index]
        series = aggregate_eval_seed_series(task_to_runs[task], algorithms)
        if not series:
            ax.axis("off")
            continue
        _plot_aggregate_series(ax, series)
        ax.set_title(f"{MAIN_TABLE_TASK_LABELS.get(task, task)} Eval Curves", loc="left", fontsize=13, fontweight="bold", color="#111827", pad=8)
        if not legend_handles:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.suptitle("Main Results: Eval Learning Curves", x=0.06, y=0.985, ha="left", fontsize=21, fontweight="bold", color="#111827")
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(0.06, 0.94),
            ncol=min(4, max(1, len(legend_handles))),
            frameon=False,
            fontsize=10,
        )
    fig.text(
        0.06,
        0.89,
        "Curves show seed-mean eval return over completed runs only; dark band = mean ± std, light band = p05–p95.",
        fontsize=10.5,
        color="#475569",
        ha="left",
        va="top",
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.84, hspace=0.36, wspace=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def render_eval_band_png(task: str, seed_runs: list[SeedRun], output_path: Path, algorithms: list[str]) -> None:
    series = aggregate_eval_seed_series(seed_runs, algorithms)
    if not series:
        return

    fig, ax = plt.subplots(figsize=(12.5, 4.4), dpi=180)
    fig.patch.set_facecolor("#f4f7fb")
    _plot_aggregate_series(ax, series)
    ax.set_title(f"{MAIN_TABLE_TASK_LABELS.get(task, task)} Eval Curves", loc="left", fontsize=15, fontweight="bold", color="#111827", pad=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(
        f"{MAIN_TABLE_TASK_LABELS.get(task, task)}: Eval Learning Curves",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="#111827",
    )
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.07, 0.93),
            ncol=min(4, max(1, len(handles))),
            frameon=False,
            fontsize=10,
        )
    fig.text(
        0.07,
        0.885,
        "Mean line with mean±std (dark) and p05–p95 (light) bands across completed seeds.",
        fontsize=10.5,
        color="#475569",
        ha="left",
        va="top",
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.78)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
