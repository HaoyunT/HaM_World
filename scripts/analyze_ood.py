from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import FINAL_ROOT

MPLCONFIGDIR = FINAL_ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ALGO_ORDER = ["hamworld", "tdmpc2", "jepa", "dreamerv3", "ppo", "sac"]
ALGO_LABELS = {
    "hamworld": "HaM-World",
    "tdmpc2": "TD-MPC2",
    "jepa": "JEPA",
    "dreamerv3": "DreamerV3",
    "ppo": "PPO",
    "sac": "SAC",
}
ALGO_COLORS = {
    "hamworld": "#2563eb",
    "tdmpc2": "#d97706",
    "jepa": "#dc2626",
    "dreamerv3": "#059669",
    "ppo": "#7c3aed",
    "sac": "#0f766e",
}
TASK_ORDER = ["reacher_easy", "finger_spin", "cheetah_run", "cartpole_swingup"]
TASK_LABELS = {
    "reacher_easy": "Reacher",
    "finger_spin": "Finger",
    "cheetah_run": "Cheetah",
    "cartpole_swingup": "Cartpole",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze OOD evaluation results and export retention summaries/plots.")
    parser.add_argument("--input-csv", required=True, help="Path to ood_results.csv")
    parser.add_argument("--summary-dir", default=None, help="Directory for summary CSV outputs.")
    parser.add_argument("--figures-dir", default=None, help="Directory for figure outputs.")
    return parser.parse_args()


def _to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _safe_mean(values: list[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    if not finite:
        return float("nan")
    return statistics.fmean(finite)


def _format_float(value: float) -> str:
    return "" if math.isnan(value) else f"{value:.6f}"


def load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "algo": row["algo"],
                    "task": row["task"],
                    "seed": int(float(row["seed"])),
                    "condition": row["condition"],
                    "episodes": int(float(row["episodes"])),
                    "return_mean": _to_float(row["return_mean"]),
                    "return_std": _to_float(row["return_std"]),
                    "mse_horizon": int(float(row["mse_horizon"])),
                    "mse_mean": _to_float(row["mse_mean"]),
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    seed_id_return = {
        (str(row["algo"]), str(row["task"]), int(row["seed"])): float(row["return_mean"])
        for row in rows
        if str(row["condition"]) == "id"
    }
    seed_id_mse = {
        (str(row["algo"]), str(row["task"]), int(row["seed"])): float(row["mse_mean"])
        for row in rows
        if str(row["condition"]) == "id"
    }

    condition_rows: list[dict[str, object]] = []
    task_groups: dict[tuple[str, str], dict[str, object]] = {}

    condition_grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row["condition"]) == "id":
            continue
        condition_grouped[(str(row["algo"]), str(row["task"]), str(row["condition"]))].append(row)

    for (algo, task, condition), group in sorted(condition_grouped.items()):
        retention_values = []
        mse_ratio_values = []
        for row in group:
            key = (algo, task, int(row["seed"]))
            id_return = seed_id_return.get(key, float("nan"))
            id_mse = seed_id_mse.get(key, float("nan"))
            if abs(id_return) > 1e-9 and not math.isnan(id_return):
                retention_values.append(float(row["return_mean"]) / id_return)
            if not math.isnan(id_mse) and id_mse > 1e-12 and not math.isnan(float(row["mse_mean"])):
                mse_ratio_values.append(float(row["mse_mean"]) / id_mse)

        condition_row = {
            "algo": algo,
            "task": task,
            "condition": condition,
            "n": len(group),
            "id_return_mean": _safe_mean([seed_id_return.get((algo, task, int(row["seed"])), float("nan")) for row in group]),
            "ood_return_mean": _safe_mean([float(row["return_mean"]) for row in group]),
            "avg_retention": _safe_mean(retention_values),
            "mse_id_mean": _safe_mean([seed_id_mse.get((algo, task, int(row["seed"])), float("nan")) for row in group]),
            "mse_ood_mean": _safe_mean([float(row["mse_mean"]) for row in group]),
            "mse_ratio": _safe_mean(mse_ratio_values),
        }
        condition_rows.append(condition_row)

        task_group = task_groups.setdefault(
            (algo, task),
            {
                "algo": algo,
                "task": task,
                "retentions": [],
                "ood_returns": [],
                "id_returns": [],
                "mse_ids": [],
                "mse_oods": [],
                "condition_retentions": [],
            },
        )
        task_group["retentions"].extend(retention_values)
        task_group["ood_returns"].extend(float(row["return_mean"]) for row in group)
        task_group["id_returns"].extend(
            seed_id_return.get((algo, task, int(row["seed"])), float("nan")) for row in group
        )
        task_group["mse_ids"].extend(
            seed_id_mse.get((algo, task, int(row["seed"])), float("nan")) for row in group
        )
        task_group["mse_oods"].extend(float(row["mse_mean"]) for row in group)
        if not math.isnan(condition_row["avg_retention"]):
            task_group["condition_retentions"].append((condition, float(condition_row["avg_retention"])))

    ranking_rows: list[dict[str, object]] = []
    overall_rows: list[dict[str, object]] = []
    for (algo, task), stats in sorted(task_groups.items()):
        avg_retention = _safe_mean(stats["retentions"])
        avg_id_return = _safe_mean(stats["id_returns"])
        avg_ood_return = _safe_mean(stats["ood_returns"])
        avg_mse_id = _safe_mean(stats["mse_ids"])
        avg_mse_ood = _safe_mean(stats["mse_oods"])
        mse_ratio = float("nan")
        if not math.isnan(avg_mse_id) and avg_mse_id > 1e-12 and not math.isnan(avg_mse_ood):
            mse_ratio = avg_mse_ood / avg_mse_id

        worst_condition = ""
        worst_retention = float("nan")
        if stats["condition_retentions"]:
            worst_condition, worst_retention = min(stats["condition_retentions"], key=lambda item: item[1])

        ranking_rows.append(
            {
                "scope": "task",
                "algo": algo,
                "task": task,
                "avg_id_return": avg_id_return,
                "avg_ood_return": avg_ood_return,
                "ood_delta": avg_ood_return - avg_id_return if not math.isnan(avg_id_return) and not math.isnan(avg_ood_return) else float("nan"),
                "avg_retention": avg_retention,
                "worst_condition": worst_condition,
                "worst_condition_retention": worst_retention,
                "avg_mse_id": avg_mse_id,
                "avg_mse_ood": avg_mse_ood,
                "mse_ratio": mse_ratio,
                "n_ood_rows": len(stats["ood_returns"]),
            }
        )
        overall_rows.append(
            {
                "algo": algo,
                "task": task,
                "avg_retention": avg_retention,
                "worst_condition_retention": worst_retention,
            }
        )

    overall_grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in overall_rows:
        overall_grouped[str(row["algo"])].append(row)

    for algo, group in sorted(overall_grouped.items()):
        ranking_rows.append(
            {
                "scope": "overall",
                "algo": algo,
                "task": "",
                "avg_id_return": float("nan"),
                "avg_ood_return": float("nan"),
                "ood_delta": float("nan"),
                "avg_retention": _safe_mean([float(row["avg_retention"]) for row in group]),
                "worst_condition": "",
                "worst_condition_retention": _safe_mean([float(row["worst_condition_retention"]) for row in group]),
                "avg_mse_id": float("nan"),
                "avg_mse_ood": float("nan"),
                "mse_ratio": float("nan"),
                "n_ood_rows": sum(int(task_groups[(algo, str(row["task"]))]["ood_returns"].__len__()) for row in group),
            }
        )

    def _rank(rows_in: list[dict[str, object]], key: str) -> None:
        sorted_rows = sorted(rows_in, key=lambda row: float(row[key]), reverse=True)
        for index, row in enumerate(sorted_rows, start=1):
            row[f"rank_by_{key}"] = index

    _rank([row for row in ranking_rows if row["scope"] == "overall"], "avg_retention")
    for task in sorted({str(row["task"]) for row in ranking_rows if row["scope"] == "task"}):
        _rank([row for row in ranking_rows if row["scope"] == "task" and row["task"] == task], "avg_retention")
    return ranking_rows, condition_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalized = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float):
                    normalized[key] = _format_float(value)
                else:
                    normalized[key] = value
            writer.writerow(normalized)


def _plot_metric(ranking_rows: list[dict[str, object]], metric_key: str, output_path: Path, title: str, y_label: str) -> None:
    task_rows = [row for row in ranking_rows if row["scope"] == "task"]
    tasks = [task for task in TASK_ORDER if any(str(row["task"]) == task for row in task_rows)]
    if not tasks:
        return

    ncols = 2 if len(tasks) > 1 else 1
    nrows = (len(tasks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.2 * nrows), dpi=200, sharey=True)
    fig.patch.set_facecolor("#f4f7fb")
    if hasattr(axes, "flatten"):
        axes = list(axes.flatten())
    else:
        axes = [axes]

    for axis, task in zip(axes, tasks):
        rows = [row for row in task_rows if row["task"] == task]
        rows.sort(key=lambda row: ALGO_ORDER.index(str(row["algo"])) if str(row["algo"]) in ALGO_ORDER else 999)
        labels = [ALGO_LABELS.get(str(row["algo"]), str(row["algo"])) for row in rows]
        values = [float(row[metric_key]) for row in rows]
        colors = [ALGO_COLORS.get(str(row["algo"]), "#4b5563") for row in rows]
        bars = axis.bar(labels, values, color=colors, width=0.72)
        axis.axhline(1.0, color="#94a3b8", linewidth=1.0, linestyle="--")
        axis.set_title(TASK_LABELS.get(task, task), loc="left", fontsize=13, fontweight="bold", color="#111827")
        axis.set_ylabel(y_label, fontsize=11, color="#334155")
        axis.set_facecolor("#ffffff")
        axis.grid(True, axis="y", color="#e5e7eb", linewidth=1.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#94a3b8")
        axis.spines["bottom"].set_color("#94a3b8")
        axis.tick_params(axis="x", labelrotation=15, colors="#475569")
        axis.tick_params(axis="y", colors="#475569")
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() * 0.5,
                value + 0.02 * max(1.0, max(values)),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#334155",
            )

    for axis in axes[len(tasks):]:
        axis.set_visible(False)

    fig.suptitle(title, x=0.06, y=0.99, ha="left", fontsize=18, fontweight="bold", color="#111827")
    fig.text(0.06, 0.91, "Dashed line marks retention=1.0; higher is better.", fontsize=10.5, color="#475569", ha="left")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.88 if len(tasks) > 2 else 0.82, wspace=0.16, hspace=0.28)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    if not input_csv.exists():
        raise SystemExit(f"Input CSV does not exist: {input_csv}")

    campaign_root = input_csv.parent.parent
    summary_dir = Path(args.summary_dir).expanduser().resolve() if args.summary_dir else campaign_root / "summary"
    figures_dir = Path(args.figures_dir).expanduser().resolve() if args.figures_dir else campaign_root / "figures"

    rows = load_rows(input_csv)
    ranking_rows, condition_rows = summarize(rows)

    ranking_fieldnames = [
        "scope",
        "algo",
        "task",
        "rank_by_avg_retention",
        "avg_id_return",
        "avg_ood_return",
        "ood_delta",
        "avg_retention",
        "worst_condition",
        "worst_condition_retention",
        "avg_mse_id",
        "avg_mse_ood",
        "mse_ratio",
        "n_ood_rows",
    ]
    condition_fieldnames = [
        "algo",
        "task",
        "condition",
        "n",
        "id_return_mean",
        "ood_return_mean",
        "avg_retention",
        "mse_id_mean",
        "mse_ood_mean",
        "mse_ratio",
    ]
    write_csv(summary_dir / "ood_retention_rankings.csv", ranking_rows, ranking_fieldnames)
    write_csv(summary_dir / "ood_condition_retention.csv", condition_rows, condition_fieldnames)

    _plot_metric(
        ranking_rows,
        metric_key="avg_retention",
        output_path=figures_dir / "ood_avg_retention_by_task.png",
        title="OOD Average Retention By Task",
        y_label="average retention",
    )
    _plot_metric(
        ranking_rows,
        metric_key="worst_condition_retention",
        output_path=figures_dir / "ood_worst_retention_by_task.png",
        title="OOD Worst-Condition Retention By Task",
        y_label="worst-condition retention",
    )

    print(summary_dir / "ood_retention_rankings.csv")
    print(summary_dir / "ood_condition_retention.csv")
    print(figures_dir / "ood_avg_retention_by_task.png")
    print(figures_dir / "ood_worst_retention_by_task.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
