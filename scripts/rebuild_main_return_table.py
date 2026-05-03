from __future__ import annotations

import csv
import statistics
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import RESULTS_MAIN_DATA_ROOT, RESULTS_MAIN_TABLE_ROOT


KEEP = ["hamworld", "dreamerv3", "tdmpc2", "ppo_low", "sac_low"]
TASKS = ["reacher_easy", "finger_spin", "cheetah_run", "cartpole_swingup"]
LABELS = {
    "hamworld": "HaM-World",
    "dreamerv3": "DreamerV3",
    "tdmpc2": "TD-MPC2",
    "ppo_low": "PPO",
    "sac_low": "SAC",
}
TASK_LABELS = {
    "reacher_easy": "Reacher",
    "finger_spin": "Finger",
    "cheetah_run": "Cheetah",
    "cartpole_swingup": "Cartpole",
}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _fmt(values: list[float]) -> str:
    return f"{_mean(values):.2f} $\\pm$ {_std(values):.2f}" if values else "--"


def main() -> int:
    manifest = RESULTS_MAIN_DATA_ROOT / "seed_run_manifest.csv"
    rows = [row for row in _load_rows(manifest) if row["algorithm"] in KEEP]

    summary_rows: list[dict[str, object]] = []
    auc_rows: list[dict[str, object]] = []
    for algorithm in KEEP:
        summary_row: dict[str, object] = {"algorithm": algorithm}
        auc_values: list[float] = []
        for task in TASKS:
            task_rows = [row for row in rows if row["algorithm"] == algorithm and row["task"] == task]
            final_values = [float(row["final_eval"]) for row in task_rows]
            task_auc_values = [float(row["auc"]) for row in task_rows]
            summary_row[f"{task}_final_mean"] = _mean(final_values)
            summary_row[f"{task}_final_std"] = _std(final_values)
            summary_row[f"{task}_n"] = len(final_values)
            if task_auc_values:
                auc_rows.append(
                    {
                        "task": task,
                        "algorithm": algorithm,
                        "label": LABELS[algorithm],
                        "auc_mean": _mean(task_auc_values),
                        "auc_std": _std(task_auc_values),
                        "n": len(task_auc_values),
                    }
                )
                auc_values.extend(task_auc_values)
        summary_row["auc_mean"] = _mean(auc_values)
        summary_row["auc_std"] = _std(auc_values)
        summary_rows.append(summary_row)

    fieldnames = ["algorithm"] + [f"{task}_{suffix}" for task in TASKS for suffix in ("final_mean", "final_std", "n")] + ["auc_mean", "auc_std"]
    outputs = [
        RESULTS_MAIN_DATA_ROOT / "paper_main_results.csv",
        RESULTS_MAIN_TABLE_ROOT / "paper_main_results.csv",
    ]
    for path in outputs:
        _write_csv(path, summary_rows, fieldnames)

    auc_fieldnames = ["task", "algorithm", "label", "auc_mean", "auc_std", "n"]
    auc_outputs = [
        RESULTS_MAIN_DATA_ROOT / "focused_auc_by_task.csv",
    ]
    for path in auc_outputs:
        _write_csv(path, auc_rows, auc_fieldnames)

    tex_lines = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Reacher & Finger & Cheetah & Cartpole & AUC \\\\",
        "\\midrule",
    ]
    for algorithm in KEEP:
        algo_rows = [row for row in rows if row["algorithm"] == algorithm]
        cells = []
        for task in TASKS:
            task_values = [float(row["final_eval"]) for row in algo_rows if row["task"] == task]
            cells.append(_fmt(task_values))
        auc_values = [float(row["auc"]) for row in algo_rows]
        cells.append(_fmt(auc_values))
        tex_lines.append(f"{LABELS[algorithm]} & " + " & ".join(cells) + " \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (RESULTS_MAIN_TABLE_ROOT / "paper_main_results.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    print(RESULTS_MAIN_DATA_ROOT / "paper_main_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
