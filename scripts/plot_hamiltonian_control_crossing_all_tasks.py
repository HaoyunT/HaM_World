#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    ANALYSIS_MECHANISM_CONTROL_CROSSING_ROOT,
    ANALYSIS_MECHANISM_TRACE_QUICK_ROOT,
    RUNS_MAIN_ROOT,
)
import plot_hamiltonian_control_crossing as single_task


@dataclass(frozen=True)
class TaskSpec:
    task: str
    checkpoint_path: Path
    trace_path: Path
    output_dir: Path


@dataclass
class TaskAnalysis:
    spec: TaskSpec
    q_reference: np.ndarray
    p_reference: np.ndarray
    q_dim: int
    p_dim: int
    pair_scores: dict[tuple[int, int], single_task.PairScore]
    pair_summaries: dict[tuple[int, int], single_task.PairSummary]
    pair_sweeps: dict[tuple[int, int], list[dict[str, float]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-render push flow and crossing-rate figures across all tasks.")
    parser.add_argument(
        "--runs-root",
        default=str(RUNS_MAIN_ROOT / "model_based" / "runs" / "hamworld"),
        help="Root directory containing per-task HaM-World runs.",
    )
    parser.add_argument(
        "--analysis-root",
        default=str(ANALYSIS_MECHANISM_TRACE_QUICK_ROOT),
        help="Root directory containing traces for one analysis sweep.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ANALYSIS_MECHANISM_CONTROL_CROSSING_ROOT.with_name("control_crossing_generated")),
        help="Directory for per-task figures and global recommendations.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Seed id used to match traces and checkpoints.")
    parser.add_argument(
        "--trace-mode",
        default="teacher_forced",
        choices=["teacher_forced", "imagined"],
        help="Which trace mode to analyze.",
    )
    parser.add_argument(
        "--tasks",
        default="",
        help="Optional comma-separated task subset. Default: infer all tasks from trace files.",
    )
    parser.add_argument("--checkpoint-step", type=int, default=100000, help="Checkpoint step to load.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0.")
    parser.add_argument("--grid-size", type=int, default=81, help="Grid size for slice evaluation.")
    parser.add_argument("--same-k", type=int, default=2, help="How many same-pair candidates to export per task.")
    parser.add_argument("--cross-k", type=int, default=2, help="How many cross-pair candidates to export per task.")
    parser.add_argument("--trace-quantile-lo", type=float, default=2.0, help="Low percentile for slice bounds.")
    parser.add_argument("--trace-quantile-hi", type=float, default=98.0, help="High percentile for slice bounds.")
    parser.add_argument(
        "--cross-threshold-quantile",
        type=float,
        default=0.60,
        help="Quantile of |∇H_slice·Δx| used for binary crossing events.",
    )
    parser.add_argument(
        "--cross-threshold-sweep",
        default="0.40,0.50,0.60,0.70,0.80,0.90",
        help="Comma-separated threshold quantiles for high/low push crossing curves.",
    )
    return parser.parse_args()


def _parse_task_filter(raw: str) -> set[str] | None:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return set(items) if items else None


def _find_latest_checkpoint(task_dir: Path, seed: int, checkpoint_step: int) -> Path:
    pattern = f"seed_{seed}_*/checkpoints/checkpoint_{checkpoint_step}.pt"
    matches = sorted(task_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint matched {pattern} under {task_dir}")
    return matches[-1].resolve()


def discover_task_specs(
    runs_root: Path,
    analysis_root: Path,
    output_dir: Path,
    seed: int,
    trace_mode: str,
    checkpoint_step: int,
    task_filter: set[str] | None,
) -> list[TaskSpec]:
    traces_dir = analysis_root / "traces"
    suffix = f"_seed{seed}_{trace_mode}.npz"
    specs: list[TaskSpec] = []
    for trace_path in sorted(traces_dir.glob(f"*{suffix}")):
        task = trace_path.name[: -len(suffix)]
        if task_filter is not None and task not in task_filter:
            continue
        checkpoint_path = _find_latest_checkpoint(runs_root / task, seed=seed, checkpoint_step=checkpoint_step)
        specs.append(
            TaskSpec(
                task=task,
                checkpoint_path=checkpoint_path,
                trace_path=trace_path.resolve(),
                output_dir=(output_dir / task).resolve(),
            )
        )
    if not specs:
        raise RuntimeError(f"No tasks discovered under {traces_dir} for seed={seed}, trace_mode={trace_mode}.")
    return specs


def _build_paper_score_row(
    task: str,
    summary: single_task.PairSummary,
    paper_score: float,
    pair_rank_within_kind: int,
    figure_rank_within_task: int,
) -> dict[str, object]:
    row = asdict(summary)
    row.update(
        {
            "task": task,
            "paper_score": float(paper_score),
            "pair_rank_within_kind": int(pair_rank_within_kind),
            "figure_rank_within_task": int(figure_rank_within_task),
        }
    )
    return row


def _safe_minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if not np.any(finite):
        return [0.0 for _ in values]
    min_value = float(np.min(array[finite]))
    max_value = float(np.max(array[finite]))
    if max_value - min_value <= 1e-12:
        return [1.0 if np.isfinite(value) else 0.0 for value in values]
    normalized = []
    for value in values:
        if not np.isfinite(value):
            normalized.append(0.0)
        else:
            normalized.append(float((value - min_value) / (max_value - min_value)))
    return normalized


def _attach_paper_scores(task_analyses: list[TaskAnalysis]) -> tuple[dict[tuple[str, int, int], float], list[dict[str, object]]]:
    key_to_summary: dict[tuple[str, int, int], single_task.PairSummary] = {}
    kind_keys: dict[str, list[tuple[str, int, int]]] = {"same": [], "cross": []}
    for analysis in task_analyses:
        for key, summary in analysis.pair_summaries.items():
            full_key = (analysis.spec.task, key[0], key[1])
            key_to_summary[full_key] = summary
            kind_keys.setdefault(summary.kind, []).append(full_key)

    key_to_paper_score: dict[tuple[str, int, int], float] = {}
    rows: list[dict[str, object]] = []
    for kind, keys in kind_keys.items():
        summaries = [key_to_summary[key] for key in keys]
        score_norm = _safe_minmax([summary.score for summary in summaries])
        lift_norm = _safe_minmax([max(0.0, summary.sweep_lift_auc) for summary in summaries])
        energy_norm = _safe_minmax([max(0.0, summary.corr_abs_delta_h_abs_control_energy_push) for summary in summaries])
        binary_norm = _safe_minmax([max(0.0, summary.sweep_binary_corr_auc) for summary in summaries])

        kind_rows: list[tuple[tuple[str, int, int], float]] = []
        for index, key in enumerate(keys):
            paper_score = 0.35 * score_norm[index] + 0.35 * lift_norm[index] + 0.20 * energy_norm[index] + 0.10 * binary_norm[index]
            key_to_paper_score[key] = float(paper_score)
            kind_rows.append((key, float(paper_score)))

        ranks = {key: rank for rank, (key, _) in enumerate(sorted(kind_rows, key=lambda item: item[1], reverse=True), start=1)}
        for key in keys:
            task, q_index, p_index = key
            rows.append(
                _build_paper_score_row(
                    task=task,
                    summary=key_to_summary[key],
                    paper_score=key_to_paper_score[key],
                    pair_rank_within_kind=ranks[key],
                    figure_rank_within_task=0,
                )
            )
    return key_to_paper_score, rows


def analyze_task(
    spec: TaskSpec,
    threshold_quantiles: list[float],
    cross_threshold_quantile: float,
) -> TaskAnalysis:
    trace = single_task._load_trace(spec.trace_path)
    valid_mask = np.asarray(trace["valid_mask"], dtype=bool)
    q_flat = single_task._flatten_valid_vectors(trace["q"], valid_mask)
    p_flat = single_task._flatten_valid_vectors(trace["p"], valid_mask)
    q_reference = np.median(q_flat, axis=0).astype(np.float32)
    p_reference = np.median(p_flat, axis=0).astype(np.float32)

    all_scores = single_task.score_pairs(trace)
    pair_scores = {(pair.q_index, pair.p_index): pair for pair in all_scores}
    pair_summaries: dict[tuple[int, int], single_task.PairSummary] = {}
    pair_sweeps: dict[tuple[int, int], list[dict[str, float]]] = {}
    for pair in all_scores:
        steps = single_task.build_pair_steps(trace, pair)
        summary = single_task.summarize_pair(pair, steps, cross_threshold_quantile=cross_threshold_quantile)
        sweep_rows, _, _ = single_task.compute_threshold_sweep(steps, threshold_quantiles)
        summary = single_task.attach_sweep_metrics(summary, sweep_rows)
        pair_summaries[(pair.q_index, pair.p_index)] = summary
        pair_sweeps[(pair.q_index, pair.p_index)] = sweep_rows

    return TaskAnalysis(
        spec=spec,
        q_reference=q_reference,
        p_reference=p_reference,
        q_dim=int(np.asarray(trace["q_dim"]).item()),
        p_dim=int(np.asarray(trace["p_dim"]).item()),
        pair_scores=pair_scores,
        pair_summaries=pair_summaries,
        pair_sweeps=pair_sweeps,
    )


def _select_task_candidates(
    analysis: TaskAnalysis,
    key_to_paper_score: dict[tuple[str, int, int], float],
    same_k: int,
    cross_k: int,
) -> tuple[list[tuple[single_task.PairScore, single_task.PairSummary, float]], list[tuple[single_task.PairScore, single_task.PairSummary, float]]]:
    ranked_same: list[tuple[single_task.PairScore, single_task.PairSummary, float]] = []
    ranked_cross: list[tuple[single_task.PairScore, single_task.PairSummary, float]] = []
    for key, summary in analysis.pair_summaries.items():
        pair = analysis.pair_scores[key]
        paper_score = key_to_paper_score[(analysis.spec.task, key[0], key[1])]
        item = (pair, summary, paper_score)
        if summary.kind == "same":
            ranked_same.append(item)
        else:
            ranked_cross.append(item)
    ranked_same.sort(key=lambda item: item[2], reverse=True)
    ranked_cross.sort(key=lambda item: item[2], reverse=True)
    return ranked_same[: max(0, int(same_k))], ranked_cross[: max(0, int(cross_k))]


def _write_task_tables(
    analysis: TaskAnalysis,
    key_to_paper_score: dict[tuple[str, int, int], float],
    output_dir: Path,
) -> None:
    rows_same: list[tuple[single_task.PairSummary, float]] = []
    rows_cross: list[tuple[single_task.PairSummary, float]] = []
    all_rows: list[dict[str, object]] = []
    for key, summary in analysis.pair_summaries.items():
        paper_score = key_to_paper_score[(analysis.spec.task, key[0], key[1])]
        if summary.kind == "same":
            rows_same.append((summary, paper_score))
        else:
            rows_cross.append((summary, paper_score))

    rows_same.sort(key=lambda item: item[1], reverse=True)
    rows_cross.sort(key=lambda item: item[1], reverse=True)
    for kind_rows in [rows_same, rows_cross]:
        for rank, (summary, paper_score) in enumerate(kind_rows, start=1):
            row = _build_paper_score_row(
                task=analysis.spec.task,
                summary=summary,
                paper_score=paper_score,
                pair_rank_within_kind=rank,
                figure_rank_within_task=0,
            )
            all_rows.append(row)

    single_task.write_csv(output_dir / "all_pairs_ranked.csv", all_rows)
    single_task.write_csv(output_dir / "top_same_pairs.csv", [row for row in all_rows if row["kind"] == "same"])
    single_task.write_csv(output_dir / "top_cross_pairs.csv", [row for row in all_rows if row["kind"] == "cross"])


def render_task_candidates(
    analysis: TaskAnalysis,
    key_to_paper_score: dict[tuple[str, int, int], float],
    same_k: int,
    cross_k: int,
    threshold_quantiles: list[float],
    grid_size: int,
    quantile_lo: float,
    quantile_hi: float,
    device: torch.device,
) -> dict[str, list[dict[str, object]]]:
    task_dir = analysis.spec.output_dir
    figures_dir = task_dir / "figures"
    same_dir = figures_dir / "same"
    cross_dir = figures_dir / "cross"
    tables_dir = task_dir / "tables"
    for directory in [same_dir, cross_dir, tables_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    _write_task_tables(analysis, key_to_paper_score, tables_dir)

    trace = single_task._load_trace(analysis.spec.trace_path)
    world_model = single_task._load_checkpoint_world_model(analysis.spec.checkpoint_path, device)
    valid_mask = np.asarray(trace["valid_mask"], dtype=bool)
    q_flat = single_task._flatten_valid_vectors(trace["q"], valid_mask)
    p_flat = single_task._flatten_valid_vectors(trace["p"], valid_mask)

    selected_same, selected_cross = _select_task_candidates(analysis, key_to_paper_score, same_k=same_k, cross_k=cross_k)
    selection_manifest: dict[str, list[dict[str, object]]] = {"same": [], "cross": []}
    for kind, selected, target_dir in [("same", selected_same, same_dir), ("cross", selected_cross, cross_dir)]:
        for rank, (pair, summary, paper_score) in enumerate(selected, start=1):
            pair_stem = f"{kind}_rank{rank}_q{pair.q_index}_p{pair.p_index}"
            slice_eval = single_task.evaluate_slice(
                world_model=world_model,
                q_reference=analysis.q_reference,
                p_reference=analysis.p_reference,
                pair=pair,
                q_values=q_flat[:, pair.q_index],
                p_values=p_flat[:, pair.p_index],
                grid_size=grid_size,
                quantile_lo=quantile_lo,
                quantile_hi=quantile_hi,
                device=device,
            )
            steps = single_task.build_pair_steps(trace, pair)
            sweep_rows, _, _ = single_task.compute_threshold_sweep(steps, threshold_quantiles)

            flow_path = target_dir / f"{pair_stem}_push_flow.png"
            curve_path = target_dir / f"{pair_stem}_push_crossing_rate.png"
            single_task.draw_pair_push_flow(flow_path, slice_eval, pair, steps, summary)
            single_task.draw_pair_push_crossing_rate_curve(curve_path, pair, summary, sweep_rows)

            manifest_row = {
                "task": analysis.spec.task,
                "kind": kind,
                "rank": rank,
                "q_index": pair.q_index,
                "p_index": pair.p_index,
                "paper_score": float(paper_score),
                "pair_score": float(pair.score),
                "sweep_lift_auc": float(summary.sweep_lift_auc),
                "corr_abs_delta_h_abs_control_energy_push": float(summary.corr_abs_delta_h_abs_control_energy_push),
                "sweep_binary_corr_auc": float(summary.sweep_binary_corr_auc),
                "flow_path": str(flow_path),
                "curve_path": str(curve_path),
            }
            selection_manifest[kind].append(manifest_row)

    with (task_dir / "task_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "task": analysis.spec.task,
                "checkpoint_path": str(analysis.spec.checkpoint_path),
                "trace_path": str(analysis.spec.trace_path),
                "q_dim": analysis.q_dim,
                "p_dim": analysis.p_dim,
                "selected": selection_manifest,
            },
            handle,
            indent=2,
        )
    return selection_manifest


def _copy_recommendation_assets(
    destination_dir: Path,
    prefix: str,
    row: dict[str, object],
) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    flow_src = Path(str(row["flow_path"]))
    curve_src = Path(str(row["curve_path"]))
    shutil.copy2(flow_src, destination_dir / f"{prefix}_push_flow.png")
    shutil.copy2(curve_src, destination_dir / f"{prefix}_push_crossing_rate.png")


def write_global_outputs(
    output_dir: Path,
    global_rows: list[dict[str, object]],
    selection_rows: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    single_task.write_csv(output_dir / "all_pairs_global_ranked.csv", global_rows)
    single_task.write_csv(output_dir / "selected_candidates.csv", selection_rows)

    recommended_dir = output_dir / "paper_recommendations"
    recommended_dir.mkdir(parents=True, exist_ok=True)
    by_kind = {
        "same": [row for row in selection_rows if row["kind"] == "same"],
        "cross": [row for row in selection_rows if row["kind"] == "cross"],
    }
    recommendation_rows: list[dict[str, object]] = []
    for kind, rows in by_kind.items():
        ordered = sorted(rows, key=lambda row: float(row["paper_score"]), reverse=True)
        if not ordered:
            continue
        best = ordered[0]
        recommendation_rows.append(best)
        _copy_recommendation_assets(recommended_dir, f"recommended_{kind}", best)

    single_task.write_csv(recommended_dir / "recommended_pairs.csv", recommendation_rows)
    with (recommended_dir / "recommended_pairs.json").open("w", encoding="utf-8") as handle:
        json.dump(recommendation_rows, handle, indent=2)

    lines = [
        "# Hamiltonian Push/Crossing Figure Recommendations",
        "",
        "This folder contains the overall best same-pair and cross-pair figures across all analyzed tasks.",
        "",
    ]
    for row in recommendation_rows:
        lines.extend(
            [
                f"## {row['kind']} pair recommendation",
                f"- task: {row['task']}",
                f"- pair: q[{row['q_index']}] / p[{row['p_index']}]",
                f"- paper_score: {float(row['paper_score']):.3f}",
                f"- sweep_lift_auc: {float(row['sweep_lift_auc']):.3f}",
                f"- corr(|ΔH|, |push|): {float(row['corr_abs_delta_h_abs_control_energy_push']):.3f}",
                f"- sweep_binary_corr_auc: {float(row['sweep_binary_corr_auc']):.3f}",
                "",
            ]
        )
    (recommended_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).expanduser().resolve()
    analysis_root = Path(args.analysis_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    task_filter = _parse_task_filter(args.tasks)
    threshold_quantiles = single_task._parse_float_list(args.cross_threshold_sweep)
    device = torch.device(args.device)

    specs = discover_task_specs(
        runs_root=runs_root,
        analysis_root=analysis_root,
        output_dir=output_dir,
        seed=int(args.seed),
        trace_mode=str(args.trace_mode),
        checkpoint_step=int(args.checkpoint_step),
        task_filter=task_filter,
    )

    analyses = [
        analyze_task(
            spec=spec,
            threshold_quantiles=threshold_quantiles,
            cross_threshold_quantile=float(args.cross_threshold_quantile),
        )
        for spec in specs
    ]
    key_to_paper_score, global_rows = _attach_paper_scores(analyses)

    selection_rows: list[dict[str, object]] = []
    for analysis in analyses:
        manifest = render_task_candidates(
            analysis=analysis,
            key_to_paper_score=key_to_paper_score,
            same_k=int(args.same_k),
            cross_k=int(args.cross_k),
            threshold_quantiles=threshold_quantiles,
            grid_size=int(args.grid_size),
            quantile_lo=float(args.trace_quantile_lo),
            quantile_hi=float(args.trace_quantile_hi),
            device=device,
        )
        selection_rows.extend(manifest["same"])
        selection_rows.extend(manifest["cross"])

    write_global_outputs(output_dir, global_rows, selection_rows)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "runs_root": str(runs_root),
                "analysis_root": str(analysis_root),
                "seed": int(args.seed),
                "trace_mode": str(args.trace_mode),
                "checkpoint_step": int(args.checkpoint_step),
                "tasks": [spec.task for spec in specs],
                "same_k": int(args.same_k),
                "cross_k": int(args.cross_k),
            },
            handle,
            indent=2,
        )

    print(output_dir / "all_pairs_global_ranked.csv")
    print(output_dir / "selected_candidates.csv")
    print(output_dir / "paper_recommendations" / "recommended_pairs.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
