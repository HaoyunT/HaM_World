from __future__ import annotations

import argparse
from pathlib import Path

from hamworld import compare as cmp
from hamworld import export_paper_assets as paper
from hamworld import long_horizon_eval as lhe
from hamworld import paper_extra_figures as extras


DEFAULT_ALGOS = ["hamworld", "tdmpc2", "dreamerv3", "jepa"]
DEFAULT_TASKS = ["reacher_easy", "finger_spin", "cheetah_run", "cartpole_swingup"]


def _resolve(path: str | None, default: Path) -> Path:
    if path is None:
        return default.resolve()
    return Path(path).expanduser().resolve()


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_formats(formats: list[str] | None) -> list[str]:
    normalized = []
    for fmt in formats or ["png"]:
        lower = fmt.lower()
        if lower not in {"svg", "png"}:
            raise ValueError(f"Unsupported figure format: {fmt}")
        if lower not in normalized:
            normalized.append(lower)
    return normalized


def _render_campaign_compare(
    campaign_root: Path,
    runs_root: Path,
    by_task_root: Path,
    tasks: list[str],
    algorithms: list[str],
    train_window: int,
    max_step: int | None,
    run_selection: str,
    formats: list[str],
) -> None:
    del campaign_root
    task_compare_root = _ensure_dir(by_task_root)
    for task in tasks:
        runs = []
        for algorithm in algorithms:
            run = cmp._load_run(
                outputs_root=runs_root,
                algorithm_id=algorithm,
                task=task,
                train_window=train_window,
                max_step=max_step,
                run_selection=run_selection,
            )
            if run is not None:
                runs.append(run)
        if len(runs) < 2:
            continue
        output_dir = _ensure_dir(task_compare_root / task)
        step_note = f"run-selection={run_selection}" + (f"; max-step={max_step}" if max_step is not None else "")
        for fmt in formats:
            cmp.render_series_compare(
                task,
                runs,
                output_dir / f"train.{fmt}",
                series_name="train_returns",
                panel_title="Train Episode Return (Smoothed)",
                y_label="train episode return",
                output_format=fmt,
                max_step=max_step,
                title=f"{task} Train Compare",
                note=step_note,
            )
            cmp.render_series_compare(
                task,
                runs,
                output_dir / f"eval.{fmt}",
                series_name="eval_returns",
                panel_title="Eval Episode Return (Mean)",
                y_label="mean eval episode return",
                output_format=fmt,
                max_step=max_step,
                title=f"{task} Eval Compare",
                note=step_note,
            )
            cmp.render_metric_compare(
                task,
                runs,
                output_dir / f"metrics.{fmt}",
                output_format=fmt,
                max_step=max_step,
                title=f"{task} Metric Compare",
                note=step_note,
            )


def _render_task_eval_bands(
    seed_runs: list[paper.SeedRun],
    by_task_root: Path,
    tasks: list[str],
    algorithms: list[str],
    formats: list[str],
) -> None:
    by_task_root = _ensure_dir(by_task_root)
    for task in tasks:
        task_runs = [run for run in seed_runs if run.task == task]
        if not task_runs:
            continue
        task_output_root = _ensure_dir(by_task_root / task)
        for fmt in formats:
            if fmt == "png":
                paper.render_eval_band_png(task, task_runs, task_output_root / "eval_band.png", algorithms)
                continue
            legend_runs = [
                cmp.RunData(
                    algorithm_id=algo,
                    algorithm=cmp.ALGORITHM_STYLES.get(algo, {"label": algo})["label"],
                    color=cmp.ALGORITHM_STYLES.get(algo, {"color": "#4b5563"})["color"],
                    task=task,
                    run_dir=Path("."),
                    train_returns=[],
                    eval_returns=[],
                    metric_history={},
                )
                for algo in algorithms
            ]
            series = paper.aggregate_eval_seed_series(task_runs, algorithms)
            if not series:
                continue
            width = 1280
            height = 420
            title_y = 40
            legend_top = 72
            note_y = 110
            panel = paper._aggregate_panel_svg(
                title=f"{paper.MAIN_TABLE_TASK_LABELS.get(task, task)} Eval Curves",
                series=series,
                left=40,
                top=130,
                width=1200,
                height=250,
            )
            svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f4f7fb"/>
<text x="40" y="{title_y}" font-size="28" font-weight="800" fill="#111827">{paper.MAIN_TABLE_TASK_LABELS.get(task, task)}: Eval Learning Curves</text>
{cmp._legend(legend_runs, 42, legend_top)}
<text x="40" y="{note_y}" font-size="13" fill="#4b5563">Mean line with mean±std (dark) and p05–p95 (light) bands across completed seeds.</text>
{panel}
</svg>"""
            (task_output_root / "eval_band.svg").write_text(svg, encoding="utf-8")


def run_campaign(args: argparse.Namespace) -> int:
    campaign_root = Path(args.campaign_root).expanduser().resolve()
    runs_root = _resolve(args.runs_root, campaign_root / "runs")
    summary_root = _ensure_dir(_resolve(args.summary_root, campaign_root / "summary"))
    figures_root = _ensure_dir(_resolve(args.figures_root, campaign_root / "figures"))
    by_task_root = _ensure_dir(figures_root / "by_tasks")
    formats = _normalize_formats(getattr(args, "formats", None))

    seed_runs = paper.select_seed_runs(runs_root, args.algorithms, args.tasks, args.train_window, args.min_step)
    if not seed_runs:
        print("No completed runs met the minimum step threshold.")
        return 1

    paper.export_main_results_table(seed_runs, summary_root, args.tasks, args.algorithms)
    paper.export_wm_metrics_table(seed_runs, summary_root, args.tasks, args.algorithms, args.tail_window)

    task_to_runs: dict[str, list[paper.SeedRun]] = {}
    for task in args.tasks:
        task_to_runs[task] = [run for run in seed_runs if run.task == task]

    for fmt in formats:
        if fmt == "png":
            paper.render_eval_grid_png(task_to_runs, figures_root / "main_results_eval_grid.png", args.algorithms)
        else:
            paper.render_eval_grid_svg(task_to_runs, figures_root / "main_results_eval_grid.svg", args.algorithms)
    _render_task_eval_bands(seed_runs, by_task_root, args.tasks, args.algorithms, formats)

    _render_campaign_compare(
        campaign_root,
        runs_root,
        by_task_root,
        args.tasks,
        args.algorithms,
        args.train_window,
        args.max_step,
        args.run_selection,
        formats,
    )
    mechanism_paths = extras.export_hamworld_mechanism_figures(seed_runs, by_task_root, args.tasks, formats)
    long_horizon_paths = lhe.export_campaign_long_horizon(
        seed_runs,
        summary_root,
        figures_root,
        by_task_root,
        args.tasks,
        args.algorithms,
        formats,
        device=getattr(args, "analysis_device", None),
        horizons=tuple(getattr(args, "long_horizon_horizons", lhe.DEFAULT_HORIZONS)),
        batch_size=int(getattr(args, "long_horizon_batch_size", 2048)),
        stability_rollouts=int(getattr(args, "stability_rollouts", 5)),
        stability_noise_scale=float(getattr(args, "stability_noise_scale", 0.01)),
        workers=int(getattr(args, "analysis_workers", 4)),
    )

    print("Exported campaign figures/tables:")
    print(summary_root / "paper_main_results.csv")
    print(summary_root / "paper_main_results.tex")
    print(summary_root / "paper_wm_metrics.csv")
    print(summary_root / "paper_wm_metrics.tex")
    for extra_path in [summary_root / "long_horizon_k357.csv"]:
        if extra_path.exists():
            print(extra_path)
    for fmt in formats:
        main_figure = figures_root / f"main_results_eval_grid.{fmt}"
        if main_figure.exists():
            print(main_figure)
        long_horizon_grid = figures_root / f"long_horizon_consistency_grid.{fmt}"
        if long_horizon_grid.exists():
            print(long_horizon_grid)
        long_horizon_stability = figures_root / f"long_horizon_stability_grid.{fmt}"
        if long_horizon_stability.exists():
            print(long_horizon_stability)
        long_horizon_relative = figures_root / f"long_horizon_consistency_relative.{fmt}"
        if long_horizon_relative.exists():
            print(long_horizon_relative)
        for grouped_path in sorted(figures_root.glob(f"long_horizon_mse*_grouped.{fmt}")) + sorted(figures_root.glob(f"long_horizon_stability*_grouped.{fmt}")):
            if grouped_path.exists():
                print(grouped_path)
    for task in args.tasks:
        for fmt in formats:
            band = by_task_root / task / f"eval_band.{fmt}"
            if band.exists():
                print(band)
            mechanism = by_task_root / task / f"hamworld_mechanism.{fmt}"
            if mechanism.exists():
                print(mechanism)
            long_horizon_consistency = by_task_root / task / f"long_horizon_consistency.{fmt}"
            if long_horizon_consistency.exists():
                print(long_horizon_consistency)
            long_horizon_stability = by_task_root / task / f"long_horizon_stability.{fmt}"
            if long_horizon_stability.exists():
                print(long_horizon_stability)
    print(by_task_root)
    return 0


def run_experiment(args: argparse.Namespace) -> int:
    experiment_roots = [Path(path).expanduser().resolve() for path in args.experiment_root]
    if not experiment_roots:
        print("Please provide at least one --experiment-root.")
        return 1

    compare_root = _ensure_dir(Path(args.compare_root).expanduser().resolve() if args.compare_root else Path("figures/compare").resolve())
    experiment_name = args.name or experiment_roots[0].name
    output_root = _ensure_dir(compare_root / "experiments" / experiment_name)
    formats = _normalize_formats(getattr(args, "formats", None))

    tasks = args.tasks or sorted(cmp._experiment_task_ids(experiment_roots))
    generated = []
    for task in tasks:
        runs = cmp._load_experiment_runs(experiment_roots, task, args.train_window, args.max_step, args.run_selection)
        if len(runs) < 2:
            continue
        task_dir = _ensure_dir(output_root / task)
        note = f"run-selection={args.run_selection}" + (f"; max-step={args.max_step}" if args.max_step is not None else "")
        for fmt in formats:
            cmp.render_series_compare(
                task,
                runs,
                task_dir / f"train.{fmt}",
                series_name="train_returns",
                panel_title="Train Episode Return (Smoothed)",
                y_label="train episode return",
                output_format=fmt,
                max_step=args.max_step,
                title=f"{task} Train Compare",
                note=note,
            )
            cmp.render_series_compare(
                task,
                runs,
                task_dir / f"eval.{fmt}",
                series_name="eval_returns",
                panel_title="Eval Episode Return (Mean)",
                y_label="mean eval episode return",
                output_format=fmt,
                max_step=args.max_step,
                title=f"{task} Eval Compare",
                note=note,
            )
            cmp.render_metric_compare(
                task,
                runs,
                task_dir / f"metrics.{fmt}",
                output_format=fmt,
                max_step=args.max_step,
                title=f"{task} Metric Compare",
                note=note,
            )
        generated.append(task_dir)

    if not generated:
        print("No experiment compare figures were generated.")
        return 1

    print("Exported experiment figures:")
    for path in generated:
        print(path)
    return 0
