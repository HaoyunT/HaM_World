from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from hamworld.runtime import ExperimentLogger, apply_overrides, expand_task_runs, load_config, make_env, module_summary
from hamworld.world_model import CanonicalDynamicsWorldModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the canonical dynamics HaM-World architecture.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "default.yaml"),
        help="Path to a YAML config.",
    )
    parser.add_argument("--task", default=None, type=str, help="Optional task id to run.")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override.")
    return parser.parse_args()


def run_from_config(config: dict) -> None:
    task = config["task"]
    env, spec = make_env(task, int(config["experiment"]["seed"]))
    del env
    logger = ExperimentLogger(
        output_root=config["experiment"]["output_dir"],
        algorithm=config["experiment"]["algorithm"],
        task_id=f"{task['id']}_viz",
        seed=int(config["experiment"]["seed"]),
    )
    model = CanonicalDynamicsWorldModel(config, spec.observation_shape[0], spec.action_shape[0])
    summary = module_summary(model)
    logger.info(summary)
    logger.save_json(
        "architecture.json",
        {
            "algorithm": config["experiment"]["algorithm"],
            "task": task["id"],
            "summary": summary,
        },
    )


def main(config: dict | None = None) -> None:
    if config is not None:
        run_from_config(config)
        return

    args = parse_args()
    resolved_config = apply_overrides(load_config(Path(args.config)), args.override)
    for run_config in expand_task_runs(resolved_config, task_filter=args.task):
        run_from_config(run_config)


if __name__ == "__main__":
    main()
