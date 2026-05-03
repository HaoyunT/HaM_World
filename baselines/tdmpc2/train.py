from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from tdmpc2.runtime import apply_overrides, expand_task_runs, load_config
from tdmpc2.trainer import TDMPC2Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TD-MPC2 baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "configs" / "default.yaml"),
        help="Path to a YAML config.",
    )
    parser.add_argument("--task", default=None, type=str, help="Optional task id to run.")
    parser.add_argument("--resume", default=None, type=str, help="Optional checkpoint path to resume from.")
    parser.add_argument("--override", action="append", default=[], help="Dotted config override.")
    return parser.parse_args()


def run_from_config(config: dict, resume_checkpoint: str | None = None) -> None:
    trainer = TDMPC2Trainer(config, resume_checkpoint=resume_checkpoint)
    trainer.run()


def main(config: dict | None = None, resume_checkpoint: str | None = None) -> None:
    if config is not None:
        run_from_config(config, resume_checkpoint=resume_checkpoint)
        return

    args = parse_args()
    if args.resume is not None:
        checkpoint = torch.load(Path(args.resume), map_location="cpu", weights_only=False)
        resolved_config = apply_overrides(checkpoint["config"], args.override)
        run_from_config(resolved_config, resume_checkpoint=args.resume)
        return

    resolved_config = apply_overrides(load_config(Path(args.config)), args.override)
    for run_config in expand_task_runs(resolved_config, task_filter=args.task):
        run_from_config(run_config)


if __name__ == "__main__":
    main()
