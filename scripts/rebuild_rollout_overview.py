from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import (
    ANALYSIS_ROLLOUT_CHECKPOINT_CACHE_ROOT,
    ANALYSIS_ROLLOUT_MANIFEST,
    ANALYSIS_ROLLOUT_RENDER_ROOT,
    RESULTS_MAIN_FIG_ROOT,
    read_csv_rows,
    resolve_repo_path,
    run_repo_python,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild rollout overview using final analysis manifests and local checkpoints.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ANALYSIS_ROLLOUT_MANIFEST,
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ANALYSIS_ROLLOUT_RENDER_ROOT,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ANALYSIS_ROLLOUT_CHECKPOINT_CACHE_ROOT,
    )
    return parser.parse_args()


def _preseed_cache(manifest: Path, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for row in read_csv_rows(manifest):
        run_dir = resolve_repo_path(row["run_dir"])
        checkpoint = run_dir / "checkpoints" / f"checkpoint_{row['final_step']}.pt"
        if not checkpoint.exists():
            continue
        dst = cache_dir / f"{row['task']}_seed{row['seed']}_checkpoint_{row['final_step']}.pt"
        if not dst.exists() or dst.stat().st_size < 1024:
            shutil.copy2(checkpoint, dst)


def main() -> int:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    _preseed_cache(args.manifest, args.cache_dir)
    run_repo_python(
        "scripts/render_hamworld_checkpoints.py",
        [
            "--manifest",
            str(args.manifest),
            "--output-dir",
            str(args.work_dir),
            "--cache-dir",
            str(args.cache_dir),
        ],
    )
    src = args.work_dir / "overview_4tasks.png"
    dst = RESULTS_MAIN_FIG_ROOT / "rollout_overview_4tasks.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
