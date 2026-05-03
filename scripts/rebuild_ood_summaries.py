from __future__ import annotations

import shutil

from common import RESULTS_OOD_DATA_ROOT, RESULTS_OOD_FIG_ROOT, RESULTS_OOD_TABLE_ROOT, run_repo_python


def main() -> int:
    input_csv = RESULTS_OOD_DATA_ROOT / "results.csv"
    tmp_dir = RESULTS_OOD_DATA_ROOT / "_ood_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    run_repo_python(
        "scripts/analyze_ood.py",
        [
            "--input-csv",
            str(input_csv),
            "--summary-dir",
            str(tmp_dir),
            "--figures-dir",
            str(RESULTS_OOD_FIG_ROOT),
        ],
    )
    export_pairs = [
        (tmp_dir / "ood_retention_rankings.csv", RESULTS_OOD_DATA_ROOT / "retention_rankings.csv"),
        (tmp_dir / "ood_retention_rankings.csv", RESULTS_OOD_TABLE_ROOT / "ood_retention_rankings.csv"),
        (tmp_dir / "ood_condition_retention.csv", RESULTS_OOD_DATA_ROOT / "condition_retention.csv"),
        (tmp_dir / "ood_condition_retention.csv", RESULTS_OOD_TABLE_ROOT / "ood_condition_retention.csv"),
    ]
    for src, dst in export_pairs:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    shutil.rmtree(tmp_dir)
    print(RESULTS_OOD_DATA_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
