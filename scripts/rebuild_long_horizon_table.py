from __future__ import annotations

import csv
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import RESULTS_MAIN_DATA_ROOT, RESULTS_MAIN_TABLE_ROOT, RUNS_MAIN_ROOT


KEEP = ["hamworld", "dreamerv3", "tdmpc2"]
KEEP_HORIZONS = {"3", "5", "7"}


def main() -> int:
    candidates = [
        RUNS_MAIN_ROOT / "model_based" / "summary" / "long_horizon_k357.csv",
        RESULTS_MAIN_DATA_ROOT / "long_horizon_k357.csv",
    ]
    src = next((path for path in candidates if path.exists()), None)
    if src is None:
        raise FileNotFoundError("Could not find long_horizon_k357.csv in runs/ or results/main/data/.")
    with src.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized = []
    for row in rows:
        row = dict(row)
        if row["horizon"] not in KEEP_HORIZONS:
            continue
        if row["algorithm"] not in KEEP:
            continue
        normalized.append(row)
    fieldnames = list(normalized[0].keys()) if normalized else []
    outputs = [
        RESULTS_MAIN_DATA_ROOT / "long_horizon_k357.csv",
        RESULTS_MAIN_TABLE_ROOT / "long_horizon_k357.csv",
    ]
    for dst in outputs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(normalized)
    stale_outputs = [
        RESULTS_MAIN_DATA_ROOT / "long_horizon_mse15.csv",
        RESULTS_MAIN_DATA_ROOT / "long_horizon_metrics.json",
        RESULTS_MAIN_TABLE_ROOT / "long_horizon_mse15.csv",
    ]
    for stale_path in stale_outputs:
        if stale_path.exists():
            stale_path.unlink()
    print(RESULTS_MAIN_DATA_ROOT / "long_horizon_k357.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
