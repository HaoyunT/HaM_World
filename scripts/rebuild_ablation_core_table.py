from __future__ import annotations

import csv
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common import RESULTS_ABLATION_DATA_ROOT, RESULTS_ABLATION_TABLE_ROOT


ORDER = ["full", "a1_wo_geom_struct", "a2_memory_none", "a3_memory_gru"]
LABELS = {
    "full": "Full HaM-World",
    "a1_wo_geom_struct": "A1: w/o geom. struct.",
    "a2_memory_none": "A2: Memory = None",
    "a3_memory_gru": "A3: Memory = GRU",
}


def _load_rows(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    return_rows = {row["algorithm"]: row for row in _load_rows(RESULTS_ABLATION_DATA_ROOT / "return_completed_variants.csv")}
    mse_rows = {(row["variant"], row["task"]): row for row in _load_rows(RESULTS_ABLATION_DATA_ROOT / "variant_task_mse6.csv")}
    ood_rows = {
        (row["variant"], row["task"], row["condition"]): row
        for row in _load_rows(RESULTS_ABLATION_DATA_ROOT / "variant_task_condition_ood.csv")
    }
    merged = []
    for variant in ORDER:
        merged.append(
            {
                "variant": variant,
                "paper_label": LABELS[variant],
                "cheetah_return_mean": return_rows[variant]["cheetah_run_final_mean"],
                "cheetah_return_std": return_rows[variant]["cheetah_run_final_std"],
                "finger_return_mean": return_rows[variant]["finger_spin_final_mean"],
                "finger_return_std": return_rows[variant]["finger_spin_final_std"],
                "cartpole_mse6_mean": mse_rows[(variant, "cartpole_swingup")]["mse6_mean"],
                "cartpole_mse6_std": mse_rows[(variant, "cartpole_swingup")]["mse6_std"],
                "reacher_mse6_mean": mse_rows[(variant, "reacher_easy")]["mse6_mean"],
                "reacher_mse6_std": mse_rows[(variant, "reacher_easy")]["mse6_std"],
                "reacher_mass_0p7_return_mean": ood_rows[(variant, "reacher_easy", "mass_0.7")]["return_mean"],
                "reacher_mass_0p7_return_std": ood_rows[(variant, "reacher_easy", "mass_0.7")]["return_std"],
                "reacher_damp_2p0_return_mean": ood_rows[(variant, "reacher_easy", "damp_2.0")]["return_mean"],
                "reacher_damp_2p0_return_std": ood_rows[(variant, "reacher_easy", "damp_2.0")]["return_std"],
            }
        )

    fieldnames = list(merged[0].keys()) if merged else []
    outputs = [
        RESULTS_ABLATION_DATA_ROOT / "paper_table.csv",
        RESULTS_ABLATION_TABLE_ROOT / "paper_ablation_core_table.csv",
    ]
    for dst in outputs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged)
    print(RESULTS_ABLATION_DATA_ROOT / "paper_table.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
