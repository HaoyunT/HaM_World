from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path


FINAL_ROOT = Path(__file__).resolve().parents[1]

RUNS_ROOT = FINAL_ROOT / "runs"
RUNS_MAIN_ROOT = RUNS_ROOT / "main"
RUNS_OOD_ROOT = RUNS_ROOT / "ood"
RUNS_ABLATION_ROOT = RUNS_ROOT / "ablation"

RESULTS_ROOT = FINAL_ROOT / "results"
RESULTS_MAIN_ROOT = RESULTS_ROOT / "main"
RESULTS_MAIN_DATA_ROOT = RESULTS_MAIN_ROOT / "data"
RESULTS_MAIN_FIG_ROOT = RESULTS_MAIN_ROOT / "figures"
RESULTS_MAIN_TABLE_ROOT = RESULTS_MAIN_ROOT / "tables"
RESULTS_OOD_ROOT = RESULTS_ROOT / "ood"
RESULTS_OOD_DATA_ROOT = RESULTS_OOD_ROOT / "data"
RESULTS_OOD_FIG_ROOT = RESULTS_OOD_ROOT / "figures"
RESULTS_OOD_TABLE_ROOT = RESULTS_OOD_ROOT / "tables"
RESULTS_ABLATION_ROOT = RESULTS_ROOT / "ablation"
RESULTS_ABLATION_DATA_ROOT = RESULTS_ABLATION_ROOT / "data"
RESULTS_ABLATION_TABLE_ROOT = RESULTS_ABLATION_ROOT / "tables"
RESULTS_MECHANISM_ROOT = RESULTS_ROOT / "mechanism"
RESULTS_MECHANISM_FIG_ROOT = RESULTS_MECHANISM_ROOT / "figures"
RESULTS_MECHANISM_TRACE_ROOT = RESULTS_MECHANISM_ROOT / "traces"
RESULTS_MECHANISM_TRACE_QUICK_ROOT = RESULTS_MECHANISM_TRACE_ROOT / "quick_seed7"
RESULTS_MECHANISM_TRACE_EP10_ROOT = RESULTS_MECHANISM_TRACE_ROOT / "ep10_seed7"
RESULTS_MECHANISM_FREERUN_ROOT = RESULTS_MECHANISM_ROOT / "freerun" / "seed7"
RESULTS_APPENDIX_ROOT = RESULTS_ROOT / "appendix"
RESULTS_APPENDIX_DATA_ROOT = RESULTS_APPENDIX_ROOT / "data"
RESULTS_APPENDIX_FIG_ROOT = RESULTS_APPENDIX_ROOT / "figures"
RESULTS_APPENDIX_TABLE_ROOT = RESULTS_APPENDIX_ROOT / "tables"

# Legacy aliases kept so older helper scripts still import cleanly.
ANALYSIS_ROOT = RESULTS_ROOT
ANALYSIS_MAIN_ROOT = RESULTS_MAIN_DATA_ROOT
ANALYSIS_OOD_ROOT = RESULTS_OOD_DATA_ROOT
ANALYSIS_ABLATION_ROOT = RESULTS_ABLATION_DATA_ROOT
ANALYSIS_APPENDIX_ROOT = RESULTS_APPENDIX_ROOT
ANALYSIS_MANIFEST_ROOT = RESULTS_ROOT / "manifests"
ANALYSIS_MECHANISM_ROOT = RESULTS_MECHANISM_ROOT
ANALYSIS_MECHANISM_TRACE_QUICK_ROOT = RESULTS_MECHANISM_TRACE_QUICK_ROOT
ANALYSIS_MECHANISM_TRACE_EP10_ROOT = RESULTS_MECHANISM_TRACE_EP10_ROOT
ANALYSIS_MECHANISM_FREERUN_ROOT = RESULTS_MECHANISM_FREERUN_ROOT
ANALYSIS_MECHANISM_DYNAMICS_MANIFEST = RESULTS_MECHANISM_TRACE_ROOT / "dynamics_traces.csv"
ANALYSIS_MECHANISM_FREERUN_MANIFEST = RESULTS_MECHANISM_TRACE_ROOT / "freerun_traces.csv"
ANALYSIS_MECHANISM_CONTROL_CROSSING_ROOT = RESULTS_MECHANISM_ROOT / "control_crossing"
ANALYSIS_MECHANISM_DIAGNOSTICS_ROOT = RESULTS_MECHANISM_ROOT / "diagnostics"
ANALYSIS_FIG_ROOT = RESULTS_MECHANISM_FIG_ROOT
ANALYSIS_ROLLOUT_ROOT = RESULTS_MAIN_ROOT / "rollout"
ANALYSIS_ROLLOUT_RENDER_ROOT = ANALYSIS_ROLLOUT_ROOT / "renders"
ANALYSIS_ROLLOUT_CHECKPOINT_CACHE_ROOT = ANALYSIS_ROLLOUT_ROOT / "cache"
ANALYSIS_ROLLOUT_MANIFEST = ANALYSIS_ROLLOUT_ROOT / "seed_manifest.csv"
PAPER_ROOT = RESULTS_ROOT
FIG_ROOT = RESULTS_ROOT
PAPER_MAIN_FIG_ROOT = RESULTS_MAIN_FIG_ROOT
PAPER_MECHANISM_FIG_ROOT = RESULTS_MECHANISM_FIG_ROOT
PAPER_APPENDIX_FIG_ROOT = RESULTS_APPENDIX_FIG_ROOT
TABLE_ROOT = RESULTS_ROOT
DOCS_ROOT = FINAL_ROOT / "docs"
SCRIPT_ROOT = FINAL_ROOT / "scripts"
CODE_ROOTS = [
    FINAL_ROOT,
    FINAL_ROOT / "baselines",
]

RAW_ROOT = RUNS_ROOT
DERIVED_ROOT = RESULTS_ROOT

MPLCONFIGDIR = FINAL_ROOT / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def ensure_repo_on_path() -> None:
    for root in reversed(CODE_ROOTS):
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)


def run_repo_python(script_relpath: str, args: list[str]) -> None:
    cmd = [sys.executable, str(FINAL_ROOT / script_relpath), *args]
    subprocess.run(cmd, cwd=FINAL_ROOT, check=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return (FINAL_ROOT / resolved).resolve()


def repo_relative_path(path: str | Path) -> str:
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        try:
            return resolved.relative_to(FINAL_ROOT).as_posix()
        except ValueError:
            return resolved.as_posix()
    return resolved.as_posix()
