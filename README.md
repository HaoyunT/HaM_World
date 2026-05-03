# HaM_World

Anonymous paper artifact repository for **HaM-World**.

This repository is a cleaned, paper-facing version of the original research workspace. It keeps:

- the canonical `HaM-World` source code
- the four baseline implementations used in the paper
- the full raw runs for the **main 5-algorithm comparison**
- compact result-layer assets for **OOD** and **ablation**
- paper figures, tables, and minimal mechanism-analysis traces

The paper PDF is intentionally not included in this repository during anonymous review.

## What Is Included

Algorithms kept in this repository:

- `HaM-World`
- `DreamerV3`
- `TD-MPC2`
- `PPO`
- `SAC`

Repository policy:

- `runs/main/` keeps raw checkpoints and logs for the main comparison.
- `results/ood/` and `results/ablation/` keep **result-level exports only**.
- `results/mechanism/` keeps the paper-used figures plus the minimal `.npz` traces needed to replot them.
- exploratory sweeps, failed reruns, duplicate experiment roots, and non-paper assets are intentionally excluded.

## Repository Layout

```text
HaM_World/
├── launch.py
├── environment.yml
├── requirements.txt
├── hamworld/                  # canonical HaM-World package
├── baselines/                 # DreamerV3 / TD-MPC2 / PPO / SAC
├── runs/
│   └── main/                  # full raw runs for the 5-algorithm main comparison
├── results/
│   ├── main/                  # main paper curves, tables, manifests
│   ├── ood/                   # compact OOD result exports
│   ├── ablation/              # compact ablation result exports
│   ├── mechanism/             # mechanism figures + minimal trace bundles
│   └── appendix/              # appendix-only assets
└── scripts/                   # rebuild / plot / analysis utilities
```

Useful paths:

- main result table: [`results/main/tables/paper_main_results.csv`](./results/main/tables/paper_main_results.csv)
- main learning curves: [`results/main/figures/main_results_curves_assembled.pdf`](./results/main/figures/main_results_curves_assembled.pdf)
- OOD summary table: [`results/ood/tables/ood_condition_retention.csv`](./results/ood/tables/ood_condition_retention.csv)
- ablation table: [`results/ablation/tables/paper_ablation_core_table.csv`](./results/ablation/tables/paper_ablation_core_table.csv)

## Installation

Recommended: `uv`

```bash
cd HaM_World
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Alternative: conda

```bash
cd HaM_World
conda env create -f environment.yml
conda activate ham_world
```

Core dependencies:

- `torch`
- `dm-control`
- `gymnasium[mujoco]`
- `gymnasium-robotics`
- `matplotlib`
- `PyYAML`

## Quick Start

List supported algorithms, presets, and configs:

```bash
python launch.py list
```

Dry-run one training plan:

```bash
python launch.py train \
  --algo hamworld \
  --preset compare_dmcontrol \
  --seed 7 \
  --output-root outputs \
  --dry-run
```

Train HaM-World on Finger + Reacher:

```bash
python launch.py train \
  --algo hamworld \
  --preset finger_reacher \
  --seed 7 \
  --output-root outputs
```

Train all five kept algorithms for one preset:

```bash
python launch.py train \
  --algo all \
  --preset compare_dmcontrol \
  --seed 7 \
  --output-root outputs
```

Shell entrypoints are also provided:

```bash
PYTHON_BIN=/path/to/python bash scripts/train/run_hamworld_paper_4tasks.sh
PYTHON_BIN=/path/to/python SEED=7 bash scripts/train/run_all_paper_5alg_4tasks.sh
```

## Rebuilding Paper Assets

Typical result rebuild commands:

```bash
python scripts/rebuild_main_return_table.py
python scripts/rebuild_long_horizon_table.py
python scripts/replot_main_curves.py
python scripts/rebuild_ablation_core_table.py
python scripts/rebuild_ood_summaries.py
python scripts/rebuild_rollout_overview.py
```

Mechanism-related replots:

```bash
python scripts/replot_h_freerun.py
python scripts/replot_phase_portrait.py
python scripts/replot_pqc.py
```

Notes:

- `results/main/` is the primary supported rebuild target.
- OOD and ablation are shipped as compact result exports rather than full raw campaigns.
- path-bearing manifests in this repository use **repository-relative paths**, so the repository can be moved without manually fixing paths.

## About `launch.py`

`launch.py` is a **convenience wrapper**, not the algorithm implementation itself.

Its role is to provide one unified CLI for the 5 kept algorithms:

- selects the algorithm package to invoke
- selects a preset such as `compare_dmcontrol` or `finger_reacher`
- resolves the correct config file for each algorithm
- applies seed / output-root / dotted overrides
- expands a multi-task config into per-task runs
- optionally resumes from a checkpoint

In other words, `launch.py` is a thin orchestration layer over:

- `hamworld/train.py`
- `baselines/dreamerv3/train.py`
- `baselines/tdmpc2/train.py`
- `baselines/ppo/train.py`
- `baselines/sac/train.py`

It is not strictly required for the core algorithm code, but it is useful in this paper repository because it gives reviewers one stable entrypoint instead of five different training interfaces.

## Data Policy

This repository intentionally separates raw runs from paper-facing assets:

- `runs/main/`: full raw run trees for the main comparison
- `results/main/`: paper-facing summaries derived from `runs/main/`
- `results/ood/`: OOD result exports only
- `results/ablation/`: ablation result exports only
- `results/mechanism/`: kept figures and minimal trace bundles for paper diagnostics

If someone wants to replay the entire original OOD or ablation training campaigns from checkpoints, this repository is **not** the full historical workspace; it is the cleaned paper artifact.

## Naming

- paper name: `HaM-World`
- repository name: `HaM_World`
- Python package name: `hamworld`

This split is intentional: the paper-facing name stays readable, while the import path stays simple and stable.

## Citation

Citation metadata can be added here after the anonymous review process.
