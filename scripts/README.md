# scripts

Paper-facing utility scripts kept with the `HaM_World/` repository.

Use a local Python environment with `torch`, `dm_control`, and plotting dependencies available.

Scope:
- supported raw-run inputs are centered on `runs/main/`
- supported paper-facing outputs live under `results/`
- some scripts still expect renderer dependencies for rollout images, but the kept mechanism trace bundles are part of this artifact

Primary scripts in this package:
- `rebuild_main_return_table.py`
- `rebuild_long_horizon_table.py`
- `replot_main_curves.py`
- `rebuild_ablation_core_table.py`
- `rebuild_ood_summaries.py`
- `train/run_all_paper_5alg_4tasks.sh`
- `train/run_hamworld_paper_4tasks.sh`

Examples:

```bash
python scripts/replot_main_curves.py
python scripts/rebuild_main_return_table.py
python scripts/rebuild_long_horizon_table.py
```
