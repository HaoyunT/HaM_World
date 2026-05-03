# scripts/train

Current shell entrypoints:
- `run_hamworld_paper_4tasks.sh`: sequentially runs HaM-World on the 4 paper tasks
- `run_all_paper_5alg_4tasks.sh`: sequentially runs the kept 5 algorithms on the 4 paper tasks

Examples:

```bash
PYTHON_BIN=/path/to/your/paper-env/bin/python \
  bash scripts/train/run_hamworld_paper_4tasks.sh

PYTHON_BIN=/path/to/your/paper-env/bin/python \
  SEED=7 \
  bash scripts/train/run_all_paper_5alg_4tasks.sh
```
