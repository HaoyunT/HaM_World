#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FINAL_ROOT}/outputs}"
ALGORITHMS="${ALGORITHMS:-hamworld dreamerv3 tdmpc2 ppo sac}"
SEED_ARG=()
EXTRA_ARGS=("$@")

if [[ -n "${SEED:-}" ]]; then
  SEED_ARG=(--seed "$SEED")
fi

for algo in ${ALGORITHMS}; do
  echo "[run] ${algo} compare_dmcontrol -> ${OUTPUT_ROOT}"
  "${PYTHON_BIN}" "${FINAL_ROOT}/launch.py" train \
    --algo "${algo}" \
    --preset compare_dmcontrol \
    --output-root "${OUTPUT_ROOT}" \
    "${SEED_ARG[@]}" \
    "${EXTRA_ARGS[@]}"

  echo "[run] ${algo} finger_reacher -> ${OUTPUT_ROOT}"
  "${PYTHON_BIN}" "${FINAL_ROOT}/launch.py" train \
    --algo "${algo}" \
    --preset finger_reacher \
    --output-root "${OUTPUT_ROOT}" \
    "${SEED_ARG[@]}" \
    "${EXTRA_ARGS[@]}"
done
