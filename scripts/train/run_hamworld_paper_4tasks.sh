#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FINAL_ROOT}/outputs}"
SEED_ARG=()
EXTRA_ARGS=("$@")

if [[ -n "${SEED:-}" ]]; then
  SEED_ARG=(--seed "$SEED")
fi

echo "[run] HaM-World compare_dmcontrol -> ${OUTPUT_ROOT}"
"${PYTHON_BIN}" "${FINAL_ROOT}/launch.py" train \
  --algo hamworld \
  --preset compare_dmcontrol \
  --output-root "${OUTPUT_ROOT}" \
  "${SEED_ARG[@]}" \
  "${EXTRA_ARGS[@]}"

echo "[run] HaM-World finger_reacher -> ${OUTPUT_ROOT}"
"${PYTHON_BIN}" "${FINAL_ROOT}/launch.py" train \
  --algo hamworld \
  --preset finger_reacher \
  --output-root "${OUTPUT_ROOT}" \
  "${SEED_ARG[@]}" \
  "${EXTRA_ARGS[@]}"
