#!/bin/bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

OUT_DIR="${OUT_DIR:-logs/paper_baselines}"
CONDA_ENV="${CONDA_ENV:-adacsm}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

exec conda run -n "$CONDA_ENV" python baselines/run_baselines.py \
  --out-dir "$OUT_DIR" \
  --cuda-device "$CUDA_DEVICE" \
  "$@"
