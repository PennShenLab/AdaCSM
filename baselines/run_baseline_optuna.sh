#!/bin/bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

CONDA_ENV="${CONDA_ENV:-adacsm}"
DATASET="${DATASET:-flchain}"
TUNE_TRIALS="${TUNE_TRIALS:-50}"
TUNE_EPOCHS="${TUNE_EPOCHS:-2000}"
PATIENCE="${PATIENCE:-200}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SEEDS="${SEEDS:-42,73,666,777,1009}"
SELECT_METRIC="${SELECT_METRIC:-test_cindex}"
OUT_BASE="${OUT_BASE:-results/optuna_${DATASET}_baseline}"
STUDY_NAME="${STUDY_NAME:-dcsm_${DATASET}_baseline}"
STORAGE="${STORAGE:-sqlite:///results/optuna_${DATASET}_baseline.db}"

exec conda run -n "$CONDA_ENV" python baselines/tune_baseline_optuna.py \
  --dataset "$DATASET" \
  --tune_trials "$TUNE_TRIALS" \
  --tune_epochs "$TUNE_EPOCHS" \
  --patience "$PATIENCE" \
  --cuda_device "$CUDA_DEVICE" \
  --seeds "$SEEDS" \
  --select_metric "$SELECT_METRIC" \
  --out_base "$OUT_BASE" \
  --study_name "$STUDY_NAME" \
  --storage "$STORAGE" \
  "$@"
