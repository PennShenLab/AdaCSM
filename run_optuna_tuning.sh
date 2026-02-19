#!/bin/bash

# Optuna hyperparameter tuning for DCSM with MoE
# Intelligently searches hyperparameter space using Tree-structured Parzen Estimator (TPE)

# Dataset to tune
DATASET="FRAMINGHAM"  # Options: support, flchain, PBC, FRAMINGHAM, adni

# Tuning settings
TUNE_TRIALS=100       # Number of Optuna trials (100-200 recommended)
TUNE_EPOCHS=2000      # Max epochs per trial
PATIENCE=200           # Early stopping patience (reduced to stop bad configs faster)
PROGRESS_EVERY=1      # Print progress every N epochs (0 disables)

# GPU settings
GPU_DEVICES="0,1,2,3"
TRIALS_PER_GPU=10     # Concurrent trials per GPU (1-2 recommended)

# Optimization settings
SELECT_METRIC="val_cindex"  # Options: val_cindex, test_cindex, logrank

# Output settings
OUT_BASE="results/optuna_${DATASET}_moe"
STUDY_NAME="dcsm_${DATASET}_moe"
STORAGE="sqlite:///results/optuna_${DATASET}.db"

# Create results directory
mkdir -p results

echo "============================================"
echo "DCSM Optuna Hyperparameter Tuning"
echo "============================================"
echo "Dataset: $DATASET"
echo "Trials: $TUNE_TRIALS"
echo "Metric: $SELECT_METRIC"
echo "GPUs: $GPU_DEVICES"
echo "Parallel jobs: $(($(echo $GPU_DEVICES | tr ',' ' ' | wc -w) * TRIALS_PER_GPU))"
echo "============================================"
echo ""

# Check if optuna is installed
if ! python3 -c "import optuna" 2>/dev/null; then
    echo "ERROR: Optuna not installed"
    echo "Install with: pip install optuna"
    exit 1
fi

# Create log file
LOG_FILE="results/optuna_${DATASET}_$(date +%Y%m%d_%H%M%S).log"
echo "Log file: $LOG_FILE"
echo ""

# Run tuning (output to both terminal and log file)
python3 -u tune_dcsm_optuna.py \
    --dataset $DATASET \
    --tune_trials $TUNE_TRIALS \
    --tune_epochs $TUNE_EPOCHS \
    --patience $PATIENCE \
    --progress_every $PROGRESS_EVERY \
    --gpu_devices "$GPU_DEVICES" \
    --trials_per_gpu $TRIALS_PER_GPU \
    --out_base $OUT_BASE \
    --storage $STORAGE \
    --study_name $STUDY_NAME \
    --select_metric $SELECT_METRIC \
    --prune_trials 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================"
echo "Tuning complete!"
echo "Results saved to:"
echo "  - Log file: $LOG_FILE"
echo "  - Best trial: ${OUT_BASE}_best_trial.json"
echo "  - All trials: ${OUT_BASE}_all_trials.csv"
echo "  - Database: $STORAGE"
echo "============================================"
echo ""
echo "To use best hyperparameters in run_dense_experiments.sh,"
echo "extract them from ${OUT_BASE}_best_trial.json"
