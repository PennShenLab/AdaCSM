#!/bin/bash

# Bash script to run top-k routing experiments with different numbers of experts
# Loops over k (top-k values) and n (number of experts)
# Only runs experiments where k <= n
# Logs detailed per-seed metrics
# 
# Optional hyperparameter tuning: Set TUNE_HYPERPARAMS=1 to grid search over:
#   - learning_rate: [0.001, 0.0001]
#   - discount: [0.5, 0.75, 1]
#   - layers: [[50], [50,50]]

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset options: support, flchain, PBC, FRAMINGHAM, adni
DATASET="support"
# DATASET="flchain"
# DATASET="PBC"
# DATASET="FRAMINGHAM"
# DATASET="adni"

CUDA_DEVICES=(0 1 2 3)
DEVICE_INDEX=0

# Hyperparameter tuning (set to 1 to enable)
TUNE_HYPERPARAMS=0

# Hyperparameter ranges (only used if TUNE_HYPERPARAMS=1)
LR_VALUES=(0.001 0.0001)
DISCOUNT_VALUES=(0.5 0.75 1)
LAYERS_VALUES=("[50]" "[50,50]")

# Fixed hyperparameters (used if TUNE_HYPERPARAMS=0)

# Support dataset
FIXED_LR=0.0001
FIXED_DISCOUNT=1
FIXED_LAYERS="[50]"

#Flchain
# FIXED_LR=0.001
# FIXED_DISCOUNT=0.75
# FIXED_LAYERS="[50]"

# PBC
# FIXED_LR=0.001
# FIXED_DISCOUNT=1
# FIXED_LAYERS="[50]"

# Framingham
# FIXED_LR=0.001
# FIXED_DISCOUNT=0.5
# FIXED_LAYERS="[50]"

FIXED_ITERS=2000
FIXED_EARLY_STOPPING=True
FIXED_PATIENCE=200

# ============================================================================
# SETUP
# ============================================================================

# Create dataset-specific tmp directory to allow parallel runs
TMP_DIR="./tmp_${DATASET}"
mkdir -p "$TMP_DIR"
# Clean any previous output files to avoid parsing stale data
rm -f "$TMP_DIR"/output_*.txt "$TMP_DIR"/perseed_*.tmp "$TMP_DIR"/summary_*.tmp "$TMP_DIR"/debug_*.log

# Output log file
LOG_FILE="topk_experiments_$(date +%Y%m%d_%H%M%S).log"

# Initialize log file
echo "Top-K Routing Experiments Log" > "$LOG_FILE"
echo "=================================================" >> "$LOG_FILE"
echo "Dataset: $DATASET" >> "$LOG_FILE"
echo "Start time: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

if [ $TUNE_HYPERPARAMS -eq 1 ]; then
    echo "Mode: HYPERPARAMETER TUNING (grid search)" >> "$LOG_FILE"
    echo "Learning rates: ${LR_VALUES[@]}" >> "$LOG_FILE"
    echo "Discounts: ${DISCOUNT_VALUES[@]}" >> "$LOG_FILE"
    echo "Layer configs: ${LAYERS_VALUES[@]}" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    echo "Per-seed detailed results:" >> "$LOG_FILE"
    echo "lr,discount,layers,n_experts,top_k,seed,best_epoch,c_index_train,c_index_val,c_index_test,logrank,rae_nc,rae_c,cal,cluster0,cluster1,collapsed" >> "$LOG_FILE"
else
    echo "Mode: FIXED HYPERPARAMETERS" >> "$LOG_FILE"
    echo "Learning rate: $FIXED_LR" >> "$LOG_FILE"
    echo "Discount: $FIXED_DISCOUNT" >> "$LOG_FILE"
    echo "Layers: $FIXED_LAYERS" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    echo "Per-seed detailed results:" >> "$LOG_FILE"
    echo "n_experts,top_k,seed,best_epoch,c_index_train,c_index_val,c_index_test,logrank,rae_nc,rae_c,cal,cluster0,cluster1,collapsed" >> "$LOG_FILE"
fi
echo "" >> "$LOG_FILE"

# Arrays for k and n values
k_values=(1 2 3 5 8 10)
n_values=(1 2 4 8 16 32)

# Array to track background processes
declare -a pids
declare -a configs

# Main experiment loop
job_count=0

if [ $TUNE_HYPERPARAMS -eq 1 ]; then
    # Loop over hyperparameters, n, and k
    for lr in "${LR_VALUES[@]}"; do
        for discount in "${DISCOUNT_VALUES[@]}"; do
            for layers in "${LAYERS_VALUES[@]}"; do
                for n in "${n_values[@]}"; do
                    for k in "${k_values[@]}"; do
                        if [ $k -le $n ]; then
                            GPU=$((job_count % ${#CUDA_DEVICES[@]}))
                            echo "Starting experiment: lr=$lr, discount=$discount, layers=$layers, n_experts=$n, top_k=$k on GPU $GPU (job $((job_count+1)))"
                            
                            (
                                cd /home/fzhuang/mref-ad/DCSM/DCSM
                                python -u main.py \
                                    --dataset $DATASET \
                                    --cuda_device $GPU \
                                    --learning_rate $lr \
                                    --discount $discount \
                                    --layers "$layers" \
                                    --iters $FIXED_ITERS \
                                    --early_stopping $FIXED_EARLY_STOPPING \
                                    --patience $FIXED_PATIENCE \
                                    --use_moe \
                                    --num_experts $n \
                                    --top_k $k > "$TMP_DIR/output_tuning_${lr}_${discount}_${n}_${k}.txt" 2>&1
                                
                                sleep 1
                                sync
                                
                                # Parse results
                                python3 << PYEOF > "$TMP_DIR/perseed_tuning_${lr}_${discount}_${n}_${k}.tmp" 2>> "$TMP_DIR/debug_tuning_${lr}_${discount}_${n}_${k}.log"
import re

lr = "$lr"
discount = "$discount"
n = $n
k = $k
with open("$TMP_DIR/output_tuning_${lr}_${discount}_${n}_${k}.txt", "r") as f:
    lines = f.readlines()

seed = None
c_idx_train = c_idx_val = c_idx_test = rae_nc = rae_c = cal = cluster0 = cluster1 = logrank = best_epoch = None

for line in lines:
    m = re.search(r'seed (\d+)', line)
    if m:
        if seed and c_idx_test:
            collapsed = 1 if (cluster0 == '0' or cluster1 == '0') else 0
            print(f"{lr},{discount},$layers,{n},{k},{seed},{best_epoch},{c_idx_train},{c_idx_val},{c_idx_test},{logrank},{rae_nc},{rae_c},{cal},{cluster0},{cluster1},{collapsed}")
        seed = m.group(1)
        c_idx_train = c_idx_val = c_idx_test = rae_nc = rae_c = cal = cluster0 = cluster1 = logrank = best_epoch = None
    
    if 'c-index on the training data:' in line:
        c_idx_train = re.search(r': ([\d.]+)', line).group(1)
    elif 'c-index on the validation data:' in line:
        c_idx_val = re.search(r': ([\d.]+)', line).group(1)
    elif 'c-index on the testing data:' in line:
        c_idx_test = re.search(r': ([\d.]+)', line).group(1)
    elif 'DCSM_rae_nc on test data:' in line:
        rae_nc = re.search(r': ([\d.]+)', line).group(1)
    elif 'DCSM_rae_c on test data:' in line:
        rae_c = re.search(r': ([\d.]+)', line).group(1)
    elif 'DCSM_cal on test data:' in line:
        cal = re.search(r': ([\d.]+)', line).group(1)
    elif 'num in cluster 0 is' in line:
        cluster0 = re.search(r'is (\d+)', line).group(1)
    elif 'num in cluster 1 is' in line:
        cluster1 = re.search(r'is (\d+)', line).group(1)
    elif 'Test statistic of test:' in line:
        logrank = re.search(r': ([\d.e+-]+)', line).group(1)
    elif 'best model is chosen from' in line:
        m = re.search(r'from (\d+)th epoch', line)
        if m:
            best_epoch = m.group(1)

if seed and c_idx_test:
    collapsed = 1 if (cluster0 == '0' or cluster1 == '0') else 0
    print(f"{lr},{discount},$layers,{n},{k},{seed},{best_epoch},{c_idx_train},{c_idx_val},{c_idx_test},{logrank},{rae_nc},{rae_c},{cal},{cluster0},{cluster1},{collapsed}")
PYEOF
                                
                                rm -f "$TMP_DIR/output_tuning_${lr}_${discount}_${n}_${k}.txt"
                            ) &
                            
                            pids+=($!)
                            configs+=("lr=$lr discount=$discount layers=$layers n=$n k=$k GPU=$GPU")
                            ((job_count++))
                        fi
                    done
                done
            done
        done
    done
else
    # Loop over n and k only (fixed hyperparameters)
    for n in "${n_values[@]}"; do
        for k in "${k_values[@]}"; do
            if [ $k -le $n ]; then
                GPU=$((job_count % ${#CUDA_DEVICES[@]}))
                echo "Starting top-k experiment: n_experts=$n, top_k=$k on GPU $GPU (job $((job_count+1)))"
                
                # Run experiment in background
                (
                    cd /home/fzhuang/mref-ad/DCSM/DCSM
                    python -u main.py \
                        --dataset $DATASET \
                        --cuda_device $GPU \
                        --learning_rate $FIXED_LR \
                        --discount $FIXED_DISCOUNT \
                        --layers "$FIXED_LAYERS" \
                        --iters $FIXED_ITERS \
                        --early_stopping $FIXED_EARLY_STOPPING \
                        --patience $FIXED_PATIENCE \
                        --use_moe \
                        --num_experts $n \
                        --top_k $k > "$TMP_DIR/output_topk_${n}_${k}.txt" 2>&1
                    
                    sleep 1
                    sync
                    
                    # Parse results
                    python3 << PYEOF > "$TMP_DIR/perseed_topk_${n}_${k}.tmp" 2>> $TMP_DIR/debug_topk_${n}_${k}.log
import re

n = $n
k = $k
with open("$TMP_DIR/output_topk_${n}_${k}.txt", "r") as f:
    lines = f.readlines()

seed = None
c_idx_train = c_idx_val = c_idx_test = rae_nc = rae_c = cal = cluster0 = cluster1 = logrank = best_epoch = None

for line in lines:
    m = re.search(r'seed (\d+)', line)
    if m:
        if seed and c_idx_test:
            collapsed = 1 if (cluster0 == '0' or cluster1 == '0') else 0
            print(f"{n},{k},{seed},{best_epoch},{c_idx_train},{c_idx_val},{c_idx_test},{logrank},{rae_nc},{rae_c},{cal},{cluster0},{cluster1},{collapsed}")
        seed = m.group(1)
        c_idx_train = c_idx_val = c_idx_test = rae_nc = rae_c = cal = cluster0 = cluster1 = logrank = best_epoch = None
    
    # Extract metrics
    if 'c-index on the training data:' in line:
        c_idx_train = re.search(r': ([\d.]+)', line).group(1)
    elif 'c-index on the validation data:' in line:
        c_idx_val = re.search(r': ([\d.]+)', line).group(1)
    elif 'c-index on the testing data:' in line:
        c_idx_test = re.search(r': ([\d.]+)', line).group(1)
    elif 'DCSM_rae_nc on test data:' in line:
        rae_nc = re.search(r': ([\d.]+)', line).group(1)
    elif 'DCSM_rae_c on test data:' in line:
        rae_c = re.search(r': ([\d.]+)', line).group(1)
    elif 'DCSM_cal on test data:' in line:
        cal = re.search(r': ([\d.]+)', line).group(1)
    elif 'num in cluster 0 is' in line:
        cluster0 = re.search(r'is (\d+)', line).group(1)
    elif 'num in cluster 1 is' in line:
        cluster1 = re.search(r'is (\d+)', line).group(1)
    elif 'Test statistic of test:' in line:
        logrank = re.search(r': ([\d.e+-]+)', line).group(1)
    elif 'best model is chosen from' in line:
        m = re.search(r'from (\d+)th epoch', line)
        if m:
            best_epoch = m.group(1)

# Output last seed if we have data
if seed and c_idx_test:
    # Compute collapsed: 1 if either cluster is 0, else 0
    collapsed = 1 if (cluster0 == '0' or cluster1 == '0') else 0
    print(f"{n},{k},{seed},{best_epoch},{c_idx_train},{c_idx_val},{c_idx_test},{logrank},{rae_nc},{rae_c},{cal},{cluster0},{cluster1},{collapsed}")
PYEOF
                
                echo "DEBUG: Python parsing completed for n=$n, k=$k, exit code=$?" >> $TMP_DIR/debug_topk_${n}_${k}.log
                
                # Ensure file is flushed
                sync
                
                ls -lh "$TMP_DIR/perseed_topk_${n}_${k}.tmp" >> $TMP_DIR/debug_topk_${n}_${k}.log 2>&1
                
                # Compute summary statistics from per-seed CSV results
                python3 << STATSEOF > "$TMP_DIR/summary_topk_${n}_${k}.tmp" 2>> $TMP_DIR/debug_topk_${n}_${k}.log
import numpy as np

try:
    with open("$TMP_DIR/perseed_topk_${n}_${k}.tmp", "r") as f:
        lines = f.readlines()
    
    if len(lines) > 0:
        c_indices = []
        logranks = []
        
        for line in lines:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 9:
                    try:
                        c_idx = float(parts[7])  # c_index_test is 8th column (index 7)
                        logrank = float(parts[6])  # logrank is 7th column (index 6)
                        c_indices.append(c_idx)
                        logranks.append(logrank)
                    except:
                        pass
        
        if len(c_indices) > 0:
            c_mean = np.mean(c_indices)
            c_std = np.std(c_indices)
            lr_mean = np.mean(logranks)
            lr_std = np.std(logranks)
        else:
            c_mean = c_std = lr_mean = lr_std = "NA"
    else:
        c_mean = c_std = lr_mean = lr_std = "NA"
    
    print(f"$n,$k,{c_mean},{c_std},{lr_mean},{lr_std}")
except Exception as e:
    print(f"$n,$k,NA,NA,NA,NA")
STATSEOF
            ) &
            
            pids+=($!)
            configs+=("n=$n k=$k GPU=$GPU")
            ((job_count++))
        fi
    done
done
fi

echo ""
echo "Started $job_count parallel experiments. Waiting for completion..."
echo ""

# Wait for all background jobs and report status
failed=0
for i in "${!pids[@]}"; do
    pid=${pids[$i]}
    config=${configs[$i]}
    if wait $pid; then
        echo "✓ Completed: $config"
    else
        echo "✗ Failed: $config"
        ((failed++))
    fi
done

echo ""

# Collect and write per-seed results
echo "DEBUG: Collecting per-seed results..."
for n in "${n_values[@]}"; do
    for k in "${k_values[@]}"; do
        if [ $k -le $n ]; then
            if [ -f "$TMP_DIR/perseed_topk_${n}_${k}.tmp" ]; then
                echo "DEBUG: Found $TMP_DIR/perseed_topk_${n}_${k}.tmp"
                cat "$TMP_DIR/perseed_topk_${n}_${k}.tmp" >> "$LOG_FILE"
                rm -f "$TMP_DIR/perseed_topk_${n}_${k}.tmp"
            else
                echo "DEBUG: Missing $TMP_DIR/perseed_topk_${n}_${k}.tmp"
            fi
        fi
    done
done

# Add parameters section if captured
if [ -f "$TMP_DIR/params.tmp" ]; then
    echo "" >> "$LOG_FILE"
    echo "Model parameters used:" >> "$LOG_FILE"
    cat "$TMP_DIR/params.tmp" >> "$LOG_FILE"
    rm -f "$TMP_DIR/params.tmp"
fi

# Add summary section
echo "" >> "$LOG_FILE"
echo "=================================================" >> "$LOG_FILE"
echo "Summary statistics across seeds:" >> "$LOG_FILE"
echo "n_experts,top_k,c_index_mean,c_index_std,logrank_mean,logrank_std" >> "$LOG_FILE"

# Collect and write summary results
echo "DEBUG: Collecting summary results..."
for n in "${n_values[@]}"; do
    for k in "${k_values[@]}"; do
        if [ $k -le $n ]; then
            if [ -f "$TMP_DIR/summary_topk_${n}_${k}.tmp" ]; then
                echo "DEBUG: Found $TMP_DIR/summary_topk_${n}_${k}.tmp"
                RESULT=$(cat "$TMP_DIR/summary_topk_${n}_${k}.tmp")
                echo "$RESULT" >> "$LOG_FILE"
                echo "  $RESULT"
                rm -f "$TMP_DIR/summary_topk_${n}_${k}.tmp"
            else
                echo "DEBUG: Missing $TMP_DIR/summary_topk_${n}_${k}.tmp"
            fi
        fi
    done
done

echo "=================================================" >> "$LOG_FILE"
echo "End time: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

echo "=================================================" 
echo "Experiment completed!"
echo "Results saved to: $LOG_FILE"
echo ""
echo "To view results:"
echo "  cat $LOG_FILE"
