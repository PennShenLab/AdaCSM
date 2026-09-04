#!/bin/bash

# Bash script to run dense MoE experiments (all experts, no top-k)
# Loops over n (number of experts) values: 1, 2, 4, 8, 16, 32
# Logs C-index and log-rank metrics with mean and variance

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
export MPLBACKEND="${MPLBACKEND:-Agg}"
export ADACSM_NO_SHOW="${ADACSM_NO_SHOW:-1}"

# Dataset options: support, flchain, PBC, FRAMINGHAM, sim
# Override without editing file:
#   DATASET=PBC bash src/run_dense_experiments.sh
DATASET="${DATASET:-FRAMINGHAM}"

CUDA_DEVICES=(0 1 2 3)
DEVICE_INDEX=1

# Hyperparameter tuning (set to 1 to enable)
TUNE_HYPERPARAMS=0

# Hyperparameter ranges (only used if TUNE_HYPERPARAMS=1)
# Standard DCSM hyperparameters
LR_VALUES=(0.0001 0.0005 0.001)
DISCOUNT_VALUES=(0.5 0.75 1)
LAYERS_VALUES=("[50]" "[100]")

# MoE-specific hyperparameters
WEIGHT_DECAY_VALUES=(0 0.0001 0.001)
MOE_DROPOUT_VALUES=(0 0.1 0.2)
LOAD_BALANCE_LAMBDA_VALUES=(0 0.001 0.01)
GATE_TEMPERATURE_VALUES=(0.5 1.0 2.0)
# Optional: routing noise (usually 0 or small values)
ROUTING_NOISE_STD_VALUES=(0)

# Fixed hyperparameters (used if TUNE_HYPERPARAMS=0)
# Based on DCSM's best hyperparameters for each dataset

# Support dataset
# FIXED_LR=0.0001
# FIXED_DISCOUNT=1
# FIXED_LAYERS="[50]"

#Flchain
# FIXED_LR=0.001
# FIXED_DISCOUNT=0.75
# FIXED_LAYERS="[50]"

# PBC
# FIXED_LR=0.001
# FIXED_DISCOUNT=1
# FIXED_LAYERS="[50]"

# Framingham
FIXED_LR=0.001
FIXED_DISCOUNT=0.5
FIXED_LAYERS="[50]"

# MoE-specific fixed hyperparameters
FIXED_WEIGHT_DECAY=0.0001
FIXED_MOE_DROPOUT=0.1
FIXED_GATE_DROPOUT=0.0
FIXED_LOAD_BALANCE_LAMBDA=0.01
FIXED_GATE_TEMPERATURE=1.0
FIXED_ROUTING_NOISE_STD=0.0

FIXED_ITERS=2000
FIXED_EARLY_STOPPING=True
FIXED_PATIENCE=200

# Create dataset-specific tmp directory to allow parallel runs
TMP_DIR="./tmp_${DATASET}"
mkdir -p "$TMP_DIR"
# Clean any previous output files to avoid parsing stale data
rm -f "$TMP_DIR"/output_*.txt "$TMP_DIR"/perseed_*.tmp "$TMP_DIR"/summary_*.tmp "$TMP_DIR"/debug_*.log

# Output log file
LOG_FILE="dense_experiments_$(date +%Y%m%d_%H%M%S).log"

# Initialize log file
echo "Dense MoE Experiments Log (all experts, no top-k)" > "$LOG_FILE"
echo "=================================================" >> "$LOG_FILE"
echo "Dataset: $DATASET" >> "$LOG_FILE"
echo "Start time: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

if [ $TUNE_HYPERPARAMS -eq 1 ]; then
    echo "Mode: HYPERPARAMETER TUNING (grid search)" >> "$LOG_FILE"
    echo "Learning rates: ${LR_VALUES[@]}" >> "$LOG_FILE"
    echo "Discounts: ${DISCOUNT_VALUES[@]}" >> "$LOG_FILE"
    echo "Layer configs: ${LAYERS_VALUES[@]}" >> "$LOG_FILE"
    echo "Weight decay: ${WEIGHT_DECAY_VALUES[@]}" >> "$LOG_FILE"
    echo "MoE dropout: ${MOE_DROPOUT_VALUES[@]}" >> "$LOG_FILE"
    echo "Load balance lambda: ${LOAD_BALANCE_LAMBDA_VALUES[@]}" >> "$LOG_FILE"
    echo "Gate temperature: ${GATE_TEMPERATURE_VALUES[@]}" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    echo "Per-seed detailed results:" >> "$LOG_FILE"
    echo "lr,discount,layers,n_experts,weight_decay,moe_dropout,load_balance_lambda,gate_temp,seed,best_epoch,c_index_train,c_index_val,c_index_test,logrank,rae_nc,rae_c,cal,cluster0,cluster1,collapsed" >> "$LOG_FILE"
else
    echo "Mode: FIXED HYPERPARAMETERS" >> "$LOG_FILE"
    echo "Learning rate: $FIXED_LR" >> "$LOG_FILE"
    echo "Discount: $FIXED_DISCOUNT" >> "$LOG_FILE"
    echo "Layers: $FIXED_LAYERS" >> "$LOG_FILE"
    echo "Weight decay: $FIXED_WEIGHT_DECAY" >> "$LOG_FILE"
    echo "MoE dropout: $FIXED_MOE_DROPOUT" >> "$LOG_FILE"
    echo "Load balance lambda: $FIXED_LOAD_BALANCE_LAMBDA" >> "$LOG_FILE"
    echo "Gate temperature: $FIXED_GATE_TEMPERATURE" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    echo "Per-seed detailed results:" >> "$LOG_FILE"
    echo "n_experts,seed,best_epoch,c_index_train,c_index_val,c_index_test,logrank,rae_nc,rae_c,cal,cluster0,cluster1,collapsed" >> "$LOG_FILE"
fi
echo "" >> "$LOG_FILE"

# Array for n values (number of experts)
n_values=(1 2 4 8 16 32)
# n_values=(1 2) # for testing

# Array to track background processes
declare -a pids
declare -a configs
job_count=0

if [ $TUNE_HYPERPARAMS -eq 1 ]; then
    # Hyperparameter tuning loop - parallel across hyperparams and n values
    for lr in "${LR_VALUES[@]}"; do
        for discount in "${DISCOUNT_VALUES[@]}"; do
            for layers in "${LAYERS_VALUES[@]}"; do
                for wd in "${WEIGHT_DECAY_VALUES[@]}"; do
                    for moe_drop in "${MOE_DROPOUT_VALUES[@]}"; do
                        for lb_lambda in "${LOAD_BALANCE_LAMBDA_VALUES[@]}"; do
                            for gate_temp in "${GATE_TEMPERATURE_VALUES[@]}"; do
                                for n in "${n_values[@]}"; do
                                    layers_safe=$(echo "$layers" | tr -d '[], ')
                                    GPU=$((job_count % ${#CUDA_DEVICES[@]}))
                                    
                                    echo "Starting dense experiment: lr=$lr, discount=$discount, layers=$layers, wd=$wd, moe_drop=$moe_drop, lb=$lb_lambda, temp=$gate_temp, n=$n on GPU $GPU (job $((job_count+1)))"
                                    
                                    (
                                        cd "$PROJECT_ROOT"
                                        python -u main.py \
                                            --dataset $DATASET \
                                            --cuda_device ${CUDA_DEVICES[$GPU]} \
                                            --learning_rate $lr \
                                            --discount $discount \
                                            --layers "$layers" \
                                            --weight_decay $wd \
                                            --moe_dropout $moe_drop \
                                            --load_balance_lambda $lb_lambda \
                                            --gate_temperature $gate_temp \
                                            --iters $FIXED_ITERS \
                                            --early_stopping $FIXED_EARLY_STOPPING \
                                            --patience $FIXED_PATIENCE \
                                            --use_moe \
                                            --num_experts $n > "$TMP_DIR/output_dense_${lr}_${discount}_${layers_safe}_${wd}_${moe_drop}_${lb_lambda}_${gate_temp}_${n}.txt" 2>&1
                                    ) &
                                    
                                    pids+=($!)
                                    configs+=("lr=$lr discount=$discount layers=$layers wd=$wd moe_drop=$moe_drop lb=$lb_lambda temp=$gate_temp n=$n")
                                    ((job_count++))
                                done
                            done
                        done
                    done
                done
            done
        done
    done
else
    # Fixed hyperparameters - only loop over n values
    for i in "${!n_values[@]}"; do
        n=${n_values[$i]}
        
        GPU=$((i % ${#CUDA_DEVICES[@]}))
        
        echo "Starting dense experiment: n_experts=$n on GPU $GPU (job $((i+1))/${#n_values[@]})"
        
        (
            cd "$PROJECT_ROOT"
            python main.py \
                --dataset $DATASET \
                --cuda_device $GPU \
                --learning_rate $FIXED_LR \
                --discount $FIXED_DISCOUNT \
                --layers "$FIXED_LAYERS" \
                --weight_decay $FIXED_WEIGHT_DECAY \
                --moe_dropout $FIXED_MOE_DROPOUT \
                --gate_dropout $FIXED_GATE_DROPOUT \
                --load_balance_lambda $FIXED_LOAD_BALANCE_LAMBDA \
                --gate_temperature $FIXED_GATE_TEMPERATURE \
                --routing_noise_std $FIXED_ROUTING_NOISE_STD \
                --iters $FIXED_ITERS \
                --early_stopping $FIXED_EARLY_STOPPING \
                --patience $FIXED_PATIENCE \
                --use_moe \
                --num_experts $n > "$TMP_DIR/output_${n}.txt" 2>&1
        
        echo "DEBUG: Output file created for n=$n" >> $TMP_DIR/debug_${n}.log
        ls -lh "$TMP_DIR/output_${n}.txt" >> $TMP_DIR/debug_${n}.log
        
        # Save output for parsing
        OUTPUT_FILE="$TMP_DIR/output_${n}.txt"
        
        # Extract params from first occurrence
        if [ $n -eq 1 ]; then
            grep "^param:" "$OUTPUT_FILE" | head -1 > "$TMP_DIR/params.tmp"
        fi
        
        # Use Python to parse - more reliable than awk state machine
        echo "DEBUG: Starting Python parsing for n=$n" >> $TMP_DIR/debug_${n}.log
        python3 << PYEOF > "$TMP_DIR/perseed_${n}.tmp" 2>> $TMP_DIR/debug_${n}.log
import re

n = $n
with open("$TMP_DIR/output_${n}.txt", "r") as f:
    lines = f.readlines()

seed = None
c_idx_train = c_idx_val = c_idx_test = rae_nc = rae_c = cal = cluster0 = cluster1 = logrank = best_epoch = None

for line in lines:
    # Extract seed
    m = re.search(r'seed (\d+)', line)
    if m:
        # If we have data from previous seed, output it first
        if seed and c_idx_test:
            # Compute collapsed: 1 if either cluster is 0, else 0
            collapsed = 1 if (cluster0 == '0' or cluster1 == '0') else 0
            print(f"{n},{seed},{best_epoch},{c_idx_train},{c_idx_val},{c_idx_test},{logrank},{rae_nc},{rae_c},{cal},{cluster0},{cluster1},{collapsed}")
        # Reset for new seed
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
    print(f"{n},{seed},{best_epoch},{c_idx_train},{c_idx_val},{c_idx_test},{logrank},{rae_nc},{rae_c},{cal},{cluster0},{cluster1},{collapsed}")
PYEOF
        
        echo "DEBUG: Python parsing completed for n=$n, exit code=$?" >> $TMP_DIR/debug_${n}.log
        
        # Ensure file is flushed
        sync
        
        ls -lh "$TMP_DIR/perseed_${n}.tmp" >> $TMP_DIR/debug_${n}.log 2>&1
        
        # Extract summary statistics with awk
        awk '
            /C Index results/ { getline; cindex_line = $0 }
            /logrank results/ { getline; logrank_line = $0 }
            END {
                if (cindex_line != "") {
                    match(cindex_line, /DCSM:([0-9.]+)±([0-9.]+)/, arr)
                    c_mean = arr[1]
                    c_std = arr[2]
                } else {
                    c_mean = "NA"
                    c_std = "NA"
                }
                if (logrank_line != "") {
                    match(logrank_line, /DCSM:([0-9.]+)±([0-9.]+)/, arr)
                    lr_mean = arr[1]
                    lr_std = arr[2]
                } else {
                    lr_mean = "NA"
                    lr_std = "NA"
                }
                print "'$n'," c_mean "," c_std "," lr_mean "," lr_std
            }
        ' "$OUTPUT_FILE" > "$TMP_DIR/summary_${n}.tmp"
        
        # Clean up output file
        rm -f "$OUTPUT_FILE"
    ) &
    
    # Store PID
    pids+=($!)
    configs+=("n=$n GPU=$GPU")
    done
fi

echo ""
echo "Started ${#pids[@]} parallel experiments. Waiting for completion..."
echo ""

# Wait for all background jobs to complete
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

if [ $TUNE_HYPERPARAMS -eq 1 ]; then
    # Parse tuning mode results
    for lr in "${LR_VALUES[@]}"; do
        for discount in "${DISCOUNT_VALUES[@]}"; do
            for layers in "${LAYERS_VALUES[@]}"; do
                for wd in "${WEIGHT_DECAY_VALUES[@]}"; do
                    for moe_drop in "${MOE_DROPOUT_VALUES[@]}"; do
                        for lb_lambda in "${LOAD_BALANCE_LAMBDA_VALUES[@]}"; do
                            for gate_temp in "${GATE_TEMPERATURE_VALUES[@]}"; do
                                for n in "${n_values[@]}"; do
                                    layers_safe=$(echo "$layers" | tr -d '[], ')
                                    OUTPUT_FILE="$TMP_DIR/output_dense_${lr}_${discount}_${layers_safe}_${wd}_${moe_drop}_${lb_lambda}_${gate_temp}_${n}.txt"
                                    
                                    if [ -f "$OUTPUT_FILE" ]; then
                                        python3 << PYEOF >> "$LOG_FILE" 2>> $TMP_DIR/debug_dense_tuning.log
import re

lr = "$lr"
discount = "$discount"
wd = "$wd"
moe_drop = "$moe_drop"
lb_lambda = "$lb_lambda"
gate_temp = "$gate_temp"
n = $n
with open("$OUTPUT_FILE", "r") as f:
    lines = f.readlines()

seed = None
c_idx_train = c_idx_val = c_idx_test = rae_nc = rae_c = cal = cluster0 = cluster1 = logrank = best_epoch = None

for line in lines:
    m = re.search(r'seed (\d+)', line)
    if m:
        if seed and c_idx_test:
            collapsed = 1 if (cluster0 == '0' or cluster1 == '0') else 0
            print(f"{lr},{discount},$layers,{n},{wd},{moe_drop},{lb_lambda},{gate_temp},{seed},{best_epoch},{c_idx_train},{c_idx_val},{c_idx_test},{logrank},{rae_nc},{rae_c},{cal},{cluster0},{cluster1},{collapsed}")
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
    print(f"{lr},{discount},$layers,{n},{wd},{moe_drop},{lb_lambda},{gate_temp},{seed},{best_epoch},{c_idx_train},{c_idx_val},{c_idx_test},{logrank},{rae_nc},{rae_c},{cal},{cluster0},{cluster1},{collapsed}")
PYEOF
                                        rm -f "$OUTPUT_FILE"
                                    else
                                        echo "DEBUG: Missing $OUTPUT_FILE" >> $TMP_DIR/debug_dense_tuning.log
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done
else
    # Parse fixed hyperparameter mode results
    for n in "${n_values[@]}"; do
        if [ -f "$TMP_DIR/perseed_${n}.tmp" ]; then
            echo "DEBUG: Found $TMP_DIR/perseed_${n}.tmp"
            cat "$TMP_DIR/perseed_${n}.tmp" >> "$LOG_FILE"
            rm -f "$TMP_DIR/perseed_${n}.tmp"
        else
            echo "DEBUG: Missing $TMP_DIR/perseed_${n}.tmp"
        fi
    done
fi

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
echo "n_experts,c_index_mean,c_index_std,logrank_mean,logrank_std" >> "$LOG_FILE"

# Collect and write summary results
echo "DEBUG: Collecting summary results..."
for n in "${n_values[@]}"; do
    if [ -f "$TMP_DIR/summary_${n}.tmp" ]; then
        echo "DEBUG: Found $TMP_DIR/summary_${n}.tmp"
        RESULT=$(cat "$TMP_DIR/summary_${n}.tmp")
        echo "$RESULT" >> "$LOG_FILE"
        echo "  $RESULT"
        rm -f "$TMP_DIR/summary_${n}.tmp"
    else
        echo "DEBUG: Missing $TMP_DIR/summary_${n}.tmp"
    fi
done

# Final cleanup to ensure no temp files remain
# rm -f $TMP_DIR/perseed_*.tmp $TMP_DIR/summary_*.tmp $TMP_DIR/params.tmp $TMP_DIR/output_*.txt 2>/dev/null
echo "DEBUG: Keeping temp files for inspection"

echo "=================================================" >> "$LOG_FILE"
echo "End time: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Find best hyperparameters if in tuning mode
if [ $TUNE_HYPERPARAMS -eq 1 ]; then
    echo "" >> "$LOG_FILE"
    echo "=================================================" >> "$LOG_FILE"
    echo "BEST HYPERPARAMETER SELECTION (per n_experts)" >> "$LOG_FILE"
    echo "=================================================" >> "$LOG_FILE"
    
    python3 << BESTPYEOF >> "$LOG_FILE"
import pandas as pd
import sys

# Read the log file to extract results
log_content = open("$LOG_FILE", "r").read()

# Extract the per-seed results section
lines = log_content.split('\n')
results_section_start = None
for i, line in enumerate(lines):
    if 'Per-seed detailed results:' in line:
        results_section_start = i + 2
        break

if results_section_start is None:
    print("Could not find results section")
    sys.exit(1)

# Parse CSV data
results_lines = []
for i in range(results_section_start, len(lines)):
    line = lines[i].strip()
    if not line:
        continue
    if line.startswith('='):
        break
    if line.startswith('lr,discount,layers') or line.startswith('n_experts,seed'):
        continue
    if line and not line.startswith('#'):
        results_lines.append(line)

if not results_lines:
    print("No results found")
    sys.exit(1)

# Create dataframe
data = []
for line in results_lines:
    parts = line.split(',')
    if len(parts) >= 13:  # Updated for new hyperparameters
        try:
            data.append({
                'lr': parts[0],
                'discount': parts[1],
                'layers': parts[2],
                'n_experts': int(parts[3]),
                'weight_decay': parts[4],
                'moe_dropout': parts[5],
                'load_balance_lambda': parts[6],
                'gate_temp': parts[7],
                'seed': int(parts[8]),
                'best_epoch': parts[9],
                'c_index_train': float(parts[10]),
                'c_index_val': float(parts[11]),
                'c_index_test': float(parts[12]),
                'logrank': float(parts[13]) if parts[13] != 'NA' else None,
            })
        except Exception as e:
            pass

df = pd.DataFrame(data)

if len(df) > 0:
    # Group by n_experts and hyperparameters
    for n in sorted(df['n_experts'].unique()):
        df_n = df[df['n_experts'] == n]
        
        summary_flat = df_n.groupby(['lr', 'discount', 'layers', 'weight_decay', 'moe_dropout', 
                                      'load_balance_lambda', 'gate_temp']).apply(
            lambda x: pd.Series({
                'c_val_mean': x['c_index_val'].mean(),
                'c_val_std': x['c_index_val'].std(),
                'c_test_mean': x['c_index_test'].mean(),
                'c_test_std': x['c_index_test'].std(),
                'logrank_mean': x['logrank'].mean(),
                'logrank_std': x['logrank'].std(),
                'n_seeds': len(x)
            })
        ).reset_index()
        
        # Sort by validation C-index and logrank
        summary_flat = summary_flat.sort_values(['c_val_mean', 'logrank_mean'], 
                                                ascending=[False, True])
        
        best = summary_flat.iloc[0]
        
        print(f"\n{'='*140}")
        print(f"n_experts = {n}")
        print(f"{'='*140}")
        print(f"\nHyperparameter Search Results (Top 10):")
        print(f"{'LR':<8} {'Disc':<6} {'Layers':<10} {'WD':<8} {'MoE_Drop':<10} {'LB_λ':<8} {'Temp':<6} {'Val C-idx':<15} {'Test C-idx':<15} {'Logrank':<15}")
        print("-" * 130)
        for _, row in summary_flat.head(10).iterrows():
            print(f"{row['lr']:<8} {row['discount']:<6} {str(row['layers']):<10} "
                  f"{row['weight_decay']:<8} {row['moe_dropout']:<10} {row['load_balance_lambda']:<8} {row['gate_temp']:<6} "
                  f"{row['c_val_mean']:.4f}±{row['c_val_std']:.4f}  "
                  f"{row['c_test_mean']:.4f}±{row['c_test_std']:.4f}  "
                  f"{row['logrank_mean']:.4f}±{row['logrank_std']:.4f}")
        
        print(f"\nBEST: lr={best['lr']}, discount={best['discount']}, layers={best['layers']}, "
              f"wd={best['weight_decay']}, moe_drop={best['moe_dropout']}, "
              f"lb_lambda={best['load_balance_lambda']}, gate_temp={best['gate_temp']}")
        print(f"  Val C-idx: {best['c_val_mean']:.4f}±{best['c_val_std']:.4f}, "
              f"Test C-idx: {best['c_test_mean']:.4f}±{best['c_test_std']:.4f}, "
              f"Logrank: {best['logrank_mean']:.2f}±{best['logrank_std']:.2f}")
BESTPYEOF
fi

echo "=================================================" 
echo "Experiment completed!"
echo "Results saved to: $LOG_FILE"
echo ""
echo "To view results:"
echo "  cat $LOG_FILE"
