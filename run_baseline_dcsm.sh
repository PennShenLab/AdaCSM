#!/bin/bash

# Bash script to run baseline DCSM (no MoE)
# Runs with default seeds and logs detailed metrics
#
# Optional hyperparameter tuning: Set TUNE_HYPERPARAMS=1 to grid search over:
#   - learning_rate: [0.001, 0.0001]
#   - discount: [0.5, 0.75, 1]
#   - layers: [[50], [50,50]]

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset options: support, flchain, PBC, FRAMINGHAM, sim
# DATASET="support"
# DATASET="flchain"
# DATASET="PBC"
DATASET="FRAMINGHAM"
# DATASET="adni"

# Hyperparameter tuning (set to 1 to enable)
TUNE_HYPERPARAMS=1

# GPU devices for parallel execution
CUDA_DEVICES=(0 1 2 3)

# Hyperparameter ranges (only used if TUNE_HYPERPARAMS=1)
LR_VALUES=(0.001 0.0001)
DISCOUNT_VALUES=(0.5 0.75 1)
LAYERS_VALUES=("[50]" "[50,50]")

# Fixed hyperparameters (used if TUNE_HYPERPARAMS=0)
FIXED_LR=0.001
FIXED_DISCOUNT=0.5
FIXED_LAYERS="[50]"
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
rm -f "$TMP_DIR"/output_baseline.txt "$TMP_DIR"/perseed_baseline.tmp "$TMP_DIR"/summary_baseline.tmp "$TMP_DIR"/debug_baseline.log

# Output log file
LOG_FILE="baseline_dcsm_$(date +%Y%m%d_%H%M%S).log"

# Initialize log file
echo "Baseline DCSM Experiments Log (no MoE)" > "$LOG_FILE"
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
    echo "lr,discount,layers,seed,best_epoch,c_index_train,c_index_val,c_index_test,logrank,rae_nc,rae_c,cal,cluster0,cluster1,collapsed" >> "$LOG_FILE"
else
    echo "Mode: FIXED HYPERPARAMETERS" >> "$LOG_FILE"
    echo "Learning rate: $FIXED_LR" >> "$LOG_FILE"
    echo "Discount: $FIXED_DISCOUNT" >> "$LOG_FILE"
    echo "Layers: $FIXED_LAYERS" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    echo "Per-seed detailed results:" >> "$LOG_FILE"
    echo "seed,best_epoch,c_index_train,c_index_val,c_index_test,logrank,rae_nc,rae_c,cal,cluster0,cluster1,collapsed" >> "$LOG_FILE"
fi
echo "" >> "$LOG_FILE"

if [ $TUNE_HYPERPARAMS -eq 1 ]; then
    # Hyperparameter tuning loop - run in parallel
    declare -a pids
    declare -a configs
    job_count=0
    
    for lr in "${LR_VALUES[@]}"; do
        for discount in "${DISCOUNT_VALUES[@]}"; do
            for layers in "${LAYERS_VALUES[@]}"; do
                layers_safe=$(echo "$layers" | tr -d '[], ')
                GPU=$((job_count % ${#CUDA_DEVICES[@]}))
                echo "Starting hyperparameter tuning: lr=$lr, discount=$discount, layers=$layers on GPU $GPU (job $((job_count+1)))"
                
                (
                    cd /home/fzhuang/mref-ad/DCSM/DCSM
                    python -u main.py \
                        --dataset $DATASET \
                        --cuda_device ${CUDA_DEVICES[$GPU]} \
                        --learning_rate $lr \
                        --discount $discount \
                        --layers "$layers" \
                        --iters $FIXED_ITERS \
                        --early_stopping $FIXED_EARLY_STOPPING \
                        --patience $FIXED_PATIENCE > "$TMP_DIR/output_baseline_${lr}_${discount}_${layers_safe}.txt" 2>&1
                ) &
                
                pids+=($!)
                configs+=("lr=$lr discount=$discount layers=$layers")
                ((job_count++))
            done
        done
    done
    
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
    
    # Parse results from all completed jobs
    for lr in "${LR_VALUES[@]}"; do
        for discount in "${DISCOUNT_VALUES[@]}"; do
            for layers in "${LAYERS_VALUES[@]}"; do
                layers_safe=$(echo "$layers" | tr -d '[], ')
                OUTPUT_FILE="$TMP_DIR/output_baseline_${lr}_${discount}_${layers_safe}.txt"
                
                if [ -f "$OUTPUT_FILE" ]; then
                    # Parse results
                    python3 << PYEOF >> "$LOG_FILE" 2>> $TMP_DIR/debug_baseline.log
import re

lr = "$lr"
discount = "$discount"
with open("$OUTPUT_FILE", "r") as f:
    lines = f.readlines()

seed = None
c_idx_train = c_idx_val = c_idx_test = rae_nc = rae_c = cal = cluster0 = cluster1 = logrank = best_epoch = None

for line in lines:
    m = re.search(r'seed (\d+)', line)
    if m:
        if seed and c_idx_test:
            collapsed = 1 if (cluster0 == '0' or cluster1 == '0') else 0
            print(f"{lr},{discount},$layers,{seed},{best_epoch},{c_idx_train},{c_idx_val},{c_idx_test},{logrank},{rae_nc},{rae_c},{cal},{cluster0},{cluster1},{collapsed}")
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
    print(f"{lr},{discount},$layers,{seed},{best_epoch},{c_idx_train},{c_idx_val},{c_idx_test},{logrank},{rae_nc},{rae_c},{cal},{cluster0},{cluster1},{collapsed}")
PYEOF
                    
                    rm -f "$OUTPUT_FILE"
                else
                    echo "DEBUG: Missing $OUTPUT_FILE" >> $TMP_DIR/debug_baseline.log
                fi
            done
        done
    done
else
    echo "Running baseline DCSM experiment on GPU 0"
    
    cd /home/fzhuang/mref-ad/DCSM/DCSM
    python -u main.py \
        --dataset $DATASET \
        --cuda_device 0 \
        --learning_rate $FIXED_LR \
        --discount $FIXED_DISCOUNT \
        --layers "$FIXED_LAYERS" \
        --iters $FIXED_ITERS \
        --early_stopping $FIXED_EARLY_STOPPING \
        --patience $FIXED_PATIENCE > "$TMP_DIR/output_baseline.txt" 2>&1
    
    # Wait for file to be written and flush
    sleep 1
    sync
    
    # Save output for parsing
    OUTPUT_FILE="$TMP_DIR/output_baseline.txt"
    
    # Extract params
    grep "^param:" "$OUTPUT_FILE" | head -1 > "$TMP_DIR/params.tmp"
    
    # Use Python to parse - more reliable than awk state machine
    echo "DEBUG: Starting Python parsing" >> $TMP_DIR/debug_baseline.log
    python3 << PYEOF > "$TMP_DIR/perseed_baseline.tmp" 2>> $TMP_DIR/debug_baseline.log
import re

with open("$TMP_DIR/output_baseline.txt", "r") as f:
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
            print(f"{seed},{best_epoch},{c_idx_train},{c_idx_val},{c_idx_test},{logrank},{rae_nc},{rae_c},{cal},{cluster0},{cluster1},{collapsed}")
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
    print(f"{seed},{best_epoch},{c_idx_train},{c_idx_val},{c_idx_test},{logrank},{rae_nc},{rae_c},{cal},{cluster0},{cluster1},{collapsed}")
PYEOF

    echo "DEBUG: Python parsing completed, exit code=$?" >> $TMP_DIR/debug_baseline.log
    
    # Ensure file is flushed
    sync
    
    ls -lh "$TMP_DIR/perseed_baseline.tmp" >> $TMP_DIR/debug_baseline.log 2>&1
    
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
        print c_mean "," c_std "," lr_mean "," lr_std
    }
' "$OUTPUT_FILE" > "$TMP_DIR/summary_baseline.tmp"

# Clean up output file
rm -f "$OUTPUT_FILE"

echo "✓ Completed baseline DCSM"
echo ""

# Collect and write per-seed results
echo "DEBUG: Collecting per-seed results..."
if [ -f "$TMP_DIR/perseed_baseline.tmp" ]; then
    echo "DEBUG: Found $TMP_DIR/perseed_baseline.tmp"
    cat "$TMP_DIR/perseed_baseline.tmp" >> "$LOG_FILE"
    rm -f "$TMP_DIR/perseed_baseline.tmp"
else
    echo "DEBUG: Missing $TMP_DIR/perseed_baseline.tmp"
fi
fi

# Add parameters section if captured (only for fixed mode)
if [ $TUNE_HYPERPARAMS -eq 0 ] && [ -f "$TMP_DIR/params.tmp" ]; then
    echo "" >> "$LOG_FILE"
    echo "Model parameters used:" >> "$LOG_FILE"
    cat "$TMP_DIR/params.tmp" >> "$LOG_FILE"
    rm -f "$TMP_DIR/params.tmp"
fi

# Add summary section (only for fixed mode)
if [ $TUNE_HYPERPARAMS -eq 0 ]; then
    echo "" >> "$LOG_FILE"
    echo "=================================================" >> "$LOG_FILE"
    echo "Summary statistics across seeds:" >> "$LOG_FILE"
    echo "c_index_mean,c_index_std,logrank_mean,logrank_std" >> "$LOG_FILE"

    # Collect and write summary results
    echo "DEBUG: Collecting summary results..."
    if [ -f "$TMP_DIR/summary_baseline.tmp" ]; then
        echo "DEBUG: Found $TMP_DIR/summary_baseline.tmp"
        RESULT=$(cat "$TMP_DIR/summary_baseline.tmp")
        echo "$RESULT" >> "$LOG_FILE"
        echo "  $RESULT"
        rm -f "$TMP_DIR/summary_baseline.tmp"
    else
        echo "DEBUG: Missing $TMP_DIR/summary_baseline.tmp"
    fi
fi

echo "=================================================" >> "$LOG_FILE"
echo "End time: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Find best hyperparameters if in tuning mode
if [ $TUNE_HYPERPARAMS -eq 1 ]; then
    echo "" >> "$LOG_FILE"
    echo "=================================================" >> "$LOG_FILE"
    echo "BEST HYPERPARAMETER SELECTION" >> "$LOG_FILE"
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
    if line.startswith('lr,discount,layers'):
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
    if len(parts) >= 8:
        try:
            data.append({
                'lr': parts[0],
                'discount': parts[1],
                'layers': parts[2],
                'seed': int(parts[3]),
                'best_epoch': parts[4],
                'c_index_train': float(parts[5]),
                'c_index_val': float(parts[6]),
                'c_index_test': float(parts[7]),
                'logrank': float(parts[8]) if parts[8] != 'NA' else None,
            })
        except:
            pass

df = pd.DataFrame(data)

# Group by hyperparameters and compute statistics
if len(df) > 0:
    summary_flat = df.groupby(['lr', 'discount', 'layers']).apply(
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
    
    # Sort by validation C-index (descending), then by logrank (ascending = better p-value)
    summary_flat = summary_flat.sort_values(['c_val_mean', 'logrank_mean'], 
                                            ascending=[False, True])
    
    best = summary_flat.iloc[0]
    
    print("\nHyperparameter Search Results (sorted by validation C-index):")
    print("=" * 120)
    print(f"{'LR':<10} {'Discount':<12} {'Layers':<15} {'Val C-idx':<15} {'Test C-idx':<15} {'Logrank':<15}")
    print("-" * 120)
    for _, row in summary_flat.iterrows():
        print(f"{row['lr']:<10} {row['discount']:<12} {str(row['layers']):<15} "
              f"{row['c_val_mean']:.4f}±{row['c_val_std']:.4f}  "
              f"{row['c_test_mean']:.4f}±{row['c_test_std']:.4f}  "
              f"{row['logrank_mean']:.4f}±{row['logrank_std']:.4f}")
    
    print("\n" + "=" * 120)
    print("BEST HYPERPARAMETERS:")
    print("=" * 120)
    print(f"Learning rate:    {best['lr']}")
    print(f"Discount:         {best['discount']}")
    print(f"Layers:           {best['layers']}")
    print(f"\nValidation C-index:  {best['c_val_mean']:.4f} ± {best['c_val_std']:.4f}")
    print(f"Test C-index:        {best['c_test_mean']:.4f} ± {best['c_test_std']:.4f}")
    print(f"Logrank p-value:     {best['logrank_mean']:.4f} ± {best['logrank_std']:.4f}")
    print(f"Number of seeds:     {int(best['n_seeds'])}")
    print("=" * 120)
    
    # Also extract per-seed test results for the best hyperparameters
    best_results = df[(df['lr'] == best['lr']) & (df['discount'] == best['discount']) & (df['layers'] == best['layers'])]
    if len(best_results) > 0:
        print("\nPer-seed Test Results (for best hyperparameters):")
        print(f"{'Seed':<6} {'Test C-index':<15} {'Logrank':<15}")
        print("-" * 36)
        for _, row in best_results.iterrows():
            print(f"{int(row['seed']):<6} {row['c_index_test']:<15.4f} {row['logrank']:<15.4f}")
BESTPYEOF
fi

echo "=================================================" >> "$LOG_FILE"
echo "Experiment completed!"
echo "Results saved to: $LOG_FILE"
echo ""
echo "To view results:"
echo "  cat $LOG_FILE"

