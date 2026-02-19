#!/usr/bin/env python3
"""
Optuna-based hyperparameter tuning for DCSM with MoE.

Uses Tree-structured Parzen Estimator (TPE) for intelligent hyperparameter search
and MedianPruner for early stopping of unpromising trials.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
import subprocess
import tempfile
import re
from datetime import datetime, timezone

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for DCSM")
    
    # Dataset and experiment settings
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["support", "flchain", "PBC", "FRAMINGHAM", "adni"],
                        help="Dataset to use")
    parser.add_argument("--tune_trials", type=int, default=100,
                        help="Number of Optuna trials to run")
    parser.add_argument("--tune_epochs", type=int, default=2000,
                        help="Max epochs per trial (use with early stopping)")
    parser.add_argument("--patience", type=int, default=200,
                        help="Early stopping patience")
    
    # GPU settings
    parser.add_argument("--gpu_devices", type=str, default="0,1,2,3",
                        help="Comma-separated GPU device IDs")
    parser.add_argument("--trials_per_gpu", type=int, default=1,
                        help="Number of concurrent trials per GPU")
    
    # Output settings
    parser.add_argument("--out_base", type=str, required=True,
                        help="Base path for output files (without extension)")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (default: SQLite in results/)")
    parser.add_argument("--study_name", type=str, default=None,
                        help="Optuna study name (default: based on dataset)")
    parser.add_argument("--trial_log_dir", type=str, default="results/optuna_trial_logs",
                        help="Directory for per-trial temporary logs")
    parser.add_argument("--progress_every", type=int, default=50,
                        help="Print training progress every N epochs (0 disables)")
    
    # Optimization settings
    parser.add_argument("--select_metric", type=str, default="val_cindex",
                        choices=["val_cindex", "test_cindex", "logrank"],
                        help="Metric to optimize")
    parser.add_argument("--prune_trials", action="store_true", default=True,
                        help="Enable early pruning of unpromising trials")
    
    # Fixed trial settings (for testing)
    parser.add_argument("--n_experts_fixed", type=int, default=None,
                        help="Fix number of experts (otherwise tuned)")
    
    return parser.parse_args()


def objective(trial: optuna.Trial, args: argparse.Namespace, gpu_id: int) -> float:
    """
    Optuna objective function for a single trial.
    
    Args:
        trial: Optuna trial object
        args: Command-line arguments
        gpu_id: GPU device ID to use for this trial
        
    Returns:
        Objective value (validation C-index or logrank)
    """
    
    # Sample hyperparameters
    params = {}
    
    # Standard DCSM hyperparameters
    params["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    params["discount"] = trial.suggest_float("discount", 0.3, 1.0)
    params["layers"] = trial.suggest_categorical("layers", ["[50]", "[100]", "[50,50]"])
    
    # MoE-specific hyperparameters
    params["moe_dropout"] = trial.suggest_float("moe_dropout", 0.0, 0.5)
    params["gate_dropout"] = trial.suggest_float("gate_dropout", 0.0, 0.3)
    params["gate_temperature"] = trial.suggest_float("gate_temperature", 0.1, 5.0)
    params["load_balance_lambda"] = trial.suggest_float("load_balance_lambda", 0.0, 0.1)
    
    # Number of experts (categorical or fixed)
    if args.n_experts_fixed is not None:
        params["n_experts"] = args.n_experts_fixed
    else:
        params["n_experts"] = trial.suggest_categorical("n_experts", [1, 2, 4, 8, 16, 32])
    
    # Fixed hyperparameters (not tuned)
    params["weight_decay"] = 0.0  # Disabled - affects convergence
    params["routing_noise_std"] = 0.0  # Not tuned initially
    
    # Create per-trial log files
    log_dir = args.trial_log_dir
    active_dir = os.path.join(log_dir, "active")
    os.makedirs(active_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    trial_log = os.path.join(log_dir, f"trial_{trial.number}_gpu{gpu_id}.log")
    active_file = os.path.join(active_dir, f"trial_{trial.number}_gpu{gpu_id}.active")

    # Create temporary output file for parsing
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        tmp_output = tmp.name
    
    try:
        # Build command
        cmd = [
            "python", "-u", "main.py",
            "--dataset", args.dataset,
            "--cuda_device", str(gpu_id),
            "--learning_rate", str(params["lr"]),
            "--discount", str(params["discount"]),
            "--layers", params["layers"],
            "--weight_decay", str(params["weight_decay"]),
            "--moe_dropout", str(params["moe_dropout"]),
            "--gate_dropout", str(params["gate_dropout"]),
            "--load_balance_lambda", str(params["load_balance_lambda"]),
            "--gate_temperature", str(params["gate_temperature"]),
            "--routing_noise_std", str(params["routing_noise_std"]),
            "--iters", str(args.tune_epochs),
            "--early_stopping", "True",
            "--patience", str(args.patience),
            "--progress_every", str(args.progress_every),
            "--use_moe",
            "--num_experts", str(params["n_experts"]),
        ]
        
        # Run training
        print(f"\n[Trial {trial.number}] Running with params: {params}")
        print(f"[Trial {trial.number}] GPU: {gpu_id}")

        start_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with open(active_file, "w") as f:
            f.write(f"trial={trial.number}\n")
            f.write(f"gpu={gpu_id}\n")
            f.write(f"start={start_time}\n")
            f.write(f"params={json.dumps(params, sort_keys=True)}\n")
            f.write(f"log={trial_log}\n")

        lines = []
        with open(trial_log, "w", buffering=1) as log_f:
            log_f.write(f"[Trial {trial.number}] start {start_time}\n")
            log_f.write(f"[Trial {trial.number}] params: {json.dumps(params, sort_keys=True)}\n")
            log_f.write(f"[Trial {trial.number}] gpu: {gpu_id}\n")
            log_f.write(f"[Trial {trial.number}] status: launching process\n")
            log_f.flush()

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd="/home/fzhuang/mref-ad/DCSM/DCSM",
                bufsize=1,
            )

            for line in proc.stdout:
                lines.append(line)
                log_f.write(line)
                log_f.flush()
            proc.wait()

        output_text = "".join(lines)

        # Save output to temp file for parsing
        with open(tmp_output, "w") as f:
            f.write(output_text)

        # Parse results from output
        metrics = parse_training_output(output_text)
        
        if not metrics:
            print(f"[Trial {trial.number}] FAILED: Could not parse metrics")
            raise optuna.exceptions.TrialPruned()
        
        # Log metrics for this trial
        trial.set_user_attr("params", params)
        trial.set_user_attr("metrics", metrics)
        
        # Select objective based on args
        if args.select_metric == "val_cindex":
            objective_value = metrics["c_index_val_mean"]
            direction = "maximize"
        elif args.select_metric == "test_cindex":
            objective_value = metrics["c_index_test_mean"]
            direction = "maximize"
        elif args.select_metric == "logrank":
            objective_value = metrics["logrank_mean"]
            direction = "minimize"  # Lower is better for logrank p-value proxy
        else:
            raise ValueError(f"Unknown metric: {args.select_metric}")
        
        print(f"[Trial {trial.number}] {args.select_metric} = {objective_value:.4f}")
        print(f"[Trial {trial.number}] Metrics: val_cindex={metrics['c_index_val_mean']:.4f}, "
              f"test_cindex={metrics['c_index_test_mean']:.4f}, "
              f"logrank={metrics['logrank_mean']:.4f}")
        
        return objective_value
        
    except Exception as e:
        print(f"[Trial {trial.number}] ERROR: {e}")
        raise optuna.exceptions.TrialPruned()
        
    finally:
        # Clean up temp files
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
        if os.path.exists(trial_log):
            os.remove(trial_log)
        if os.path.exists(active_file):
            os.remove(active_file)


def parse_training_output(output: str) -> Dict[str, float]:
    """
    Parse metrics from training output.
    
    Extracts C-index and logrank statistics across all seeds.
    """
    lines = output.split('\n')
    
    # Collect per-seed results
    seeds_data = []
    current_seed = {}
    
    for line in lines:
        # Detect seed
        seed_match = re.search(r'seed (\d+)', line)
        if seed_match:
            if current_seed:
                seeds_data.append(current_seed)
            current_seed = {"seed": int(seed_match.group(1))}
        
        # Extract metrics
        if 'c-index on the training data:' in line:
            m = re.search(r': ([\d.]+)', line)
            if m:
                current_seed['c_index_train'] = float(m.group(1))
        elif 'c-index on the validation data:' in line:
            m = re.search(r': ([\d.]+)', line)
            if m:
                current_seed['c_index_val'] = float(m.group(1))
        elif 'c-index on the testing data:' in line:
            m = re.search(r': ([\d.]+)', line)
            if m:
                current_seed['c_index_test'] = float(m.group(1))
        elif 'Test statistic of test:' in line:
            m = re.search(r': ([\d.e+-]+)', line)
            if m:
                current_seed['logrank'] = float(m.group(1))
    
    # Add last seed
    if current_seed and 'c_index_val' in current_seed:
        seeds_data.append(current_seed)
    
    if not seeds_data:
        return None
    
    # Compute mean metrics
    import numpy as np
    
    metrics = {
        'c_index_train_mean': np.mean([s.get('c_index_train', 0) for s in seeds_data]),
        'c_index_val_mean': np.mean([s.get('c_index_val', 0) for s in seeds_data]),
        'c_index_test_mean': np.mean([s.get('c_index_test', 0) for s in seeds_data]),
        'logrank_mean': np.mean([s.get('logrank', 0) for s in seeds_data]),
        'c_index_train_std': np.std([s.get('c_index_train', 0) for s in seeds_data]),
        'c_index_val_std': np.std([s.get('c_index_val', 0) for s in seeds_data]),
        'c_index_test_std': np.std([s.get('c_index_test', 0) for s in seeds_data]),
        'logrank_std': np.std([s.get('logrank', 0) for s in seeds_data]),
        'n_seeds': len(seeds_data),
    }
    
    return metrics


def run_optuna_study(args: argparse.Namespace):
    """Run Optuna hyperparameter optimization study."""
    
    # Setup storage
    if args.storage is None:
        os.makedirs("results", exist_ok=True)
        args.storage = f"sqlite:///results/optuna_{args.dataset}.db"
    
    # Setup study name
    if args.study_name is None:
        args.study_name = f"dcsm_{args.dataset}_moe"
    
    # Create study
    direction = "maximize" if args.select_metric.endswith("cindex") else "minimize"
    
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10) if args.prune_trials else None
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=True,
    )
    
    print("="*80)
    print(f"Starting Optuna study: {args.study_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Trials: {args.tune_trials}")
    print(f"Metric: {args.select_metric} ({direction})")
    print(f"Storage: {args.storage}")
    print("="*80)
    
    # Parse GPU devices
    gpu_devices = [int(g.strip()) for g in args.gpu_devices.split(',')]
    n_jobs = len(gpu_devices) * args.trials_per_gpu
    
    print(f"\nRunning {n_jobs} parallel jobs across {len(gpu_devices)} GPUs")
    print(f"GPUs: {gpu_devices}")
    print()
    
    # GPU assignment function
    trial_count = [0]  # Mutable counter for closure
    
    def objective_with_gpu(trial):
        gpu_idx = trial_count[0] % len(gpu_devices)
        gpu_id = gpu_devices[gpu_idx]
        trial_count[0] += 1
        return objective(trial, args, gpu_id)
    
    # Run optimization
    study.optimize(
        objective_with_gpu,
        n_trials=args.tune_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )
    
    # Print results
    print("\n" + "="*80)
    print("Optimization complete!")
    print("="*80)
    
    best_trial = study.best_trial
    print(f"\nBest trial: #{best_trial.number}")
    print(f"  Value ({args.select_metric}): {best_trial.value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Get best trial metrics
    if "metrics" in best_trial.user_attrs:
        metrics = best_trial.user_attrs["metrics"]
        print(f"\nBest trial metrics:")
        print(f"  Val C-index: {metrics['c_index_val_mean']:.4f} ± {metrics['c_index_val_std']:.4f}")
        print(f"  Test C-index: {metrics['c_index_test_mean']:.4f} ± {metrics['c_index_test_std']:.4f}")
        print(f"  Logrank: {metrics['logrank_mean']:.4f} ± {metrics['logrank_std']:.4f}")
    
    # Save best trial to JSON
    output_json = {
        "study_name": args.study_name,
        "dataset": args.dataset,
        "best_trial_number": best_trial.number,
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "best_metrics": best_trial.user_attrs.get("metrics", {}),
        "select_metric": args.select_metric,
        "n_trials": len(study.trials),
    }
    
    out_file = f"{args.out_base}_best_trial.json"
    with open(out_file, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    print(f"\nBest trial saved to: {out_file}")
    
    # Save all trials to CSV
    df = study.trials_dataframe()
    csv_file = f"{args.out_base}_all_trials.csv"
    df.to_csv(csv_file, index=False)
    print(f"All trials saved to: {csv_file}")
    
    return study


def main():
    args = parse_args()
    
    # Check if main.py exists
    if not os.path.exists("main.py"):
        print("ERROR: main.py not found. Please run from DCSM/DCSM directory.")
        sys.exit(1)
    
    # Run study
    study = run_optuna_study(args)
    
    print("\n" + "="*80)
    print("Hyperparameter tuning complete!")
    print(f"Best {args.select_metric}: {study.best_value:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
