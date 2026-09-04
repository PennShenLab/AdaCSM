#!/usr/bin/env python3
"""Ablation study: explore number of experts and top-k routing for ADACSM on AAL-AV45.

This script systematically runs ADACSM with different expert and top-k configurations,
collecting C-index and logrank metrics to analyze the impact of model capacity and
routing sparsity on survival prediction performance.

Configurations are distributed across available GPUs and run in parallel for efficiency.

Usage:
    python scripts/ablation_experts_topk.py
    python scripts/ablation_experts_topk.py --output_dir results/ablation_experts_topk
    python scripts/ablation_experts_topk.py --num_gpus 4
    python scripts/ablation_experts_topk.py --dry_run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = "dcsm_py310"
# DEFAULT_ENV = "dcsm_p100"

# Adacsm's vanilla moe best hyperparameters 
# BASE_PARAMS = {
#     "dataset": "AAL-AV45",
#     "model": "adacsm",
#     "learning_rate": 0.004202267426643019,
#     "discount": 0.4325407435059811,
#     "layers": "[100]",
#     "batch_size": 16,
#     "moe_dropout": 0.14467848105129133,
#     "gate_dropout": 0.06626550548167708,
#     "gate_temperature": 2.1455972556493643,
#     "early_stopping": True,
#     "patience": 200,
#     "iters": 2000,
# }

# DCSM's best hyperparameters
# BASE_PARAMS = {
#     "dataset": "AAL-AV45",
#     "model": "adacsm",
#     "learning_rate": 0.009814286783946129,
#     "discount": 0.43239283078572893,
#     "layers": "[50]",
#     "batch_size": 16,
#     "early_stopping": True,
#     "patience": 200,
#     "iters": 2000,
# }

# DCSM's best hyperparameters
BASE_PARAMS = {
    "dataset": "support",
    "model": "adacsm",
    "learning_rate": 2.1166819319297947e-05,
    "discount": 0.9930091326149568,
    "layers": "[50,50]",
    "batch_size": 16,
    "early_stopping": True,
    "patience": 200,
    "iters": 2000,
}

# ADACSM's best hyperparameters 
BASE_PARAMS = {
    "dataset": "support",
    "model": "adacsm",
    "learning_rate": 2.1166819319297947e-05,
    "discount": 0.9930091326149568,
    "layers": "[50,50]",
    "batch_size": 16,
    "early_stopping": True,
    "patience": 200,
    "iters": 2000,
}

# ADACSM's best hyperparameters 
BASE_PARAMS = {
    "dataset": "PBC",
    "model": "adacsm",
    "learning_rate": 0.0004684278914635085,
    "discount": 0.8315550363111635,
    "layers": "[100]",
    "moe_dropout": 0.04771865893108901,
    "gate_dropout": 0.27076409621255565,
    "gate_temperature": 0.1191336047177437,
    "load_balance_lambda": 0.09523379541056187,
    "batch_size": 16,
    "early_stopping": True,
    "patience": 200,
    "iters": 2000,
}

# AdaCSM's best hyperparameters
BASE_PARAMS = {
    "dataset": "FRAMINGHAM",
    "model": "adacsm",
    "learning_rate": 6.207335853713997e-05,
    "discount": 0.565870961578727,
    "layers": "[100]",
    "moe_dropout": 0.054407068775861794,
    "gate_dropout": 0.09819358504756028,
    "gate_temperature": 0.85531308577469,
    "load_balance_lambda": 0.07250389653349823,
    "batch_size": 16,
    "early_stopping": True,
    "patience": 200,
    "iters": 2000,
}

BASE_PARAMS = {
    "dataset": "flchain",
    "model": "adacsm",
    "learning_rate": 0.000502235021236179,
    "discount": 0.8643128475252542,
    "layers": "[100]",
    "moe_dropout": 0.2318480504907176,
    "gate_dropout": 0.15718136499824437,
    "gate_temperature": 2.0490264301561574,
    "load_balance_lambda": 0.050937135243724584,
    "batch_size": 16,
    "early_stopping": True,
    "patience": 200,
    "iters": 2000,
}

# Ablation dimensions
EXPERT_COUNTS = [2, 4, 8, 16, 32]
# TOP_K_VALUES = [2, 4, 6, 8, 10, 11]  # Increment of 2, up to max 11 regions in AAL
# TOP_K_VALUES = [1, 2, 4, 8, 16, 32]
TOP_K_VALUES = [1, 2, 4, 8, 10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study for ADACSM: experts vs top-k on AAL-AV45 (parallel across GPUs)")
    parser.add_argument(
        "--output_dir", type=str, default="results/ablation_experts_topk",
        help="Directory for results and logs")
    parser.add_argument(
        "--python_bin", type=str, default=sys.executable,
        help="Python binary to invoke")
    parser.add_argument(
        "--num_gpus", type=int, default=4,
        help="Number of GPUs to use for parallel execution (default: 4)")
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print commands without executing")
    parser.add_argument(
        "--skip_env_check", action="store_true",
        help="Bypass conda environment guard")
    parser.add_argument(
        "--use_regional", action="store_true",
        help="Sweep top-k with regional experts (num_experts auto-discovered from regions)")
    return parser.parse_args()


def ensure_environment(skip_check: bool) -> None:
    if skip_check:
        return
    active_env = os.environ.get("CONDA_DEFAULT_ENV")
    if active_env != DEFAULT_ENV:
        raise SystemExit(
            f"This script must run inside the '{DEFAULT_ENV}' conda environment. "
            f"Detected CONDA_DEFAULT_ENV={active_env!r}.")


def build_command(python_bin: str, num_experts: int, top_k: int, cuda_device: int, use_regional: bool = False) -> List[str]:
    """Build command to run main.py with given expert and top-k config."""
    cmd = [python_bin, "-u", "main.py"]
    
    for key, value in BASE_PARAMS.items():
        if isinstance(value, bool):
            # Pass boolean as string value (e.g., --early_stopping true)
            cmd.extend([f"--{key}", str(value).lower()])
        else:
            cmd.extend([f"--{key}", str(value)])
    
    # Add ablation dimensions
    if not use_regional:
        cmd.extend(["--num_experts", str(num_experts)])
    else:
        # When using regional experts, add the flag to auto-discover experts from regions
        cmd.append("--moe_region_experts")
    
    cmd.extend(["--top_k", str(top_k)])
    cmd.extend(["--cuda_device", str(cuda_device)])
    
    return cmd


def _extract_summary_line(text: str, header: str) -> Optional[str]:
    """Extract the line following a header in output."""
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if header in line:
            if idx + 1 < len(lines):
                return lines[idx + 1].strip()
    return None


def _parse_summary_values(line: str) -> Optional[Dict[str, float]]:
    """Parse metrics line like 'ADACSM:0.6000±0.0500 from -0.1234 to 1.2345'"""
    if not line:
        return None
    match = re.search(r":\s*([0-9.]+)\s*[±\u00B1]\s*([0-9.]+)\s*from\s*([0-9.\-]+)\s*to\s*([0-9.\-]+)", line)
    if not match:
        return None
    return {
        "mean": float(match.group(1)),
        "std": float(match.group(2)),
        "ci_low": float(match.group(3)),
        "ci_high": float(match.group(4)),
    }


def parse_metrics(output: str) -> Dict[str, Any]:
    """Extract C-index and logrank from main.py output."""
    metrics: Dict[str, Any] = {}
    
    cindex_line = _extract_summary_line(output, "C Index results")
    logrank_line = _extract_summary_line(output, "logrank results")
    
    parsed_cindex = _parse_summary_values(cindex_line) if cindex_line else None
    parsed_logrank = _parse_summary_values(logrank_line) if logrank_line else None
    
    if parsed_cindex:
        metrics["c_index"] = parsed_cindex
    if parsed_logrank:
        metrics["logrank"] = parsed_logrank
    
    return metrics


def run_config(config: Tuple[int, int, int, Path, str, bool, bool]) -> Dict[str, Any]:
    """Run a single configuration (called from ThreadPoolExecutor).
    
    Args:
        config: (num_experts, top_k, cuda_device, output_dir, python_bin, dry_run, use_regional)
    """
    num_experts, top_k, cuda_device, output_dir, python_bin, dry_run, use_regional = config
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"ablation_E{num_experts:02d}_k{top_k:02d}_gpu{cuda_device}_{ts}.log"
    cmd = build_command(python_bin, num_experts, top_k, cuda_device, use_regional=use_regional)
    
    print(f"[GPU{cuda_device}] Starting: E={num_experts:2d}, k={top_k:2d}")
    
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if dry_run:
            print(f"[GPU{cuda_device}] [DRY RUN] {' '.join(cmd)}")
            return {
                "num_experts": num_experts,
                "top_k": top_k,
                "cuda_device": cuda_device,
                "status": "dry_run",
                "log_file": str(log_path),
            }
        
        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            output_chunks: List[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                output_chunks.append(line)
                log_file.write(line)
            process.wait()
            
            if process.returncode != 0:
                raise RuntimeError(
                    f"Command exited with status {process.returncode}. See {log_path} for details.")
        
        raw_output = "".join(output_chunks)
        metrics = parse_metrics(raw_output)
        
        print(f"[GPU{cuda_device}] ✓ E={num_experts:2d}, k={top_k:2d} completed")
        
        return {
            "num_experts": num_experts,
            "top_k": top_k,
            "cuda_device": cuda_device,
            "metrics": metrics,
            "status": "success",
            "log_file": str(log_path),
        }
        
    except Exception as e:
        print(f"[GPU{cuda_device}] ✗ E={num_experts:2d}, k={top_k:2d} ERROR: {str(e)[:100]}")
        return {
            "num_experts": num_experts,
            "top_k": top_k,
            "cuda_device": cuda_device,
            "error": str(e),
            "status": "error",
            "log_file": str(log_path),
        }


def main() -> None:
    args = parse_args()
    ensure_environment(args.skip_env_check)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    use_regional = args.use_regional
    
    # Build list of all valid configurations
    configs_to_run: List[Tuple[int, int]] = []
    
    if use_regional:
        # For regional experts, num_experts is auto-discovered from region count
        # So we only sweep top_k (use placeholder num_experts=1 in the list)
        for top_k in TOP_K_VALUES:
            configs_to_run.append((1, top_k))  # num_experts placeholder; will be auto-discovered
    else:
        # Standard ablation: sweep both num_experts and top_k
        for num_experts in EXPERT_COUNTS:
            for top_k in TOP_K_VALUES:
                # Skip invalid configs: top_k should not exceed num_experts
                if top_k <= num_experts:
                    configs_to_run.append((num_experts, top_k))
    
    total_configs = len(configs_to_run)
    num_gpus = args.num_gpus
    
    mode = "Regional Experts + Top-K Sweep" if use_regional else "Experts vs Top-K"
    
    print("=" * 100)
    print(f"ADACSM ABLATION STUDY: {mode} (Parallel Execution)")
    print("=" * 100)
    print(f"Dataset: {BASE_PARAMS['dataset']}")
    print(f"Model: {BASE_PARAMS['model']}")
    print(f"Regional experts: {'YES' if use_regional else 'NO'}")
    print(f"Total configurations: {total_configs}")
    print(f"Available GPUs: {num_gpus}")
    print(f"Configs per GPU: ~{total_configs // num_gpus} - {(total_configs // num_gpus) + 1}")
    print(f"Base hyperparameters:")
    for k, v in BASE_PARAMS.items():
        if k not in ["dataset", "model"]:
            print(f"  {k}: {v}")
    print("=" * 100)
    
    # Assign GPUs to configs (round-robin distribution)
    config_tuples: List[Tuple[int, int, int, Path, str, bool, bool]] = []
    gpu_assignments: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(num_gpus)}
    
    print(f"\n[QUEUE] Distributing {total_configs} configs across {num_gpus} GPUs:\n")
    
    for idx, (num_experts, top_k) in enumerate(configs_to_run):
        gpu_id = idx % num_gpus
        config_tuples.append((num_experts, top_k, gpu_id, output_dir, args.python_bin, args.dry_run, use_regional))
        gpu_assignments[gpu_id].append((num_experts, top_k))
    
    # Print assignment matrix
    for gpu_id in range(num_gpus):
        configs = gpu_assignments[gpu_id]
        if use_regional:
            print(f"GPU{gpu_id}: {len(configs)} configs → top_k values: {[k for _, k in configs]}")
        else:
            print(f"GPU{gpu_id}: {len(configs)} configs → {configs}")
    print()
    
    if args.dry_run:
        print("\n[DRY RUN] Would execute the above configuration distribution.")
        print("\nSample commands:")
        for config in config_tuples[:2]:
            num_experts, top_k, cuda_device, _, python_bin, _, use_regional = config
            cmd = build_command(python_bin, num_experts, top_k, cuda_device, use_regional=use_regional)
            print(f"\n  GPU{cuda_device}: {' '.join(cmd[-6:])}")
        print("\n[DRY RUN] Complete.\n")
        return
    
    results: List[Dict[str, Any]] = []
    completed = 0
    errors = 0
    
    print(f"[START] Submitting all {total_configs} tasks to thread pool executor...\n")
    
    # Execute in parallel using ThreadPoolExecutor
    # Set max_workers to total_configs to parallelize all combinations at once
    # (GPU scheduling will limit actual concurrency per GPU)
    max_workers = min(total_configs, 32)  # Cap at 32 to avoid resource exhaustion
    print(f"[POOL] Using {max_workers} workers for all {total_configs} configurations\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_config, config): config for config in config_tuples}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                if result.get("status") == "error":
                    errors += 1
                
                # Print progress with GPU info
                gpu_id = result.get("cuda_device", "?")
                experts = result.get("num_experts", "?")
                top_k = result.get("top_k", "?")
                status = result.get("status", "unknown")
                
                status_icon = "✓" if status == "success" else "✗"
                print(f"[GPU{gpu_id}] {status_icon} E={experts:2d} k={top_k:2d} "
                      f"[{completed:2d}/{total_configs}]")
                
            except Exception as e:
                print(f"\n[Exception] Error in executor: {str(e)[:200]}")
                errors += 1
                completed += 1
    
    # Save detailed results
    results_file = output_dir / "ablation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] Detailed results saved to: {results_file}")
    
    # Print summary table
    if use_regional:
        print("\n" + "=" * 100)
        print("SUMMARY TABLE: Regional Experts + Top-K Sweep - C-Index (mean ± std)")
        print("=" * 100)
        for record in results:
            if record.get("status") == "success" and "metrics" in record:
                top_k = record.get("top_k")
                ci_data = record["metrics"].get("c_index")
                lr_data = record["metrics"].get("logrank")
                if ci_data and lr_data:
                    ci_mean = ci_data.get("mean", 0.0)
                    ci_std = ci_data.get("std", 0.0)
                    lr_mean = lr_data.get("mean", 0.0)
                    lr_std = lr_data.get("std", 0.0)
                    print(f"Top-K={top_k:2d}  C-Index: {ci_mean:.3f}±{ci_std:.3f}  |  LogRank: {lr_mean:.2f}±{lr_std:.2f}")
    else:
        print("\n" + "=" * 100)
        print("SUMMARY TABLE: C-Index (mean ± std)")
        print("=" * 100)
        print(f"{'E↓ / k→':<8}", end="")
        for top_k in TOP_K_VALUES:
            print(f"{top_k:>12}", end="")
        print()
        print("-" * 100)
        
        for num_experts in EXPERT_COUNTS:
            print(f"E={num_experts:<5}", end="")
            for top_k in TOP_K_VALUES:
                if top_k > num_experts:
                    print(f"{'─':>12}", end="")
                    continue
                record = next((r for r in results 
                              if r.get("num_experts") == num_experts and r.get("top_k") == top_k),
                             None)
                if record and record.get("status") == "success" and "metrics" in record:
                    ci_data = record["metrics"].get("c_index")
                    if ci_data:
                        mean = ci_data.get("mean", 0.0)
                        std = ci_data.get("std", 0.0)
                        print(f"{mean:.3f}±{std:.3f}".rjust(12), end="")
                    else:
                        print(f"{'N/A':>12}", end="")
                else:
                    print(f"{'ERROR':>12}", end="")
            print()
        
        print("\n" + "=" * 100)
        print("SUMMARY TABLE: Logrank (mean ± std)")
        print("=" * 100)
        print(f"{'E↓ / k→':<8}", end="")
        for top_k in TOP_K_VALUES:
            print(f"{top_k:>12}", end="")
        print()
        print("-" * 100)
        
        for num_experts in EXPERT_COUNTS:
            print(f"E={num_experts:<5}", end="")
            for top_k in TOP_K_VALUES:
                if top_k > num_experts:
                    print(f"{'─':>12}", end="")
                    continue
                record = next((r for r in results 
                              if r.get("num_experts") == num_experts and r.get("top_k") == top_k),
                             None)
                if record and record.get("status") == "success" and "metrics" in record:
                    lr_data = record["metrics"].get("logrank")
                    if lr_data:
                        mean = lr_data.get("mean", 0.0)
                        std = lr_data.get("std", 0.0)
                        print(f"{mean:.2f}±{std:.2f}".rjust(12), end="")
                    else:
                        print(f"{'N/A':>12}", end="")
                else:
                    print(f"{'ERROR':>12}", end="")
            print()
    
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(f"Total configs: {total_configs}")
    print(f"Completed: {completed}")
    print(f"Errors: {errors}")
    print(f"Success rate: {100 * (completed - errors) / total_configs:.1f}%")
    print(f"Results directory: {output_dir}/")
    print("=" * 100)


if __name__ == "__main__":
    main()
