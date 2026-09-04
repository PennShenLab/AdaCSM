#!/usr/bin/env python3
"""
Evaluate multiple trained models across seeds and compute mean ± std for C-index and LogRank.

Usage:
    # Evaluate ADACSM models on PBC across default seeds
    python scripts/evaluate_models.py --dataset PBC --model_pattern "models/ADACSM_PBC_seed{seed}_moe.pkl"
    
    # Custom seeds
    python scripts/evaluate_models.py --dataset PBC --model_pattern "models/ADACSM_PBC_seed{seed}_moe.pkl" --seeds 42 73 666
    
    # Specify split (train/val/test)
    python scripts/evaluate_models.py --dataset PBC --model_pattern "models/ADACSM_PBC_seed{seed}_moe.pkl" --split test
    
    # Save results to JSON
    python scripts/evaluate_models.py --dataset PBC --model_pattern "models/ADACSM_PBC_seed{seed}_moe.pkl" --output results/pbc_eval.json

    # Evaluate variant checkpoints across (num_experts, top_k)
    python scripts/evaluate_models.py --dataset flchain \
        --model_pattern "models/ADACSM_flchain_seed{seed}_numexperts{num_experts}_topk{top_k}.pkl" \
        --num_experts 2 4 8 16 32 --top_k_values 1 2 4 8 10
"""

import numpy as np
import argparse
import pickle as pkl
import sys
import json
import itertools
import torch
from pathlib import Path
from types import SimpleNamespace
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines.statistics import multivariate_logrank_test
from sksurv.metrics import concordance_index_censored

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_utils import load_data


def cluster_evenly(y_test, risks, n_clusters=2):
    """Create n_clusters by sorting risk and assigning contiguous blocks.
    Returns (y_test_list, cluster_tags).
    """
    risks = np.asarray(risks).ravel()
    idx_sorted = np.argsort(risks)
    n = len(risks)
    sizes = [(n // n_clusters) + (1 if i < (n % n_clusters) else 0) for i in range(n_clusters)]
    cluster_tags = np.zeros(n, dtype=int)
    pos = 0
    for k in range(n_clusters):
        sz = sizes[k]
        sel = idx_sorted[pos:pos+sz]
        cluster_tags[sel] = k
        pos += sz
    
    # Split y_test by cluster tags
    y_test_list = []
    for k in range(n_clusters):
        idx_k = np.where(cluster_tags == k)[0]
        y_test_list.append(y_test[idx_k].tolist())
    
    return y_test_list, cluster_tags


def compute_time_horizon(y, mode='max'):
    """Compute risk prediction horizon from labels [(event, time), ...]."""
    times = np.asarray([item[1] for item in y], dtype=float)
    if mode == 'p90':
        return float(np.percentile(times, 90))
    return float(np.max(times))


def evaluate_single_model(model, X, y, seed=42, time_horizon_mode='max'):
    """
    Evaluate a single model on given data.
    
    Args:
        model: Trained DCSM/ADACSM/CoxPH model
        X: Features (preprocessed to match training pipeline)
        y: Labels [(event, time), ...]
        seed: Random seed for reproducibility
        
    Returns:
        dict with keys: c_index, logrank_stat, logrank_p, n_cluster0, n_cluster1
    """
    results = {
        'c_index': None,
        'logrank_stat': None,
        'logrank_p': None,
        'n_cluster0': 0,
        'n_cluster1': 0,
        'event_rate_cluster0': None,
        'event_rate_cluster1': None,
    }
    
    # Detect model type and get cluster assignments accordingly
    if hasattr(model, 'predict_phenotype'):
        # DCSM/AdaCSM models with learned clustering
        cluster_tags, _, _ = model.predict_phenotype(X)
    else:
        # Baseline models (CoxPH, DeepCoxPH, DSM) - use risk-based clustering
        try:
            # For lifelines CoxPHFitter
            if hasattr(model, 'predict_partial_hazard'):
                import pandas as pd
                # Get feature names from model params
                feature_names = model.params_.index.tolist()
                X_df = pd.DataFrame(X, columns=feature_names)
                risks = model.predict_partial_hazard(X_df).values.ravel()
            elif hasattr(model, 'predict_risk'):
                # DeepCoxPH, DSM: use predict_risk with configured time horizon
                time_horizon = compute_time_horizon(y, mode=time_horizon_mode)
                risks = model.predict_risk(X, time_horizon)
                risks = np.asarray(risks).ravel()
                risks = np.nan_to_num(risks, nan=0, posinf=0, neginf=0)
            else:
                # Generic fallback: assume model has some risk prediction method
                raise AttributeError(f"Model type {type(model).__name__} not supported")
            
            # Use risk-based clustering (match main.py behavior)
            _, cluster_tags = cluster_evenly(y, risks, n_clusters=2)
        except Exception as e:
            print(f"  Error: Failed to get predictions from {type(model).__name__}: {e}")
            return results
    
    # Extract times and events
    times = np.array([item[1] for item in y])
    events = np.array([item[0] for item in y])
    
    # Count clusters
    unique_clusters = np.unique(cluster_tags)
    if len(unique_clusters) < 2:
        print(f"  Warning: Only {len(unique_clusters)} cluster(s) found")
        return results

    cluster0_idx = np.where(cluster_tags == unique_clusters[0])[0]
    cluster1_idx = np.where(cluster_tags == unique_clusters[1])[0]
    
    results['n_cluster0'] = len(cluster0_idx)
    results['n_cluster1'] = len(cluster1_idx)
    
    # Event rates
    if len(cluster0_idx) > 0:
        results['event_rate_cluster0'] = events[cluster0_idx].sum() / len(cluster0_idx)
    if len(cluster1_idx) > 0:
        results['event_rate_cluster1'] = events[cluster1_idx].sum() / len(cluster1_idx)
    
    # Compute C-index using continuous risk predictions
    try:
        if hasattr(model, 'predict_risk'):
            # DCSM/AdaCSM: use predict_risk with configured time horizon
            time_horizon = compute_time_horizon(y, mode=time_horizon_mode)
            risks = model.predict_risk(X, time_horizon)
            risks = np.asarray(risks).ravel()
            risks = np.nan_to_num(risks, nan=0, posinf=0, neginf=0)
        elif hasattr(model, 'predict_partial_hazard'):
            # CoxPH: use partial hazard as risk score
            import pandas as pd
            feature_names = model.params_.index.tolist()
            X_df = pd.DataFrame(X, columns=feature_names)
            risks = model.predict_partial_hazard(X_df).values.ravel()
        else:
            raise AttributeError(f"Model has no risk prediction method")
        
        results['c_index'] = float(concordance_index_censored(events.astype(bool), times, risks)[0])
    except Exception as e:
        print(f"  Warning: C-index computation failed: {e}")
    
    # Compute LogRank test (match plot_KM: multivariate over all clusters)
    if len(unique_clusters) >= 2:
        try:
            lr = multivariate_logrank_test(
                event_durations=times,
                groups=cluster_tags,
                event_observed=events,
            )
            results['logrank_stat'] = float(lr.test_statistic)
            results['logrank_p'] = float(lr.p_value)
        except Exception as e:
            print(f"  Warning: LogRank computation failed: {e}")
    else:
        print(f"  Warning: Insufficient clusters for LogRank (n_clusters={len(unique_clusters)})")
    
    return results


def load_seed_split(dataset_args, split, seed, normalize=True):
    """Load and preprocess data for one seed, matching main.py behavior."""
    X_train, _X_val, X_test, y_train, _y_val, y_test, _ = load_data(
        dataset_args, random_state=seed,
    )

    X_val = y_val = None
    if split == 'val':
        train_events = np.array([int(item[0]) for item in y_train])
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=seed,
            stratify=train_events,
        )
        X_train, y_train = X_train_sub, y_train_sub

    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if X_val is not None:
            X_val = scaler.transform(X_val)

    if split == 'train':
        return X_train, y_train
    if split == 'val':
        return X_val, y_val
    return X_test, y_test


def align_model_cuda_device(model):
    """Align active CUDA device to the checkpoint's parameter device.

    The DCSM API uses `.cuda()` without explicit device arguments, so
    inference device depends on the active CUDA device. This function sets
    `torch.cuda.set_device(...)` to match the loaded model parameters.
    """
    if not torch.cuda.is_available():
        return None
    torch_model = getattr(model, 'torch_model', None)
    if torch_model is None or not hasattr(torch_model, 'parameters'):
        return None
    try:
        first_param = next(torch_model.parameters())
    except StopIteration:
        return None
    except Exception:
        return None

    device = first_param.device
    if device.type == 'cuda' and device.index is not None:
        torch.cuda.set_device(device.index)
        return int(device.index)
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained models across seeds and compute statistics'
    )
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., PBC, support, flchain, FRAMINGHAM)')
    parser.add_argument('--model_pattern', type=str, required=True,
                        help='Model path pattern with {seed} placeholder, e.g., "models/ADACSM_PBC_seed{seed}_moe.pkl"')
    
    # Optional arguments
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 73, 666, 777, 1009],
                        help='List of seeds to evaluate (default: 42 73 666 777 1009)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Data split to evaluate on (default: test)')
    parser.add_argument('--no_normalize', action='store_true',
                        help='Skip data normalization (use if models were trained on raw data)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results as JSON (optional)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed per-seed results')
    parser.add_argument('--num_experts', type=int, nargs='+', default=None,
                        help='Optional list of num_experts values; used when model_pattern includes {num_experts}')
    parser.add_argument('--top_k_values', type=int, nargs='+', default=None,
                        help='Optional list of top_k values; used when model_pattern includes {top_k}')
    parser.add_argument('--time_horizon_mode', type=str, default='max', choices=['max', 'p90'],
                        help='Time horizon used for predict_risk models: max or p90 (default: max)')
    
    args = parser.parse_args()
    
    print(f"Evaluating models on {args.dataset} dataset")
    print(f"Model pattern: {args.model_pattern}")
    print(f"Seeds: {args.seeds}")
    print(f"Split: {args.split}")
    print(f"Time horizon mode: {args.time_horizon_mode}")
    print(f"Normalize: {not args.no_normalize}\n")

    pattern_fields = {
        field_name
        for _, field_name, _, _ in __import__('string').Formatter().parse(args.model_pattern)
        if field_name
    }
    required_fields = {'seed'}
    missing_required = required_fields - pattern_fields
    if missing_required:
        raise ValueError(f"model_pattern must include placeholders: {sorted(required_fields)}")

    uses_num_experts = 'num_experts' in pattern_fields
    uses_top_k = 'top_k' in pattern_fields

    if uses_num_experts and not args.num_experts:
        raise ValueError("model_pattern includes {num_experts} but --num_experts was not provided")
    if uses_top_k and not args.top_k_values:
        raise ValueError("model_pattern includes {top_k} but --top_k_values was not provided")

    if args.num_experts and not uses_num_experts:
        print("Warning: --num_experts provided but model_pattern has no {num_experts}; values will be ignored")
    if args.top_k_values and not uses_top_k:
        print("Warning: --top_k_values provided but model_pattern has no {top_k}; values will be ignored")

    variant_configs = []
    n_values = args.num_experts if uses_num_experts else [None]
    k_values = args.top_k_values if uses_top_k else [None]
    for n_val, k_val in itertools.product(n_values, k_values):
        if n_val is not None and k_val is not None and k_val > n_val:
            continue
        variant_configs.append({'num_experts': n_val, 'top_k': k_val})

    if not variant_configs:
        raise ValueError("No valid variant configs to evaluate (check --num_experts and --top_k_values)")

    print(f"Variant configs: {len(variant_configs)}")
    if len(variant_configs) <= 20:
        for cfg in variant_configs:
            print(f"  - num_experts={cfg['num_experts']}, top_k={cfg['top_k']}")
    print("")
    
    # Load dataset once per seed (match training/evaluation pipeline in main.py)
    print("Preparing per-seed data splits...")
    dataset_args = SimpleNamespace(
        dataset=args.dataset,
        is_generate_sim=False,
        num_inst=0,
        num_feat=0,
        is_save_sim=False,
    )

    seed_data = {}
    for seed in args.seeds:
        X_seed, y_seed = load_seed_split(
            dataset_args,
            split=args.split,
            seed=seed,
            normalize=not args.no_normalize,
        )
        seed_data[seed] = (X_seed, y_seed)

    sample_count = len(seed_data[args.seeds[0]][1]) if args.seeds else 0
    print(f"Evaluating on {args.split} split: ~{sample_count} samples per seed\n")
    
    # Evaluate each seed for each variant config
    all_results = []

    for cfg in variant_configs:
        n_val = cfg['num_experts']
        k_val = cfg['top_k']
        print(f"[Config] num_experts={n_val}, top_k={k_val}")

        for seed in args.seeds:
            format_kwargs = {
                'seed': seed,
                'num_experts': n_val,
                'top_k': k_val,
            }
            model_path = args.model_pattern.format(**format_kwargs)

            try:
                print(f"  [Seed {seed}] Loading model from: {model_path}")
                with open(model_path, 'rb') as f:
                    model = pkl.load(f)

                aligned_device = align_model_cuda_device(model)
                if args.verbose and aligned_device is not None:
                    print(f"    Aligned CUDA device to cuda:{aligned_device}")

                X_seed, y_seed = seed_data[seed]

                results = evaluate_single_model(
                    model, X_seed, y_seed,
                    seed=seed,
                    time_horizon_mode=args.time_horizon_mode,
                )
                results['seed'] = seed
                results['model_path'] = model_path
                results['num_experts'] = n_val
                results['top_k'] = k_val
                all_results.append(results)

                if args.verbose or len(args.seeds) <= 5:
                    print(f"    C-index: {results['c_index']:.4f}" if results['c_index'] else "    C-index: N/A")
                    print(f"    LogRank: {results['logrank_stat']:.4f}" if results['logrank_stat'] else "    LogRank: N/A")
                    print(f"    LogRank p-value: {results['logrank_p']:.6f}" if results['logrank_p'] else "    LogRank p-value: N/A")
                    print(f"    Cluster sizes: {results['n_cluster0']} / {results['n_cluster1']}")
                    if results['event_rate_cluster0'] is not None and results['event_rate_cluster1'] is not None:
                        print(f"    Event rates: {results['event_rate_cluster0']:.3f} / {results['event_rate_cluster1']:.3f}")
                    print("")

            except FileNotFoundError:
                print(f"    Error: Model file not found: {model_path}")
                continue
            except Exception as e:
                print(f"    Error loading/evaluating model: {e}")
                continue

    if not all_results:
        print("No models were successfully evaluated!")
        return

    # Compute statistics per variant config
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    def build_summary(results_subset):
        c_indices = [r['c_index'] for r in results_subset if r['c_index'] is not None]
        logrank_stats = [r['logrank_stat'] for r in results_subset if r['logrank_stat'] is not None]
        logrank_ps = [r['logrank_p'] for r in results_subset if r['logrank_p'] is not None]

        summary = {
            'dataset': args.dataset,
            'split': args.split,
            'n_seeds_evaluated': len(results_subset),
            'n_seeds_requested': len(args.seeds),
        }

        if c_indices:
            c_mean = np.mean(c_indices)
            c_std = np.std(c_indices, ddof=1) if len(c_indices) > 1 else 0.0
            c_min = np.min(c_indices)
            c_max = np.max(c_indices)
            summary['c_index_mean'] = float(c_mean)
            summary['c_index_std'] = float(c_std)
            summary['c_index_min'] = float(c_min)
            summary['c_index_max'] = float(c_max)

        if logrank_stats:
            lr_mean = np.mean(logrank_stats)
            lr_std = np.std(logrank_stats, ddof=1) if len(logrank_stats) > 1 else 0.0
            lr_min = np.min(logrank_stats)
            lr_max = np.max(logrank_stats)
            summary['logrank_mean'] = float(lr_mean)
            summary['logrank_std'] = float(lr_std)
            summary['logrank_min'] = float(lr_min)
            summary['logrank_max'] = float(lr_max)

        if logrank_ps:
            p_mean = np.mean(logrank_ps)
            p_std = np.std(logrank_ps, ddof=1) if len(logrank_ps) > 1 else 0.0
            summary['logrank_p_mean'] = float(p_mean)
            summary['logrank_p_std'] = float(p_std)

        return summary

    grouped_summaries = []
    grouped_keys = [(cfg['num_experts'], cfg['top_k']) for cfg in variant_configs]
    for n_val, k_val in grouped_keys:
        subset = [r for r in all_results if r.get('num_experts') == n_val and r.get('top_k') == k_val]
        summary = build_summary(subset)
        summary['num_experts'] = n_val
        summary['top_k'] = k_val
        grouped_summaries.append(summary)

        print(f"num_experts={n_val}, top_k={k_val}")
        if 'c_index_mean' in summary:
            print(f"  C-index: {summary['c_index_mean']:.4f} ± {summary['c_index_std']:.4f} "
                  f"(range: {summary['c_index_min']:.4f} to {summary['c_index_max']:.4f})")
        else:
            print("  C-index: N/A (no valid results)")

        if 'logrank_mean' in summary:
            print(f"  LogRank: {summary['logrank_mean']:.4f} ± {summary['logrank_std']:.4f} "
                  f"(range: {summary['logrank_min']:.4f} to {summary['logrank_max']:.4f})")
        else:
            print("  LogRank: N/A (no valid results)")

        if 'logrank_p_mean' in summary:
            print(f"  LogRank p-value: {summary['logrank_p_mean']:.6f} ± {summary['logrank_p_std']:.6f}")
        print("")

    print("="*60)
    
    # Save to JSON if requested
    if args.output:
        output_data = {
            'summary_by_variant': grouped_summaries,
            'per_seed_results': all_results,
            'config': {
                'dataset': args.dataset,
                'model_pattern': args.model_pattern,
                'seeds': args.seeds,
                'split': args.split,
                'normalize': not args.no_normalize,
                'time_horizon_mode': args.time_horizon_mode,
                'num_experts': args.num_experts,
                'top_k_values': args.top_k_values,
            }
        }
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
