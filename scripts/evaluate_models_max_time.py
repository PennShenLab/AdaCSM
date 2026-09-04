#!/usr/bin/env python3
"""
Modified evaluate_models.py that uses max() time instead of 90th percentile.
This is for comparison purposes only.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import pickle
import glob
import os
import sys
import argparse
import warnings
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from lifelines.statistics import multivariate_logrank_test
import pandas as pd

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_data
from utils.general_utils import test_DCSM

def cluster_evenly(y_test, risks, n_clusters=2):
    """Create n_clusters by sorting risk and assigning contiguous blocks."""
    risks = np.asarray(risks).ravel()
    idx_sorted = np.argsort(risks)
    n = len(risks)
    cluster_tags = np.zeros(n, dtype=int)
    
    block_size = n // n_clusters
    for i in range(n_clusters - 1):
        cluster_tags[idx_sorted[i*block_size:(i+1)*block_size]] = i
    cluster_tags[idx_sorted[(n_clusters-1)*block_size:]] = n_clusters - 1
    
    y_test_list = [y_test[i] for i in range(len(y_test))]
    return y_test_list, cluster_tags

def evaluate_single_model(model_path, dataset_name, X_test, y_test, use_max_time=False):
    """Evaluate a single trained model."""
    results = {
        'c_index': np.nan,
        'logrank': np.nan,
        'n_cluster0': 0,
        'n_cluster1': 0,
        'event_rate_cluster0': np.nan,
        'event_rate_cluster1': np.nan,
    }
    
    # Try loading the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"  Error: Could not load model from {model_path}: {e}")
        return results
    
    X = X_test.copy()
    
    # Get initial cluster assignments for logrank computation
    try:
        if hasattr(model, 'predict_phenotype'):
            # DCSM/AdaCSM: uses learned clustering
            phenotypes = model.predict_phenotype(X)
            if isinstance(phenotypes, tuple):
                phenotypes = phenotypes[0]
            _, cluster_tags = cluster_evenly(y_test, phenotypes, n_clusters=2)
        elif hasattr(model, 'predict_risk'):
            # DeepCoxPH, DSM: use predict_risk with time horizon
            time_horizon_y = [item[1] for item in y_test]
            if use_max_time:
                time_horizon = float(np.max(time_horizon_y))
            else:
                time_horizon = float(np.percentile(time_horizon_y, 90))
            risks = model.predict_risk(X, time_horizon)
            risks = np.asarray(risks).ravel()
            risks = np.nan_to_num(risks, nan=0, posinf=0, neginf=0)
            _, cluster_tags = cluster_evenly(y_test, risks, n_clusters=2)
        elif hasattr(model, 'predict_partial_hazard'):
            # CoxPH: already has learned clustering
            import pandas as pd
            feature_names = model.params_.index.tolist()
            X_df = pd.DataFrame(X, columns=feature_names)
            risks = model.predict_partial_hazard(X_df).values.ravel()
            _, cluster_tags = cluster_evenly(y_test, risks, n_clusters=2)
        else:
            raise AttributeError(f"Model type {type(model).__name__} not supported")
    except Exception as e:
        print(f"  Error: Failed to get predictions from {type(model).__name__}: {e}")
        return results
    
    # Extract times and events
    times = np.array([item[1] for item in y_test])
    events = np.array([item[0] for item in y_test])
    
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
            # DCSM/AdaCSM/DeepCoxPH/DSM: use predict_risk with time horizon
            if use_max_time:
                time_horizon = float(np.max(times))
            else:
                time_horizon = float(np.percentile(times, 90))
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
        print(f"  Error computing C-index: {e}")
    
    # Compute logrank statistic
    try:
        logrank_result = multivariate_logrank_test(
            times[cluster0_idx],
            np.zeros(len(cluster0_idx)),
            events[cluster0_idx]
        )
        logrank_0 = logrank_result.test_statistic
        
        logrank_result = multivariate_logrank_test(
            times[cluster1_idx],
            np.zeros(len(cluster1_idx)),
            events[cluster1_idx]
        )
        logrank_1 = logrank_result.test_statistic
        
        results['logrank'] = logrank_0 + logrank_1
    except Exception as e:
        print(f"  Error computing logrank: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--model_pattern', type=str, required=True, 
                       help='Model file pattern with {seed} placeholder')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--use_max_time', action='store_true', help='Use max() instead of 90th percentile')
    parser.add_argument('--normalize', type=bool, default=True)
    
    args = parser.parse_args()
    
    print(f"Evaluating models on {args.dataset} dataset")
    print(f"Model pattern: {args.model_pattern}")
    print(f"Using: {'MAX TIME' if args.use_max_time else '90th percentile'}")
    
    # Load data
    class DataArgs:
        def __init__(self, dataset):
            self.dataset = dataset
            self.is_generate_sim = False
            self.num_inst = 10000
            self.num_feat = 1000
            self.is_save_sim = False
    
    data_args = DataArgs(args.dataset)
    X_train, _X_val, X_test, y_train, _y_val, y_test, _ = load_data(
        data_args, random_state=42,
    )
    
    # Determine split
    if args.split == 'test':
        X_use, y_use = X_test, y_test
    else:
        X_use, y_use = X_train, y_train
    
    # Normalize
    if args.normalize:
        scaler = StandardScaler()
        X_use = scaler.fit_transform(X_use)
    
    # Find seed values from pattern
    seeds_in_results = set()
    for f in glob.glob(args.model_pattern.replace('{seed}', '*')):
        try:
            # Extract seed from filename
            for seed_name in ['42', '73', '666', '777', '1009']:
                if f'seed{seed_name}' in f:
                    seeds_in_results.add(int(seed_name))
                    break
        except:
            pass
    
    seeds = sorted(list(seeds_in_results))
    print(f"Seeds: {seeds}")
    print(f"Split: {args.split}")
    print(f"Normalize: {args.normalize}\n")
    
    # Evaluate models
    all_results = []
    for seed in seeds:
        model_path = args.model_pattern.replace('{seed}', str(seed))
        if not os.path.exists(model_path):
            print(f"[Seed {seed}] Model not found: {model_path}")
            continue
        
        print(f"[Seed {seed}] Loading model from: {model_path}")
        results = evaluate_single_model(model_path, args.dataset, X_use, y_use, use_max_time=args.use_max_time)
        
        print(f"    C-index: {results['c_index']:.4f}")
        print(f"    LogRank: {results['logrank']:.4f}")
        print(f"    Cluster sizes: {results['n_cluster0']} / {results['n_cluster1']}")
        print(f"    Event rates: {results['event_rate_cluster0']:.3f} / {results['event_rate_cluster1']:.3f}\n")
        
        results['seed'] = seed
        all_results.append(results)
    
    # Summary statistics
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    if all_results:
        df = pd.DataFrame(all_results)
        c_indices = df['c_index'].values
        logrankstats = df['logrank'].values
        
        print(f"C-index: {c_indices.mean():.4f} ± {c_indices.std():.4f}")
        print(f"  (range: {c_indices.min():.4f} to {c_indices.max():.4f})")
        print(f"LogRank: {logrankstats.mean():.4f} ± {logrankstats.std():.4f}")
        print(f"  (range: {logrankstats.min():.4f} to {logrankstats.max():.4f})")

if __name__ == '__main__':
    main()
