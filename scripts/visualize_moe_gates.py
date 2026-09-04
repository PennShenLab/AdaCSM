"""
Visualize MoE Gating Patterns to Detect Expert Collapse

This script analyzes the gating weights from a trained DCSM MoE model to:
1. Check if experts are being used or if there's collapse to 2-3 experts
2. Compare expert assignment patterns between high-risk and low-risk patients
3. Visualize expert specialization via heatmaps

Usage:
    python scripts/visualize_moe_gates.py --dataset FRAMINGHAM --model_path models/ADACSM_FRAMINGHAM_seed42_moe.pkl
    
    Or train a new model with specified hyperparameters:
    python scripts/visualize_moe_gates.py --dataset FRAMINGHAM --learning_rate 0.0007 --discount 0.36 \
        --layers "[50,50]" --num_experts 32 --top_k 2 --gate_temperature 1.5

    conda run -n adacsm python scripts/visualize_moe_gates.py \
        --dataset FRAMINGHAM \
        --seed 42 \
        --model_path models/ADACSM_FRAMINGHAM_seed42_numexperts32_topk2.pkl \
        --output_dir results/gate_viz_framingham_seed42_numexperts32_topk2 \
        --subgroup_mode weighted \
        --canonical_reference_model models/ADACSM_FRAMINGHAM_seed42_numexperts32_topk2.pkl \
        --canonical_experts 5,7,10,31 \
        --profile_experts 5,14

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle as pkl
import os
import sys
import torch
import io
from collections import defaultdict
import ast

# Add parent directory to path to import utilities
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_utils import load_data
try:
    from utils.data_utils import build_brain_region_expert_splits
except ImportError:
    build_brain_region_expert_splits = None
from utils.general_utils import train_test_AdaCSM
from utils.eval_utils import evaluate_and_plot


class CPUUnpickler(pkl.Unpickler):
    """Load CUDA-pickled models on CPU-only machines."""

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


def _expert_label(expert_idx, expert_label_map=None):
    """Return display label for an expert index."""
    if expert_label_map is not None and expert_idx in expert_label_map:
        return f'Expert {expert_label_map[expert_idx]}'
    return f'Expert {expert_idx}'


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize MoE Gating Patterns')
    
    # Model loading options
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pickled trained model (if None, will train new model)')
    parser.add_argument('--dataset', type=str, default='AAL-AV45',
                        help='Dataset name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for data splitting')
    parser.add_argument('--cuda_device', type=int, default=0,
                        help='CUDA device index')
    
    # Model hyperparameters (used if training new model)
    parser.add_argument('--learning_rate', type=float, default=0.0007428870363215633,
                        help='Learning rate')
    parser.add_argument('--discount', type=float, default=0.35957415873680715,
                        help='Discount parameter')
    parser.add_argument('--layers', type=str, default='[50,50]',
                        help='Hidden layer sizes as string, e.g., "[50,50]"')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_experts', type=int, default=32,
                        help='Number of experts')
    parser.add_argument('--top_k', type=int, default=2,
                        help='Top-k experts to use')
    parser.add_argument('--moe_dropout', type=float, default=0.07282739027874861,
                        help='MoE dropout rate')
    parser.add_argument('--gate_dropout', type=float, default=0.026842205531115236,
                        help='Gate dropout rate')
    parser.add_argument('--gate_temperature', type=float, default=1.469037512872845,
                        help='Gate temperature')
    parser.add_argument('--load_balance_lambda', type=float, default=0.04232677228780192,
                        help='Load balance loss weight')
    parser.add_argument('--iters', type=int, default=2000,
                        help='Training iterations')
    parser.add_argument('--patience', type=int, default=200,
                        help='Early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay')
    parser.add_argument('--routing_noise_std', type=float, default=0.0,
                        help='Routing noise std')
    
    # Visualization options
    parser.add_argument('--risk_percentile', type=float, default=50,
                        help='Percentile to split high/low risk (default: 50 = median)')
    parser.add_argument('--output_dir', type=str, default='results/moe_gates',
                        help='Directory to save visualizations')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Normalize input data for model inference')
    parser.add_argument('--no_normalize', action='store_false', dest='normalize',
                        help='Disable normalization for model inference')
    parser.add_argument('--subgroup_mode', type=str, default='weighted',
                        choices=['weighted', 'hard'],
                        help='Subgroup expert aggregation: weighted gate mass or hard argmax assignment')
    parser.add_argument('--legend_threshold', type=float, default=0.03,
                        help='Minimum subgroup fraction/mass for experts to appear in subgroup legends')
    parser.add_argument('--canonical_reference_model', type=str, default=None,
                        help='Optional reference model path to remap experts by subgroup pattern')
    parser.add_argument('--canonical_experts', type=str, default='5,7,10,31',
                        help='Comma-separated canonical expert ids from reference model')
    parser.add_argument('--profile_experts', type=str, default=None,
                        help='Optional comma-separated expert ids for clinical profile plot (e.g., "5,14")')
    
    # Region experts (for AAL datasets)
    parser.add_argument('--moe_region_experts', action='store_true',
                        help='Use region-specific experts for AAL datasets')
    parser.add_argument('--region_lookup_path', type=str, default='datasets/AAL-lookup-table.csv',
                        help='Path to region lookup table')
    parser.add_argument('--region_group_column', type=str, default='Region',
                        help='Column name for region grouping')
    
    return parser.parse_args()


def load_or_train_model(args, X_train, X_test, y_train, y_test, expert_feature_splits=None, expert_names=None,
                        X_val=None, y_val=None):
    """Load a trained model or train a new one with specified hyperparameters."""
    if args.model_path and os.path.exists(args.model_path):
        print(f'Loading model from {args.model_path}...')
        with open(args.model_path, 'rb') as f:
            try:
                model = pkl.load(f)
            except Exception:
                f.seek(0)
                model = CPUUnpickler(f).load()
        try:
            if hasattr(model, 'set_device'):
                model.set_device('cpu')
        except Exception:
            pass
        print('Model loaded successfully.')
        return model, None
    else:
        if args.model_path:
            print(f'Warning: Model path {args.model_path} not found. Training new model...')
        else:
            print('No model path specified. Training new model with provided hyperparameters...')
        
        # Parse layers
        layers = ast.literal_eval(args.layers)
        
        # Build parameter dictionary
        param = {
            'learning_rate': args.learning_rate,
            'layers': layers,
            'k': 2,
            'iters': args.iters,
            'distribution': 'Weibull',
            'discount': args.discount,
            'patience': args.patience,
            'early_stopping': True,
            'batch_size': args.batch_size,
        }
        
        print(f'Training ADACSM model with {args.num_experts} experts (top-k={args.top_k})...')
        model, c_index, pred, pred_time, rae_nc, rae_c = train_test_AdaCSM(
            param, X_train, X_test, y_train, y_test,
            seed=args.seed, fix=True,
            num_experts=args.num_experts,
            top_k=args.top_k,
            moe_dropout=args.moe_dropout,
            gate_dropout=args.gate_dropout,
            gate_temperature=args.gate_temperature,
            routing_noise_std=args.routing_noise_std,
            weight_decay=args.weight_decay,
            load_balance_lambda=args.load_balance_lambda,
            progress_every=0,
            expert_feature_splits=expert_feature_splits,
            expert_names=expert_names,
            early_stop_metric='cindex',
            ranking_loss_lambda=0.0,
            X_val=X_val, y_val=y_val,
        )

        e_test = np.array([[item[0] * 1 for item in y_test]]).T
        t_test = np.array([[item[1] for item in y_test]]).T

        try:
            test_eval = evaluate_and_plot(
                risks=pred[:, 0],
                t=t_test[:, 0],
                e=e_test[:, 0],
                model_name='ADACSM',
                data_name=args.dataset,
                seed=args.seed,
                n_clusters=2,
            )
        except Exception:
            test_eval = {'logrank_stat': None, 'logrank_p': None}

        train_metrics = {
            'test_cindex': float(c_index),
            'test_logrank': test_eval.get('logrank_stat') if test_eval else None,
            'test_logrank_p': test_eval.get('logrank_p') if test_eval else None,
            'test_ibs': getattr(model, '_last_test_ibs', None),
        }

        print('\n' + '=' * 80)
        print('NEW MODEL TRAINING METRICS')
        print('=' * 80)
        print(f"test_cindex: {train_metrics['test_cindex']:.4f}")
        if train_metrics['test_logrank'] is not None:
            print(f"test_logrank: {train_metrics['test_logrank']:.4f}")
        else:
            print('test_logrank: nan')
        if train_metrics['test_ibs'] is not None:
            print(f"test_ibs: {float(train_metrics['test_ibs']):.4f}")
        else:
            print('test_ibs: nan')
        print('=' * 80)
        
        print(f'Model trained. Test C-index: {c_index:.4f}')
        return model, train_metrics


def extract_gate_weights(model, X_data):
    """Extract gating weights for all samples in X_data.
    
    Returns:
        gate_weights: numpy array of shape (n_samples, n_experts)
    """
    # Ensure model is in eval mode
    model.torch_model.eval()
    
    # Preprocess data (handle normalization, convert to tensor)
    if hasattr(model, '_preprocess_test_data'):
        X_tensor = model._preprocess_test_data(X_data)
        if not isinstance(X_tensor, torch.Tensor):
            X_tensor = torch.from_numpy(np.asarray(X_tensor)).float()
    else:
        # Fallback: manual preprocessing
        X_tensor = torch.from_numpy(np.asarray(X_data)).float()

    # Determine model device robustly and move input there to avoid mismatches
    try:
        # Prefer parameters from moe_layer if available
        if hasattr(model.torch_model, 'moe_layer') and model.torch_model.moe_layer is not None:
            param_iter = model.torch_model.moe_layer.parameters()
        else:
            param_iter = model.torch_model.parameters()
        first_param = next(param_iter, None)
        model_device = first_param.device if first_param is not None else torch.device('cpu')
    except Exception:
        model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move tensor to the same device as the model
    X_tensor = X_tensor.to(model_device)

    # Extract gate weights
    with torch.no_grad():
        if hasattr(model.torch_model, 'moe_layer') and model.torch_model.moe_layer is not None:
            gate_weights = model.torch_model.moe_layer.inspect_gate_weights(X_tensor)
            gate_weights_np = gate_weights.detach().cpu().numpy()
        else:
            raise ValueError('Model does not have MoE layer. Cannot extract gating weights.')

    return gate_weights_np


def compute_risk_scores(model, X_data):
    """Compute risk scores for stratification (higher = higher risk)."""
    # Use model predictions as risk scores
    # For survival models, we can use predicted time (inverted) or predicted scale parameter
    try:
        # Get predictions from model
        predictions = model.predict_survival(X_data, t=None)
        # Use the scale parameter (lower scale = higher risk in Weibull)
        risk_scores = -predictions  # Negative because lower predicted time = higher risk
    except:
        # Fallback: use predicted times directly
        predictions = model.predict_mean(X_data)
        risk_scores = -predictions  # Negative because lower predicted time = higher risk
    
    return risk_scores.flatten()


def stratify_by_risk(risk_scores, percentile=50):
    """Stratify patients into high-risk and low-risk based on percentile.
    
    Returns:
        high_risk_idx: indices of high-risk patients
        low_risk_idx: indices of low-risk patients
    """
    threshold = np.percentile(risk_scores, percentile)
    high_risk_idx = np.where(risk_scores > threshold)[0]
    low_risk_idx = np.where(risk_scores <= threshold)[0]

    # Robust fallback when risk scores are tied/degenerate and one side becomes empty.
    if len(high_risk_idx) == 0 or len(low_risk_idx) == 0:
        n = len(risk_scores)
        if n <= 1:
            return np.array([], dtype=int), np.arange(n, dtype=int)
        # Deterministic rank-based split (upper tail = high-risk)
        order = np.argsort(risk_scores, kind='mergesort')
        split = int(np.floor((percentile / 100.0) * n))
        split = max(1, min(n - 1, split))
        low_risk_idx = order[:split]
        high_risk_idx = order[split:]
    
    return high_risk_idx, low_risk_idx


def plot_gating_heatmap(gate_weights_high, gate_weights_low, expert_names=None,
                        expert_label_map=None,
                        output_path='results/moe_gates/gating_heatmap.png',
                        top_k=None):
    """Plot heatmap comparing average gating weights for high-risk vs low-risk patients.
    
    Args:
        gate_weights_high: (n_high_risk, n_experts) array
        gate_weights_low: (n_low_risk, n_experts) array
        expert_names: List of expert names (optional)
        output_path: Path to save the figure
        top_k: Number of top-k experts used (for annotation)
    """
    n_experts = gate_weights_high.shape[1]
    
    # Compute average gating weights for each group
    avg_high = gate_weights_high.mean(axis=0)
    avg_low = gate_weights_low.mean(axis=0)
    
    # Combine into single array for heatmap
    heatmap_data = np.vstack([avg_high, avg_low])
    
    # Expert labels
    if expert_names is None:
        expert_labels = [_expert_label(i, expert_label_map) for i in range(n_experts)]
    else:
        expert_labels = [name[:20] for name in expert_names]  # Truncate long names
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.5, 1], hspace=0.4, wspace=0.3)
    
    # --- Main heatmap ---
    ax1 = fig.add_subplot(gs[0, :])
    sns.heatmap(heatmap_data, 
                xticklabels=expert_labels,
                yticklabels=['High Risk', 'Low Risk'],
                cmap='YlOrRd',
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Average Gating Weight'},
                ax=ax1)
    
    title = f'Expert Gating Weights: High-Risk vs Low-Risk Patients'
    if top_k is not None:
        title += f' (top-k={top_k})'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Expert Index', fontsize=12)
    ax1.set_ylabel('Risk Group', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    # --- Bar plot comparing average weights (filtered to non-zero experts) ---
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Filter to experts with meaningful weights (both groups combined)
    max_weights = np.maximum(avg_high, avg_low)
    min_weight_threshold = 0.01
    active_mask = max_weights > min_weight_threshold
    active_indices = np.where(active_mask)[0]
    
    if len(active_indices) > 0:
        # Use only active experts
        active_high = avg_high[active_indices]
        active_low = avg_low[active_indices]
        active_labels = [_expert_label(i, expert_label_map) for i in active_indices]
        
        x = np.arange(len(active_indices))
        width = 0.35
        ax2.bar(x - width/2, active_high, width, label='High Risk', alpha=0.8, color='#e74c3c')
        ax2.bar(x + width/2, active_low, width, label='Low Risk', alpha=0.8, color='#3498db')
        ax2.set_xticks(x)
        ax2.set_xticklabels(active_labels, fontsize=10)
        ax2.set_xlabel('Expert Index', fontsize=12)
        ax2.set_ylabel('Average Gating Weight', fontsize=12)
        ax2.set_title(f'Active Experts with Weights > {min_weight_threshold} (n={len(active_indices)})', 
                      fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
    else:
        # Fallback: show all experts if none meet threshold
        x = np.arange(n_experts)
        width = 0.35
        ax2.bar(x - width/2, avg_high, width, label='High Risk', alpha=0.8, color='#e74c3c')
        ax2.bar(x + width/2, avg_low, width, label='Low Risk', alpha=0.8, color='#3498db')
        ax2.set_xlabel('Expert Index', fontsize=12)
        ax2.set_ylabel('Average Gating Weight', fontsize=12)
        ax2.set_title('Average Gating Weight per Expert (All Experts)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
    
    # --- Weight difference plot (filtered to active experts) ---
    ax3 = fig.add_subplot(gs[1, 1])
    diff_labels = None
    diff_values = None
    
    if len(active_indices) > 0:
        diff = active_high - active_low
        colors = ['#e74c3c' if d > 0 else '#3498db' for d in diff]
        x_diff = np.arange(len(active_indices))
        bars = ax3.bar(x_diff, diff, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
        diff_labels = active_labels
        diff_values = diff
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, diff)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=12, fontweight='bold')
        
        ax3.set_xticks(x_diff)
        ax3.set_xticklabels(active_labels, rotation=0, ha='center', fontsize=12)
        ax3.set_xlabel('')
        ax3.set_ylabel('Weight Difference (High Risk - Low Risk)', fontsize=13, labelpad=16)
        ax3.set_title('')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.grid(axis='y', alpha=0.3)
        ax3.tick_params(axis='y', labelsize=12)
        ax3.text(0.99, 0.95, 'High-risk preference (+)', transform=ax3.transAxes,
                 fontsize=12, color='#e74c3c', va='top', ha='right')
        ax3.text(0.99, 0.05, 'Low-risk preference (-)', transform=ax3.transAxes,
                 fontsize=12, color='#3498db', va='bottom', ha='right')
    else:
        diff = avg_high - avg_low
        colors = ['#e74c3c' if d > 0 else '#3498db' for d in diff]
        x = np.arange(n_experts)
        ax3.bar(x, diff, color=colors, alpha=0.7, edgecolor='black', linewidth=0.8)
        diff_labels = [_expert_label(i, expert_label_map) for i in range(n_experts)]
        diff_values = diff
        ax3.set_xticks(x)
        ax3.set_xticklabels(diff_labels, rotation=0, ha='center', fontsize=12)
        ax3.set_xlabel('')
        ax3.set_ylabel('Weight Difference (High Risk - Low Risk)', fontsize=13, labelpad=16)
        ax3.set_title('')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax3.grid(axis='y', alpha=0.3)
        ax3.tick_params(axis='y', labelsize=12)
        ax3.text(0.99, 0.95, 'High-risk preference (+)', transform=ax3.transAxes,
                 fontsize=12, color='#e74c3c', va='top', ha='right')
        ax3.text(0.99, 0.05, 'Low-risk preference (-)', transform=ax3.transAxes,
                 fontsize=12, color='#3498db', va='bottom', ha='right')
    
    # --- Expert utilization statistics ---
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    # Compute statistics
    # For top-k routing, count how many patients primarily use each expert
    top_expert_high = np.argmax(gate_weights_high, axis=1)
    top_expert_low = np.argmax(gate_weights_low, axis=1)
    
    usage_high = np.bincount(top_expert_high, minlength=n_experts) / len(top_expert_high) * 100
    usage_low = np.bincount(top_expert_low, minlength=n_experts) / len(top_expert_low) * 100
    
    # Find most used experts
    top_3_high = np.argsort(usage_high)[-3:][::-1]
    top_3_low = np.argsort(usage_low)[-3:][::-1]
    
    # Check for collapse
    collapse_threshold = 60  # If top-3 experts handle >60% of patients
    collapse_high = usage_high[top_3_high].sum()
    collapse_low = usage_low[top_3_low].sum()
    
    stats_text = f"EXPERT UTILIZATION STATISTICS\n\n"
    stats_text += f"High-Risk Patients (n={len(gate_weights_high)}):\n"
    stats_text += f"  Top-3 Primary Experts: {top_3_high[0]}, {top_3_high[1]}, {top_3_high[2]}\n"
    stats_text += f"  Top-3 Usage: {usage_high[top_3_high[0]]:.1f}%, {usage_high[top_3_high[1]]:.1f}%, {usage_high[top_3_high[2]]:.1f}% "
    stats_text += f"(Total: {collapse_high:.1f}%)\n"
    if collapse_high > collapse_threshold:
        stats_text += f"  ⚠️  WARNING: Possible expert collapse! Top-3 experts handle {collapse_high:.1f}% of patients.\n"
    else:
        stats_text += f"  ✓ Good distribution across experts.\n"
    
    stats_text += f"\nLow-Risk Patients (n={len(gate_weights_low)}):\n"
    stats_text += f"  Top-3 Primary Experts: {top_3_low[0]}, {top_3_low[1]}, {top_3_low[2]}\n"
    stats_text += f"  Top-3 Usage: {usage_low[top_3_low[0]]:.1f}%, {usage_low[top_3_low[1]]:.1f}%, {usage_low[top_3_low[2]]:.1f}% "
    stats_text += f"(Total: {collapse_low:.1f}%)\n"
    if collapse_low > collapse_threshold:
        stats_text += f"  ⚠️  WARNING: Possible expert collapse! Top-3 experts handle {collapse_low:.1f}% of patients.\n"
    else:
        stats_text += f"  ✓ Good distribution across experts.\n"
    
    # Compute entropy as another measure of diversity
    def entropy(weights):
        # Average entropy across patients
        eps = 1e-10
        return -np.mean(np.sum(weights * np.log(weights + eps), axis=1))
    
    entropy_high = entropy(gate_weights_high)
    entropy_low = entropy(gate_weights_low)
    
    stats_text += f"\nGating Entropy (higher = more diverse routing):\n"
    stats_text += f"  High-Risk: {entropy_high:.3f}\n"
    stats_text += f"  Low-Risk: {entropy_low:.3f}\n"
    
    # Check for non-zero experts
    nonzero_high = np.sum(avg_high > 0.001)
    nonzero_low = np.sum(avg_low > 0.001)
    stats_text += f"\nActive Experts (avg weight > 0.001):\n"
    stats_text += f"  High-Risk: {nonzero_high}/{n_experts}\n"
    stats_text += f"  Low-Risk: {nonzero_low}/{n_experts}\n"
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved gating heatmap to {output_path}')
    plt.close()

    # Save weight-difference plot as a separate PNG for cleaner inspection.
    if diff_labels is not None and diff_values is not None:
        diff_output_path = output_path.replace('gating_heatmap', 'expert_preference_difference')
        fig_diff, ax_diff = plt.subplots(figsize=(max(10, len(diff_labels) * 0.45), 6))
        x_diff = np.arange(len(diff_labels))
        diff_colors = ['#e74c3c' if d > 0 else '#3498db' for d in diff_values]
        bars = ax_diff.bar(x_diff, diff_values, color=diff_colors, alpha=0.8, edgecolor='black', linewidth=1.0)

        for bar, val in zip(bars, diff_values):
            height = bar.get_height()
            ax_diff.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f'{val:.3f}',
                ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=12,
                fontweight='bold',
            )

        ax_diff.set_xticks(x_diff)
        ax_diff.set_xticklabels(diff_labels, rotation=0, ha='center', fontsize=12)
        ax_diff.set_xlabel('')
        ax_diff.set_ylabel('Weight Difference (High Risk - Low Risk)', fontsize=13, labelpad=18)
        ax_diff.set_title('')
        ax_diff.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax_diff.grid(axis='y', alpha=0.3)
        ax_diff.tick_params(axis='y', labelsize=12)
        ax_diff.text(0.99, 0.96, 'High-risk preference (+)', transform=ax_diff.transAxes,
                     fontsize=14, color='#e74c3c', va='top', ha='right')
        ax_diff.text(0.99, 0.04, 'Low-risk preference (-)', transform=ax_diff.transAxes,
                     fontsize=14, color='#3498db', va='bottom', ha='right')
        fig_diff.tight_layout()
        fig_diff.savefig(diff_output_path, dpi=300, bbox_inches='tight')
        print(f'Saved expert preference difference plot to {diff_output_path}')
        plt.close(fig_diff)
    
    # Return statistics for further analysis
    return {
        'avg_high': avg_high,
        'avg_low': avg_low,
        'usage_high': usage_high,
        'usage_low': usage_low,
        'entropy_high': entropy_high,
        'entropy_low': entropy_low,
        'collapse_high': collapse_high,
        'collapse_low': collapse_low,
    }


def plot_individual_patient_gates(gate_weights, risk_scores, expert_names=None,
                                   output_path='results/moe_gates/individual_gates.png',
                                   n_samples=50):
    """Plot heatmap showing gating weights for individual patients, sorted by risk.
    
    Args:
        gate_weights: (n_patients, n_experts) array
        risk_scores: (n_patients,) array
        expert_names: List of expert names (optional)
        output_path: Path to save the figure
        n_samples: Number of patients to visualize (top/bottom by risk)
    """
    n_patients, n_experts = gate_weights.shape
    
    # Sort patients by risk score
    sorted_idx = np.argsort(risk_scores)
    
    # Select top and bottom patients
    if n_patients > n_samples:
        # Take top and bottom n_samples/2
        top_idx = sorted_idx[-n_samples//2:]
        bottom_idx = sorted_idx[:n_samples//2]
        selected_idx = np.concatenate([top_idx, bottom_idx])
    else:
        selected_idx = sorted_idx
    
    gate_subset = gate_weights[selected_idx]
    risk_subset = risk_scores[selected_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, max(10, len(selected_idx) * 0.3)))
    
    # Expert labels
    if expert_names is None:
        expert_labels = [f'E{i}' for i in range(n_experts)]
    else:
        expert_labels = [name[:15] for name in expert_names]
    
    # Patient labels with risk scores
    patient_labels = [f'P{i} (risk={risk_subset[i]:.2f})' for i in range(len(selected_idx))]
    
    # Plot heatmap
    sns.heatmap(gate_subset,
                xticklabels=expert_labels,
                yticklabels=patient_labels,
                cmap='viridis',
                cbar_kws={'label': 'Gating Weight'},
                ax=ax)
    
    ax.set_title(f'Individual Patient Gating Patterns (n={len(selected_idx)}, sorted by risk)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Expert Index', fontsize=12)
    ax.set_ylabel('Patient (Lower = Lower Risk)', fontsize=12)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved individual patient gates to {output_path}')
    plt.close()


# --- Clinical interpretability plotting functions ---
def plot_expert_clinical_profiles(gate_weights, X, column_names, output_path, top_n=3,
                                  expert_label_map=None, selected_experts=None):
    """Plot average clinical features for selected experts or top-N most used experts."""
    n_experts = gate_weights.shape[1]
    avg_weights = gate_weights.mean(axis=0)
    # Exclude SEX from profile visualization to reduce clutter.
    profile_indices = [i for i, c in enumerate(column_names) if c != 'SEX']
    profile_columns = [column_names[i] for i in profile_indices]
    if selected_experts:
        top_experts = [e for e in selected_experts if 0 <= int(e) < n_experts]
    else:
        top_experts = np.argsort(avg_weights)[-top_n:][::-1].tolist()
    if len(top_experts) == 0:
        print('No valid experts selected for clinical profile plot.')
        return

    fig, axes = plt.subplots(1, len(top_experts), figsize=(6 * len(top_experts), 6))
    if len(top_experts) == 1:
        axes = [axes]
    for i, expert in enumerate(top_experts):
        assigned = np.argmax(gate_weights, axis=1) == expert
        if assigned.sum() == 0:
            continue
        # Prefer raw (unnormalized) feature values for clinical profiles when available
        if hasattr(plot_expert_clinical_profiles, 'X_raw') and plot_expert_clinical_profiles.X_raw is not None:
            X_used = plot_expert_clinical_profiles.X_raw
        else:
            X_used = X
        mean_profile = X_used[assigned][:, profile_indices].mean(axis=0)
        ax = axes[i]
        ax.barh(range(len(profile_columns)), mean_profile, color='#6fa8dc', alpha=0.8)
        ax.set_yticks(range(len(profile_columns)))
        ax.set_yticklabels(profile_columns, fontsize=10)
        ax.set_title(f'{_expert_label(expert, expert_label_map)}: n={assigned.sum()}', fontsize=14)
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved expert clinical profiles to {output_path}')
    plt.close()

def _compute_subgroup_expert_fractions(gate_weights, masks, mode='weighted'):
    """Compute per-subgroup expert fractions using weighted or hard assignments."""
    n_groups = len(masks)
    n_experts = gate_weights.shape[1]
    counts = np.zeros((n_groups, n_experts), dtype=float)
    for i, mask in enumerate(masks):
        if np.sum(mask) == 0:
            continue
        gw = gate_weights[mask]
        if mode == 'hard':
            assigned = np.argmax(gw, axis=1)
            counts[i] = np.bincount(assigned, minlength=n_experts)
        else:
            # Weighted mode preserves top-k probability mass and avoids argmax collapse artifacts.
            counts[i] = gw.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        counts = counts / counts.sum(axis=1, keepdims=True)
    return np.nan_to_num(counts)


def _compute_expert_quartile_signatures(gate_weights, values, bins=4):
    """Return expert signatures over quartiles for pattern matching."""
    quantiles = np.percentile(values, np.linspace(0, 100, bins + 1))
    masks = []
    for i in range(bins):
        if i < bins - 1:
            mask = (values >= quantiles[i]) & (values < quantiles[i + 1])
        else:
            mask = (values >= quantiles[i])
        masks.append(mask)
    counts = _compute_subgroup_expert_fractions(gate_weights, masks, mode='weighted')  # [bins, experts]
    sig = counts.T  # [experts, bins]
    norms = np.linalg.norm(sig, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return sig / norms


def build_canonical_expert_map(current_gate_weights, ref_gate_weights, feature_values, canonical_expert_ids, bins=4):
    """Greedy one-to-one mapping from current experts to canonical ids by quartile-pattern similarity."""
    cur_sig = _compute_expert_quartile_signatures(current_gate_weights, feature_values, bins=bins)
    ref_sig = _compute_expert_quartile_signatures(ref_gate_weights, feature_values, bins=bins)
    mapping = {}
    used_current = set()
    for ref_id in canonical_expert_ids:
        if ref_id < 0 or ref_id >= ref_sig.shape[0]:
            continue
        sims = cur_sig @ ref_sig[ref_id]
        order = np.argsort(sims)[::-1]
        chosen = None
        for c in order:
            ci = int(c)
            if ci not in used_current:
                chosen = ci
                break
        if chosen is not None:
            mapping[chosen] = int(ref_id)
            used_current.add(chosen)
    return mapping


def plot_subgroup_expert_assignment(gate_weights, X, column_names, output_path, feature='AGE', bins=4,
                                    legend_threshold=0.01, xlabel=None, mode='weighted', expert_label_map=None):
    """Plot expert assignment distribution by clinical subgroup (e.g., age quartiles)."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    if feature not in column_names:
        print(f'Feature {feature} not found in columns.')
        return
    idx = column_names.index(feature)
    # Use unnormalized values if available (map patients back to original values)
    if hasattr(plot_subgroup_expert_assignment, 'X_raw') and plot_subgroup_expert_assignment.X_raw is not None:
        values = plot_subgroup_expert_assignment.X_raw[:, idx]
    else:
        values = X[:, idx]
    quantiles = np.percentile(values, np.linspace(0, 100, bins+1))
    subgroup_labels = [f'Q{i+1}' for i in range(bins)]
    subgroup_masks = []
    for i in range(bins):
        mask = (values >= quantiles[i]) & (values < quantiles[i+1]) if i < bins-1 else (values >= quantiles[i])
        subgroup_masks.append(mask)
    counts = _compute_subgroup_expert_fractions(gate_weights, subgroup_masks, mode=mode)
    # Only include experts with max fraction >= threshold
    max_fractions = counts.max(axis=0)
    present_experts = np.where(max_fractions >= legend_threshold)[0]
    n_present = len(present_experts)
    palette = sns.color_palette('Set2', n_present)
    plt.figure(figsize=(3.8+1.0*n_present, 3.5))
    bottom = np.zeros(bins)
    handles = []
    for i, expert in enumerate(present_experts):
        bar = plt.bar(
            subgroup_labels, counts[:, expert], bottom=bottom,
            label=_expert_label(expert, expert_label_map), color=palette[i],
        )
        handles.append(bar)
        bottom += counts[:, expert]
    ylabel = 'Fraction Assigned' if mode == 'hard' else 'Fraction Gate Mass'
    plt.ylabel(ylabel, fontsize=18, labelpad=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # Derive a sensible x-label from the feature if not provided
    label = xlabel if xlabel is not None else f'{feature} Quartile'
    plt.xlabel(label, fontsize=18, labelpad=14)
    plt.legend(
        handles=[h[0] for h in handles],
        labels=[_expert_label(e, expert_label_map) for e in present_experts],
        bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15, title='Expert',
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved subgroup expert assignment plot to {output_path}')
    plt.close()


def plot_sysbp_expert_assignment(gate_weights, X, column_names, output_path, bins=4, legend_threshold=0.01,
                                 mode='weighted', expert_label_map=None):
    """Wrapper to plot SYSBP quartile assignments (follows AGE plotting style)."""
    if 'SYSBP' not in column_names:
        print('SYSBP column not found.')
        return
    # Reuse the generic subgroup plot (it will prefer raw values if attached)
    plot_subgroup_expert_assignment(gate_weights, X, column_names, output_path,
                                    feature='SYSBP', bins=bins,
                                    legend_threshold=legend_threshold,
                                    xlabel='Systolic Blood Pressure Quartile (mm Hg)',
                                    mode=mode,
                                    expert_label_map=expert_label_map)

def plot_diabetes_expert_assignment(gate_weights, X, column_names, output_path, legend_threshold=0.01,
                                    mode='weighted', expert_label_map=None):
    # Find diabetes column
    if 'DIABETES' not in column_names:
        print('DIABETES column not found.')
        return
    idx = column_names.index('DIABETES')
    # Prefer raw unnormalized values if available; otherwise use X
    if hasattr(plot_diabetes_expert_assignment, 'X_raw') and plot_diabetes_expert_assignment.X_raw is not None:
        diabetes_raw = plot_diabetes_expert_assignment.X_raw[:, idx]
    else:
        diabetes_raw = X[:, idx]

    # Coerce raw sample values to binary: 0 => non-diabetic, anything else => diabetic
    # This ensures we bin by the original sample label even if values are floats or scaled.
    diabetes_bin = np.zeros_like(diabetes_raw, dtype=int)
    try:
        # Treat NaN as non-diabetic (0)
        mask_nonzero = ~np.isclose(diabetes_raw, 0) & ~np.isnan(diabetes_raw)
        diabetes_bin[mask_nonzero] = 1
    except Exception:
        # Fallback: any non-zero truthy value is diabetic
        diabetes_bin = (diabetes_raw != 0).astype(int)

    unique_vals = np.unique(diabetes_bin)
    if not np.array_equal(np.sort(unique_vals), np.array([0]) ) and not np.array_equal(np.sort(unique_vals), np.array([0, 1])):
        print(f'Warning: DIABETES column coerced to unexpected binary values: {unique_vals}. Proceeding with bins {unique_vals}.')

    subgroup_labels = ['Non-Diabetic', 'Diabetic']
    subgroup_masks = []
    for val in [0, 1]:
        mask = (diabetes_bin == val)
        subgroup_masks.append(mask)
    counts = _compute_subgroup_expert_fractions(gate_weights, subgroup_masks, mode=mode)
    max_fractions = counts.max(axis=0)
    present_experts = np.where(max_fractions >= legend_threshold)[0]
    n_present = len(present_experts)
    palette = sns.color_palette('Set2', n_present)
    plt.figure(figsize=(3.8+1.0*n_present, 3.5))
    bottom = np.zeros(2)
    handles = []
    for i, expert in enumerate(present_experts):
        bar = plt.bar(
            subgroup_labels, counts[:, expert], bottom=bottom,
            label=_expert_label(expert, expert_label_map), color=palette[i],
        )
        handles.append(bar)
        bottom += counts[:, expert]
    ylabel = 'Fraction Assigned' if mode == 'hard' else 'Fraction Gate Mass'
    plt.ylabel(ylabel, fontsize=18, labelpad=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # Use a clear, consistent x-label (match style used for age quartiles)
    plt.xlabel('Diabetes Status', fontsize=18, labelpad=14)
    plt.legend(
        handles=[h[0] for h in handles],
        labels=[_expert_label(e, expert_label_map) for e in present_experts],
        bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15, title='Expert',
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved diabetes subgroup expert assignment plot to {output_path}')
    plt.close()


def plot_ldl_expert_assignment(gate_weights, X, column_names, output_path, bins=4, legend_threshold=0.01,
                               mode='weighted', expert_label_map=None):
    """Plot expert assignment distribution by LDL subgroup (quartiles by default).

    Mirrors `plot_subgroup_expert_assignment` but searches for common LDL column names
    and labels the x-axis as 'LDL Quartile'. Prefers raw values attached at runtime.
    """
    # Detect LDL column robustly: prefer exact matches, otherwise any column containing 'ldl' (case-insensitive)
    idx = None
    found_name = None
    # First try common exact names
    possible_names = ['LDLC']
    for name in possible_names:
        if name in column_names:
            idx = column_names.index(name)
            found_name = name
            break
    # If not found, try case-insensitive substring match
    if idx is None:
        lower_cols = [c.lower() for c in column_names]
        ldl_candidates = [i for i, c in enumerate(lower_cols) if 'ldl' in c]
        if len(ldl_candidates) > 0:
            idx = ldl_candidates[0]
            found_name = column_names[idx]
        else:
            # Provide helpful debugging output listing available columns (shortened)
            sample_cols = ', '.join(column_names)
            print('LDL column not found. Available columns (first 30):')
            print(sample_cols)
            return

    # Prefer raw (unnormalized) values when available
    if hasattr(plot_ldl_expert_assignment, 'X_raw') and plot_ldl_expert_assignment.X_raw is not None:
        values = plot_ldl_expert_assignment.X_raw[:, idx]
    else:
        values = X[:, idx]

    quantiles = np.percentile(values, np.linspace(0, 100, bins+1))
    subgroup_labels = [f'Q{i+1}' for i in range(bins)]
    subgroup_masks = []
    for i in range(bins):
        if i < bins-1:
            mask = (values >= quantiles[i]) & (values < quantiles[i+1])
        else:
            mask = (values >= quantiles[i])
        subgroup_masks.append(mask)
    counts = _compute_subgroup_expert_fractions(gate_weights, subgroup_masks, mode=mode)

    max_fractions = counts.max(axis=0)
    present_experts = np.where(max_fractions >= legend_threshold)[0]
    n_present = len(present_experts)
    if n_present == 0:
        present_experts = np.arange(n_experts)
        n_present = n_experts

    palette = sns.color_palette('Set2', n_present)
    plt.figure(figsize=(3.8+1.0*n_present, 3.5))
    bottom = np.zeros(bins)
    handles = []
    for i, expert in enumerate(present_experts):
        bar = plt.bar(
            subgroup_labels, counts[:, expert], bottom=bottom,
            label=_expert_label(expert, expert_label_map), color=palette[i],
        )
        handles.append(bar)
        bottom += counts[:, expert]

    ylabel = 'Fraction Assigned' if mode == 'hard' else 'Fraction Gate Mass'
    plt.ylabel(ylabel, fontsize=18, labelpad=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('LDL Quartile', fontsize=18, labelpad=14)
    plt.legend(
        handles=[h[0] for h in handles],
        labels=[_expert_label(e, expert_label_map) for e in present_experts],
        bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=15, title='Expert',
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved LDL subgroup expert assignment plot to {output_path} (column: {found_name})')
    plt.close()

def plot_patient_case_study(gate_weights, X, column_names, risk_scores, output_path, patient_idx):
    """Plot clinical features and gating weights for a single patient."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(range(gate_weights.shape[1]), gate_weights[patient_idx], color='#b4a7d6')
    ax1.set_xlabel('Expert Index')
    ax1.set_ylabel('Gating Weight')
    ax1.set_title(f'Patient {patient_idx} Gating (Risk={risk_scores[patient_idx]:.2f})')
    ax2 = ax1.twinx()
    ax2.plot(range(len(column_names)), X[patient_idx], 'o-', color='#6fa8dc')
    ax2.set_ylabel('Clinical Feature Value')
    ax2.set_xticks(range(len(column_names)))
    ax2.set_xticklabels(column_names, rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved patient case study plot to {output_path}')
    plt.close()

def print_column_info(X, column_names):
    print('Available columns and unique values:')
    for i, col in enumerate(column_names):
        unique_vals = set(X[:, i])
        print(f'{col}: {sorted(list(unique_vals))[:10]}')

def main():
    args = parse_args()
    
    # Set CUDA device when available; otherwise run on CPU.
    if torch.cuda.is_available():
        torch.cuda.set_device(args.cuda_device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print('=' * 80)
    print('MoE GATING ANALYSIS')
    print('=' * 80)
    print(f'Dataset: {args.dataset}')
    print(f'Seed: {args.seed}')
    print(f'Num Experts: {args.num_experts}')
    print(f'Top-K: {args.top_k}')
    print('=' * 80)
    
    # Load data
    print('\nLoading data...')
    
    # Mock args object for load_data function
    class DataArgs:
        pass
    
    data_args = DataArgs()
    data_args.dataset = args.dataset
    data_args.is_normalize = args.normalize
    data_args.is_cluster = True
    data_args.is_generate_sim = False
    data_args.is_save_sim = False
    data_args.num_inst = 200
    data_args.num_feat = 10

    loaded = load_data(data_args, random_state=args.seed)
    if len(loaded) == 7:
        X_train, X_val, X_test, y_train, y_val, y_test, column_names = loaded
    elif len(loaded) == 5:
        X_train, X_test, y_train, y_test, column_names = loaded
        # Backward-compatible fallback for call sites expecting validation split.
        X_val, y_val = X_test, y_test
    else:
        raise ValueError(f'Unsupported load_data return length: {len(loaded)}')

    print(
        f'Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}',
    )
    
    # Prepare data
    e_train = np.array([[item[0] * 1 for item in y_train]]).T
    t_train = np.array([[item[1] for item in y_train]]).T
    e_test = np.array([[item[0] * 1 for item in y_test]]).T
    t_test = np.array([[item[1] for item in y_test]]).T
    
    if args.normalize:
        print(
            'Note: covariates are already train-only scaled in load_data; '
            'skipping duplicate StandardScaler.',
        )

    # Load truly unnormalized covariates for interpretability plots.
    # Keep model inference inputs controlled by `args.normalize`.
    try:
        raw_data_args = DataArgs()
        raw_data_args.dataset = args.dataset
        raw_data_args.is_normalize = False
        raw_data_args.is_cluster = True
        raw_data_args.is_generate_sim = False
        raw_data_args.is_save_sim = False
        raw_data_args.num_inst = 200
        raw_data_args.num_feat = 10
        loaded_raw = load_data(raw_data_args, random_state=args.seed)
        if len(loaded_raw) == 7:
            _X_train_raw, _X_val_raw, X_test_raw, _y_train_raw, _y_val_raw, _y_test_raw, _column_names_raw = loaded_raw
        elif len(loaded_raw) == 5:
            _X_train_raw, X_test_raw, _y_train_raw, _y_test_raw, _column_names_raw = loaded_raw
        else:
            raise ValueError(f'Unsupported raw load_data return length: {len(loaded_raw)}')
        print('Loaded unnormalized covariates for subgroup/profile plots.')
    except Exception as e:
        print(f'Warning: failed to load unnormalized covariates ({e}); falling back to current X_test values.')
        try:
            X_test_raw = X_test.copy() if isinstance(X_test, np.ndarray) else np.asarray(X_test).copy()
        except Exception:
            X_test_raw = np.asarray(X_test).copy()

    # Attach raw test features to plotting functions so they prefer unnormalized values
    plot_expert_clinical_profiles.X_raw = X_test_raw
    plot_subgroup_expert_assignment.X_raw = X_test_raw
    plot_diabetes_expert_assignment.X_raw = X_test_raw
    plot_ldl_expert_assignment.X_raw = X_test_raw
    plot_patient_case_study.X_raw = X_test_raw
    plot_sysbp_expert_assignment.X_raw = X_test_raw
    
    # Handle region experts if needed
    expert_feature_splits = None
    expert_names = None
    if args.moe_region_experts:
        aal_supported = {'AAL-AV45', 'AAL-FDG', 'AAL-VBM'}
        if args.dataset in aal_supported and column_names:
            print('\nBuilding region-specific expert assignments...')
            splits, names, unmatched = build_brain_region_expert_splits(
                column_names,
                lookup_path=args.region_lookup_path,
                grouping_col=args.region_group_column
            )
            expert_feature_splits = splits
            expert_names = names
            args.num_experts = len(expert_feature_splits)
            print(f'Created {len(expert_feature_splits)} region-based experts')
            if unmatched:
                print(f'Warning: {len(unmatched)} ROIs not matched to regions')
    
    # Load or train model
    print('\n' + '=' * 80)
    model, train_metrics = load_or_train_model(
        args, X_train, X_test, y_train, y_test, expert_feature_splits, expert_names,
        X_val=X_val, y_val=y_val,
    )
    print('=' * 80)
    
    # Extract gating weights for test set
    print('\nExtracting gating weights from test set...')
    gate_weights_test = extract_gate_weights(model, X_test)
    print(f'Gate weights shape: {gate_weights_test.shape}')
    
    # Compute risk scores
    print('Computing risk scores...')
    risk_scores = compute_risk_scores(model, X_test)
    print(f'Risk scores range: [{risk_scores.min():.3f}, {risk_scores.max():.3f}]')
    
    # Stratify by risk
    # Prefer ADACSM cluster assignments when available on the model; otherwise fall back to percentile split
    cluster_labels = None
    try:
        if hasattr(model, 'predict_cluster'):
            cluster_labels = np.asarray(model.predict_cluster(X_test))
        elif hasattr(model, 'cluster_assignments'):
            cluster_labels = np.asarray(model.cluster_assignments)
        elif hasattr(model, 'cluster_labels'):
            cluster_labels = np.asarray(model.cluster_labels)
    except Exception:
        cluster_labels = None

    if cluster_labels is not None and len(cluster_labels) == len(risk_scores):
        print('Using model-provided cluster assignments for stratification.')
        uniq = np.unique(cluster_labels)
        # compute mean risk per cluster and pick cluster with largest mean risk as 'high-risk'
        mean_risk = {c: risk_scores[cluster_labels == c].mean() for c in uniq}
        high_cluster = max(mean_risk, key=mean_risk.get)
        high_risk_idx = np.where(cluster_labels == high_cluster)[0]
        low_risk_idx = np.where(cluster_labels != high_cluster)[0]
        print(f'High-risk cluster: {high_cluster} (n={len(high_risk_idx)})')
    else:
        print(f'Stratifying patients at {args.risk_percentile}th percentile...')
        high_risk_idx, low_risk_idx = stratify_by_risk(risk_scores, percentile=args.risk_percentile)
        print(f'High-risk patients: {len(high_risk_idx)}, Low-risk patients: {len(low_risk_idx)}')
    
    gate_weights_high = gate_weights_test[high_risk_idx]
    gate_weights_low = gate_weights_test[low_risk_idx]
    
    # Get expert names from model if available
    if expert_names is None and hasattr(model, 'expert_names') and model.expert_names:
        expert_names = model.expert_names

    # Optional remapping of current experts to canonical expert ids by subgroup pattern similarity.
    expert_label_map = None
    if args.canonical_reference_model and os.path.exists(args.canonical_reference_model):
        try:
            canonical_ids = [int(x.strip()) for x in args.canonical_experts.split(',') if x.strip()]
            with open(args.canonical_reference_model, 'rb') as f:
                ref_model = pkl.load(f)
            ref_gate_weights = extract_gate_weights(ref_model, X_test)
            if 'AGE' in column_names:
                age_idx = column_names.index('AGE')
                age_values = X_test_raw[:, age_idx]
                expert_label_map = build_canonical_expert_map(
                    gate_weights_test, ref_gate_weights, age_values, canonical_ids, bins=4,
                )
                if expert_label_map:
                    print(f'Canonical expert remap (current -> canonical): {expert_label_map}')
        except Exception as e:
            print(f'Warning: failed to apply canonical expert remap ({e})')
    
    # Generate visualizations
    print('\n' + '=' * 80)
    print('GENERATING VISUALIZATIONS')
    print('=' * 80)
    
    # Main heatmap
    output_file = os.path.join(args.output_dir, 
                               f'gating_heatmap_{args.dataset}_seed{args.seed}.png')
    stats = plot_gating_heatmap(gate_weights_high, gate_weights_low, 
                               expert_names=expert_names,
                               expert_label_map=expert_label_map,
                               output_path=output_file,
                               top_k=args.top_k)
    
    # Individual patient plot
    output_file2 = os.path.join(args.output_dir,
                                f'individual_gates_{args.dataset}_seed{args.seed}.png')
    plot_individual_patient_gates(gate_weights_test, risk_scores,
                                 expert_names=expert_names,
                                 output_path=output_file2,
                                 n_samples=min(50, len(risk_scores)))
    
    # Clinical interpretability plots
    print('\nGenerating clinical interpretability plots...')
    selected_profile_experts = None
    if args.profile_experts:
        try:
            selected_profile_experts = [int(x.strip()) for x in args.profile_experts.split(',') if x.strip()]
        except Exception:
            selected_profile_experts = None
    # 1. Expert-wise clinical profiles (top-3 experts)
    output_file3 = os.path.join(args.output_dir, f'expert_clinical_profiles_{args.dataset}_seed{args.seed}.png')
    plot_expert_clinical_profiles(
        gate_weights_test, X_test, column_names, output_file3, top_n=3,
        expert_label_map=expert_label_map, selected_experts=selected_profile_experts,
    )
    # 2. Age expert assignment (AGE quartiles)
    output_file4 = os.path.join(args.output_dir, f'age_expert_assignment_{args.dataset}_seed{args.seed}.png')
    plot_subgroup_expert_assignment(
        gate_weights_test, X_test, column_names, output_file4, feature='AGE', bins=4,
        mode=args.subgroup_mode, legend_threshold=args.legend_threshold,
        expert_label_map=expert_label_map,
    )
    # 2b. SYSBP expert assignment (Systolic Blood Pressure quartiles)
    output_file_sysbp = os.path.join(args.output_dir, f'sysbp_expert_assignment_{args.dataset}_seed{args.seed}.png')
    plot_sysbp_expert_assignment(
        gate_weights_test, X_test, column_names, output_file_sysbp, bins=4,
        mode=args.subgroup_mode, legend_threshold=args.legend_threshold,
        expert_label_map=expert_label_map,
    )
    # 3. Diabetes subgroup expert assignment
    output_file7 = os.path.join(args.output_dir, f'diabetes_expert_assignment_{args.dataset}_seed{args.seed}.png')
    plot_diabetes_expert_assignment(
        gate_weights_test, X_test, column_names, output_file7,
        mode=args.subgroup_mode, legend_threshold=args.legend_threshold,
        expert_label_map=expert_label_map,
    )
    # 4. LDL subgroup expert assignment (quartiles)
    output_file_ldl = os.path.join(args.output_dir, f'ldl_expert_assignment_{args.dataset}_seed{args.seed}.png')
    plot_ldl_expert_assignment(
        gate_weights_test, X_test, column_names, output_file_ldl, bins=4,
        mode=args.subgroup_mode, legend_threshold=args.legend_threshold,
        expert_label_map=expert_label_map,
    )
    # 4. Patient-level case studies (highest and lowest risk)
    if column_names and len(column_names) == X_test.shape[1]:
        idx_high = np.argmax(risk_scores)
        idx_low = np.argmin(risk_scores)
        output_file5 = os.path.join(args.output_dir, f'patient_case_highrisk_{args.dataset}_seed{args.seed}.png')
        output_file6 = os.path.join(args.output_dir, f'patient_case_lowrisk_{args.dataset}_seed{args.seed}.png')
        plot_patient_case_study(gate_weights_test, X_test, column_names, risk_scores, output_file5, idx_high)
        plot_patient_case_study(gate_weights_test, X_test, column_names, risk_scores, output_file6, idx_low)
    else:
        print('Skipping patient case-study plots: feature names unavailable or dimension mismatch.')
    
    # Print summary
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f"High-risk collapse (top-3 usage): {stats['collapse_high']:.1f}%")
    print(f"Low-risk collapse (top-3 usage): {stats['collapse_low']:.1f}%")
    print(f"High-risk entropy: {stats['entropy_high']:.3f}")
    print(f"Low-risk entropy: {stats['entropy_low']:.3f}")
    
    if stats['collapse_high'] > 60 or stats['collapse_low'] > 60:
        print("\n⚠️  WARNING: Expert collapse detected!")
        print("Most patients are being routed to the same 2-3 experts.")
        print("This may explain flat C-index across time horizons.")
        print("\nSuggestions:")
        print("  1. Reduce top_k to force diversity")
        print("  2. Increase gate_temperature to soften routing decisions")
        print("  3. Increase load_balance_lambda to encourage expert diversity")
        print("  4. Try different num_experts")
    else:
        print("\n✓ Experts show reasonable diversity in routing patterns.")
    
    print('\n' + '=' * 80)
    print('Done!')
    print('=' * 80)


if __name__ == '__main__':
    main()
