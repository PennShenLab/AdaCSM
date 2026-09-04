#!/usr/bin/env python3
"""Generate SUPPORT expert-assignment plots for dementia and meanbp."""

from __future__ import annotations

import io
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.data_utils import load_data  # noqa: E402
from visualize_moe_gates import extract_gate_weights, plot_subgroup_expert_assignment  # noqa: E402


class CPUUnpickler(pickle.Unpickler):
    """Load CUDA-pickled models on CPU-only environments."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def load_model_cpu(model_path: Path):
    with model_path.open("rb") as f:
        try:
            model = pickle.load(f)
        except Exception:
            f.seek(0)
            model = CPUUnpickler(f).load()
    if hasattr(model, "set_device"):
        try:
            model.set_device("cpu")
        except Exception:
            pass
    return model


def plot_binary_subgroup(gate_weights, x, column_names, feature_name, output_path):
    """Plot weighted expert mass for a binary feature (0/1)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    if feature_name not in column_names:
        print(f"Feature {feature_name} not found in columns.")
        return
    idx = column_names.index(feature_name)
    values = x[:, idx]
    values_bin = (values != 0).astype(int)

    masks = [values_bin == 0, values_bin == 1]
    labels = [f"{feature_name}=0", f"{feature_name}=1"]
    n_experts = gate_weights.shape[1]
    counts = np.zeros((2, n_experts), dtype=float)
    for i, m in enumerate(masks):
        if np.any(m):
            counts[i] = gate_weights[m].sum(axis=0)
    counts = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)

    present = np.where(counts.max(axis=0) >= 0.01)[0]
    palette = sns.color_palette("Set2", len(present))
    plt.figure(figsize=(4.8 + 0.7 * len(present), 4.0))
    bottom = np.zeros(2)
    for i, e in enumerate(present):
        plt.bar(labels, counts[:, e], bottom=bottom, color=palette[i], label=f"Expert {e}")
        bottom += counts[:, e]
    plt.ylabel("Fraction Gate Mass", fontsize=16)
    plt.xlabel("Dementia Status", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12, title="Expert")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved dementia subgroup expert assignment plot to {output_path}")


def main() -> None:
    model_path = REPO_ROOT / "models" / "ADACSM_support_seed42_numexperts32_topk2.pkl"
    output_dir = REPO_ROOT / "results" / "interpretability_support_seed42"
    output_dir.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        dataset="support",
        is_normalize=True,
        is_cluster=True,
        is_generate_sim=False,
        is_save_sim=False,
    )
    raw_args = SimpleNamespace(
        dataset="support",
        is_normalize=False,
        is_cluster=True,
        is_generate_sim=False,
        is_save_sim=False,
    )

    x_train, x_test, y_train, y_test, column_names = load_data(args, random_state=42)
    _x_train_raw, x_test_raw, _y_train_raw, _y_test_raw, _raw_cols = load_data(
        raw_args, random_state=42
    )

    model = load_model_cpu(model_path)
    gate_weights = extract_gate_weights(model, x_test)
    plot_subgroup_expert_assignment.X_raw = x_test_raw

    out_meanbp = output_dir / "meanbp_expert_assignment_support_seed42.png"
    out_dementia = output_dir / "dementia_expert_assignment_support_seed42.png"

    plot_subgroup_expert_assignment(
        gate_weights,
        x_test,
        column_names,
        str(out_meanbp),
        feature="meanbp",
        bins=4,
        mode="weighted",
        legend_threshold=0.01,
        xlabel="Mean Blood Pressure Quartile",
    )
    plot_binary_subgroup(
        gate_weights=gate_weights,
        x=x_test_raw,
        column_names=column_names,
        feature_name="dementia",
        output_path=str(out_dementia),
    )

    print(f"Saved: {out_meanbp}")
    print(f"Saved: {out_dementia}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
