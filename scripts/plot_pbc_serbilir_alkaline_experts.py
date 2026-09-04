#!/usr/bin/env python3
"""Generate PBC expert-assignment subgroup plots for serBilir and alkaline."""

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


def main() -> None:
    model_path = REPO_ROOT / "models" / "ADACSM_PBC_seed42_moe_topk2.pkl"
    output_dir = REPO_ROOT / "results" / "interpretability_pbc_seed42"
    output_dir.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        dataset="PBC",
        is_normalize=True,
        is_cluster=True,
        is_generate_sim=False,
        is_save_sim=False,
    )
    raw_args = SimpleNamespace(
        dataset="PBC",
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

    out_serbilir = output_dir / "serbilir_expert_assignment_PBC_seed42.png"
    out_alkaline = output_dir / "alkaline_expert_assignment_PBC_seed42.png"

    plot_subgroup_expert_assignment(
        gate_weights,
        x_test,
        column_names,
        str(out_serbilir),
        feature="serBilir",
        bins=4,
        mode="weighted",
        legend_threshold=0.01,
        xlabel="serBilir Quartile",
    )
    plot_subgroup_expert_assignment(
        gate_weights,
        x_test,
        column_names,
        str(out_alkaline),
        feature="alkaline",
        bins=4,
        mode="weighted",
        legend_threshold=0.01,
        xlabel="alkaline Quartile",
    )

    print(f"Saved: {out_serbilir}")
    print(f"Saved: {out_alkaline}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
