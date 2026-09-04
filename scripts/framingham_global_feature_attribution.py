#!/usr/bin/env python3
"""Compare intrinsic expert profiles vs SHAP on FRAMINGHAM.

This script produces:
1) A publication-ready figure comparing global feature importance from:
   - Intrinsic expert profiles (from MoE routing and expert-conditioned means)
   - Post-hoc SHAP values (KernelExplainer on model risk output)
2) A CSV table with normalized scores and ranks
3) A text summary with alignment metrics (Spearman and top-k overlap)

Run with near-all-sample SHAP:
KMP_DUPLICATE_LIB_OK=TRUE conda run -n dcsm python scripts/framingham_global_feature_attribution.py --background_size 300 --explain_size 3489 --nsamples 300
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Prevent OpenMP duplicate runtime crashes on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure repo root imports work when launched from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.data_utils import load_data  # noqa: E402


class CPUUnpickler(pickle.Unpickler):
    """Load CUDA-pickled models on CPU-only machines."""

    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        return super().find_class(module, name)


def load_model_cpu(model_path: Path):
    """Load a potentially CUDA-pickled model in a CPU-safe way."""
    with model_path.open("rb") as handle:
        try:
            model = pickle.load(handle)
        except Exception:
            handle.seek(0)
            model = CPUUnpickler(handle).load()
    # Force CPU execution for checkpoints trained/saved on CUDA.
    try:
        if hasattr(model, "set_device"):
            model.set_device("cpu")
    except Exception:
        pass
    try:
        if hasattr(model, "torch_model") and model.torch_model is not None:
            model.torch_model = model.torch_model.to(torch.device("cpu"))
    except Exception:
        pass
    if hasattr(model, "cuda"):
        try:
            model.cuda = False
        except Exception:
            pass
    return model


def extract_gate_weights(model, x_data: np.ndarray) -> np.ndarray:
    """Extract gate weights from model.moe_layer for all samples."""
    model.torch_model.eval()

    # Avoid model._preprocess_test_data because older checkpoints may retain CUDA device metadata.
    x_tensor = torch.from_numpy(np.asarray(x_data))

    try:
        if hasattr(model.torch_model, "moe_layer") and model.torch_model.moe_layer is not None:
            param_iter = model.torch_model.moe_layer.parameters()
        else:
            param_iter = model.torch_model.parameters()
        first_param = next(param_iter, None)
        model_device = first_param.device if first_param is not None else torch.device("cpu")
    except Exception:
        model_device = torch.device("cpu")

    tensor_dtype = torch.float64
    try:
        if hasattr(model, "_tensor_dtype"):
            tensor_dtype = model._tensor_dtype()
    except Exception:
        tensor_dtype = torch.float64
    x_tensor = x_tensor.to(device=model_device, dtype=tensor_dtype)

    with torch.no_grad():
        if hasattr(model.torch_model, "moe_layer") and model.torch_model.moe_layer is not None:
            gate_weights = model.torch_model.moe_layer.inspect_gate_weights(x_tensor)
            return gate_weights.detach().cpu().numpy()
    raise RuntimeError("Model does not expose moe_layer.inspect_gate_weights.")


def compute_risk_scores(model, x: np.ndarray) -> np.ndarray:
    """Return 1D risk scores (larger means higher risk)."""
    if hasattr(model, "predict_mean"):
        pred_time = np.asarray(model.predict_mean(x)).reshape(-1)
        return -pred_time
    if hasattr(model, "predict_risk"):
        # Some model implementations require an explicit time horizon for predict_risk.
        t_horizon = float(np.median(np.asarray(getattr(model, "times", [1.0]))))
        risk = np.asarray(model.predict_risk(x, t=t_horizon)).reshape(-1)
        return risk
    if hasattr(model, "predict_survival"):
        pred = np.asarray(model.predict_survival(x, t=None)).reshape(-1)
        return -pred
    raise RuntimeError("Model does not expose predict_risk/predict_mean/predict_survival.")


def intrinsic_profile_importance(
    gate_weights: np.ndarray,
    x_raw: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    """Compute intrinsic importance from expert clinical profiles.

    Score(feature_j) = sum_e p(e) * |mu_ej - mu_j| / std_j
    where:
      - p(e): expert usage as hard top-1 assignment frequency
      - mu_ej: expert-conditioned mean for feature j
      - mu_j: global mean for feature j
      - std_j: global std for feature j (stabilized)
    """
    n_experts = gate_weights.shape[1]
    assigned = np.argmax(gate_weights, axis=1)
    global_mean = x_raw.mean(axis=0)
    global_std = x_raw.std(axis=0)
    global_std = np.where(global_std < 1e-8, 1.0, global_std)

    importance = np.zeros(x_raw.shape[1], dtype=float)
    for expert_idx in range(n_experts):
        mask = assigned == expert_idx
        if not np.any(mask):
            continue
        usage = float(np.mean(mask))
        expert_mean = x_raw[mask].mean(axis=0)
        standardized_shift = np.abs(expert_mean - global_mean) / global_std
        importance += usage * standardized_shift

    if np.allclose(importance.sum(), 0.0):
        raise RuntimeError("Intrinsic importance is all zeros; cannot rank features.")
    return importance


def shap_importance_kernel(
    predict_fn,
    x_model: np.ndarray,
    background_size: int,
    explain_size: int,
    nsamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute global SHAP importance via KernelExplainer on risk output."""
    import shap

    rng = np.random.default_rng(seed)
    n = x_model.shape[0]
    bg_idx = rng.choice(n, size=min(background_size, n), replace=False)
    ex_idx = rng.choice(n, size=min(explain_size, n), replace=False)
    x_background = x_model[bg_idx]
    x_explain = x_model[ex_idx]

    explainer = shap.KernelExplainer(predict_fn, x_background)
    shap_values = explainer.shap_values(x_explain, nsamples=nsamples)
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:  # older APIs can return (1, n, d)
        shap_values = shap_values[0]

    importance = np.mean(np.abs(shap_values), axis=0)
    if np.allclose(importance.sum(), 0.0):
        raise RuntimeError("SHAP importance is all zeros; cannot rank features.")
    return importance, ex_idx


def build_shap_target_fn(model, x_reference: np.ndarray):
    """Build a SHAP target function that is informative on this checkpoint.

    Preference order:
      1) Survival risk score if non-constant
      2) Routing preference between top-2 active experts (gate_i - gate_j)
    """
    probe_x = x_reference[: min(512, len(x_reference))]
    risk_probe = compute_risk_scores(model, probe_x)
    risk_std = float(np.std(risk_probe))
    if risk_std > 1e-8:
        def risk_fn(x_batch: np.ndarray) -> np.ndarray:
            return compute_risk_scores(model, np.asarray(x_batch))

        return risk_fn, "survival_risk", {"risk_std": risk_std}

    gate_probe = extract_gate_weights(model, probe_x)
    avg_gate = gate_probe.mean(axis=0)
    top2 = np.argsort(avg_gate)[-2:][::-1]
    if len(top2) < 2:
        raise RuntimeError("Could not identify two active experts for routing SHAP fallback.")
    e0, e1 = int(top2[0]), int(top2[1])

    def routing_fn(x_batch: np.ndarray) -> np.ndarray:
        gw = extract_gate_weights(model, np.asarray(x_batch))
        return gw[:, e0] - gw[:, e1]

    diagnostics = {
        "risk_std": risk_std,
        "routing_expert_a": e0,
        "routing_expert_b": e1,
        "routing_probe_std": float(np.std(gate_probe[:, e0] - gate_probe[:, e1])),
    }
    return routing_fn, f"routing_pref_expert{e0}_minus_expert{e1}", diagnostics


def normalize_importance(v: np.ndarray) -> np.ndarray:
    """Normalize vector to sum to 1."""
    s = float(np.sum(v))
    if s <= 0:
        return np.zeros_like(v)
    return v / s


def build_comparison_figure(
    df: pd.DataFrame,
    top_n: int,
    out_png: Path,
    out_pdf: Path,
    spearman: float,
    overlap_k: int,
    overlap_ratio: float,
) -> None:
    """Create conference-ready comparison figure."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "legend.fontsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.linewidth": 1.2,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )

    # Top features by average normalized importance from both methods.
    ranked = df.copy()
    ranked["mean_norm"] = 0.5 * (
        ranked["intrinsic_importance_norm"] + ranked["shap_importance_norm"]
    )
    ranked = ranked.sort_values("mean_norm", ascending=False).head(top_n).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Panel A: side-by-side bars
    ax0 = axes[0]
    y = np.arange(len(ranked))
    bar_h = 0.45
    ax0.barh(
        y - bar_h / 2,
        ranked["intrinsic_importance_norm"],
        height=bar_h,
        color="#4E79A7",
        alpha=0.9,
        label="Intrinsic Expert Profiles",
    )
    ax0.barh(
        y + bar_h / 2,
        ranked["shap_importance_norm"],
        height=bar_h,
        color="#E15759",
        alpha=0.9,
        label="Post-hoc SHAP",
    )
    ax0.set_yticks(y)
    ax0.set_yticklabels(ranked["feature"])
    ax0.set_xlabel("Normalized global importance")
    ax0.set_title("Top feature attribution comparison (FRAMINGHAM)")
    ax0.legend(frameon=False, loc="lower right", fontsize=18)

    # Panel B: agreement scatter for all features
    ax1 = axes[1]
    ax1.scatter(
        df["intrinsic_importance_norm"],
        df["shap_importance_norm"],
        c="#59A14F",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
        s=165,
    )
    ax1.set_xlabel("Intrinsic importance (normalized)")
    ax1.set_ylabel("SHAP importance (normalized)")
    ax1.set_title("Global alignment across all features")

    top_annotate = (
        df.sort_values("shap_importance_norm", ascending=False)
        .head(min(5, len(df)))
        .index.tolist()
    )
    for idx in top_annotate:
        row = df.loc[idx]
        ax1.annotate(
            row["feature"],
            (row["intrinsic_importance_norm"], row["shap_importance_norm"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=11,
        )

    text = (
        f"Spearman rank corr = {spearman:.3f}\n"
        f"Top-{overlap_k} overlap = {overlap_ratio:.1%}"
    )
    ax1.text(
        0.04,
        0.96,
        text,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=15,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "#999999"},
    )

    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.16, wspace=0.22)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="FRAMINGHAM",
        help="Dataset name. Default is FRAMINGHAM.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/ADACSM_FRAMINGHAM_seed42_moe_topk2_numexperts32.pkl",
        help="Path to trained AdaCSM model checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--background_size", type=int, default=80)
    parser.add_argument("--explain_size", type=int, default=200)
    parser.add_argument("--nsamples", type=int, default=300)
    parser.add_argument("--top_n_plot", type=int, default=12)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/interpretability_framingham_global",
    )
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FRAMINGHAM GLOBAL ATTRIBUTION: INTRINSIC PROFILES VS SHAP")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Output dir: {output_dir}")

    # Load normalized covariates for model inference.
    class DataArgs:
        pass

    data_args = DataArgs()
    data_args.dataset = args.dataset
    data_args.is_normalize = True
    data_args.is_cluster = True
    data_args.is_generate_sim = False
    data_args.is_save_sim = False
    x_train, x_test, _y_train, _y_test, column_names = load_data(data_args, random_state=args.seed)

    # Load raw covariates for profile interpretation.
    raw_data_args = DataArgs()
    raw_data_args.dataset = args.dataset
    raw_data_args.is_normalize = False
    raw_data_args.is_cluster = True
    raw_data_args.is_generate_sim = False
    raw_data_args.is_save_sim = False
    _x_train_raw, x_test_raw, _y_train_raw, _y_test_raw, _col_raw = load_data(
        raw_data_args, random_state=args.seed
    )

    model = load_model_cpu(REPO_ROOT / args.model_path)
    gate_weights = extract_gate_weights(model, x_test)
    print(f"Loaded test data: {x_test.shape}, gate weights: {gate_weights.shape}")

    intrinsic = intrinsic_profile_importance(gate_weights, x_test_raw, column_names)
    shap_target_fn, shap_target_name, shap_diag = build_shap_target_fn(model, x_test)
    shap_imp, explained_idx = shap_importance_kernel(
        predict_fn=shap_target_fn,
        x_model=x_test,
        background_size=args.background_size,
        explain_size=args.explain_size,
        nsamples=args.nsamples,
        seed=args.seed,
    )

    intrinsic_norm = normalize_importance(intrinsic)
    shap_norm = normalize_importance(shap_imp)

    df = pd.DataFrame(
        {
            "feature": column_names,
            "intrinsic_importance": intrinsic,
            "intrinsic_importance_norm": intrinsic_norm,
            "shap_importance": shap_imp,
            "shap_importance_norm": shap_norm,
        }
    )
    df["intrinsic_rank"] = df["intrinsic_importance_norm"].rank(method="average", ascending=False)
    df["shap_rank"] = df["shap_importance_norm"].rank(method="average", ascending=False)
    df["rank_gap"] = (df["intrinsic_rank"] - df["shap_rank"]).abs()
    df = df.sort_values("shap_importance_norm", ascending=False).reset_index(drop=True)

    spearman = float(df["intrinsic_rank"].corr(df["shap_rank"], method="spearman"))
    topk = min(10, len(df))
    top_intrinsic = set(df.nsmallest(topk, "intrinsic_rank")["feature"])
    top_shap = set(df.nsmallest(topk, "shap_rank")["feature"])
    overlap = sorted(top_intrinsic & top_shap)
    overlap_ratio = len(overlap) / max(1, topk)

    out_csv = output_dir / "framingham_intrinsic_vs_shap.csv"
    out_txt = output_dir / "framingham_intrinsic_vs_shap_summary.txt"
    out_png = output_dir / "framingham_intrinsic_vs_shap.png"
    out_pdf = output_dir / "framingham_intrinsic_vs_shap.pdf"

    df.to_csv(out_csv, index=False)
    build_comparison_figure(
        df=df,
        top_n=args.top_n_plot,
        out_png=out_png,
        out_pdf=out_pdf,
        spearman=spearman,
        overlap_k=topk,
        overlap_ratio=overlap_ratio,
    )

    summary_lines = [
        "Global feature attribution alignment (FRAMINGHAM)",
        f"model_path: {args.model_path}",
        f"shap_target: {shap_target_name}",
        f"n_test: {x_test.shape[0]}",
        f"n_features: {x_test.shape[1]}",
        f"shap_explained_samples: {len(explained_idx)}",
        f"spearman_rank_corr: {spearman:.6f}",
        f"top_{topk}_overlap_ratio: {overlap_ratio:.6f}",
        f"top_{topk}_overlap_features: {', '.join(overlap) if overlap else '(none)'}",
        f"target_diagnostics: {shap_diag}",
        "",
        "Top 10 by SHAP normalized importance:",
    ]
    for _, row in df.head(10).iterrows():
        summary_lines.append(
            f"- {row['feature']}: shap={row['shap_importance_norm']:.6f}, "
            f"intrinsic={row['intrinsic_importance_norm']:.6f}, "
            f"ranks=({int(row['shap_rank'])},{int(row['intrinsic_rank'])})"
        )
    out_txt.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Saved figure PNG: {out_png}")
    print(f"Saved figure PDF: {out_pdf}")
    print(f"Saved ranking CSV: {out_csv}")
    print(f"Saved summary TXT: {out_txt}")
    print(f"SHAP target: {shap_target_name}")
    print(f"Target diagnostics: {shap_diag}")
    print(f"Spearman rank corr: {spearman:.4f}")
    print(f"Top-{topk} overlap: {len(overlap)}/{topk} ({overlap_ratio:.1%})")


if __name__ == "__main__":
    main()
