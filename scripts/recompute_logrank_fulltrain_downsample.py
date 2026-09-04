#!/usr/bin/env python3
"""
Recompute logrank with "train on full cohort, evaluate on 95% subsamples".

Protocol:
1) Build a full cohort feature matrix for each dataset by concatenating the
   train/val/test splits from `load_data(..., random_state=42)`.
2) Train one model on the full cohort.
3) For each seed in {42, 73, 666, 777, 1009}, sample 95% of subjects
   (without replacement), score risk on that subset, and compute logrank
   (median risk split).
4) Report mean ± std over seeds.

This script is intentionally independent from the standard train/val/test flow
in `main.py`, per request.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
from sksurv.metrics import concordance_index_censored

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import load_data
from utils.general_utils import (
    _logrank_median_risk_split,
    train_test_AdaCSM,
    train_test_CoxPH,
    train_test_DCSM,
    train_test_DPCH,
    train_test_DSM,
)


SEEDS = [42, 73, 666, 777, 1009]


@dataclass
class Experiment:
    row_name: str
    dataset: str
    model: str
    param: Dict
    moe: Dict | None = None


def build_experiments() -> List[Experiment]:
    # Keep this aligned with the table rows currently used in the project.
    exps: List[Experiment] = []

    dcsm_by_dataset = {
        # order provided by user: flchain, support, PBC, FRAMINGHAM
        "flchain": {"learning_rate": 2.12e-5, "discount": 0.9930, "layers": [50, 50]},
        "support": {"learning_rate": 0.00038746282668289796, "discount": 0.8780685567761914, "layers": [100]},
        "PBC": {"learning_rate": 4.01e-4, "discount": 0.6950, "layers": [50, 50]},
        "FRAMINGHAM": {"learning_rate": 8.07e-3, "discount": 0.7760, "layers": [100]},
    }

    deepsurv_by_dataset = {
        # user-provided settings: SUPPORT, PBC, FRAMINGHAM, FLCHAIN
        "support": {"learning_rate": 1.31e-4, "layers": [100], "batch_size": 100},
        "PBC": {"learning_rate": 2.14e-4, "layers": [50, 50], "batch_size": 100},
        "FRAMINGHAM": {"learning_rate": 7.06e-4, "layers": [100], "batch_size": 16},
        "flchain": {"learning_rate": 5.88e-3, "layers": [50], "batch_size": 100},
    }
    dsm_by_dataset = {
        # Match dsm_*_td_metrics.log per-dataset hyperparameters.
        "flchain": {"learning_rate": 3.79e-05, "layers": [50], "discount": 0.6438, "batch_size": 128},
        "support": {"learning_rate": 0.000167, "layers": [50, 50], "discount": 0.899, "batch_size": 32},
        "PBC": {"learning_rate": 0.000362, "layers": [50, 50], "discount": 0.9334, "batch_size": 32},
        "FRAMINGHAM": {"learning_rate": 0.00154, "layers": [50, 50], "discount": 0.4655, "batch_size": 16},
    }

    for ds in ["flchain", "support", "PBC", "FRAMINGHAM"]:
        exps.append(
            Experiment(
                row_name="Cox PH",
                dataset=ds,
                model="coxph",
                param={"penalizer": 1e-4, "l1_ratio": 0.0},
            )
        )
        exps.append(
            Experiment(
                row_name="DeepSurv",
                dataset=ds,
                model="deepcoxph",
                param={
                    "learning_rate": deepsurv_by_dataset[ds]["learning_rate"],
                    "layers": deepsurv_by_dataset[ds]["layers"],
                    "iters": 2000,
                    "l1_penalty": 0.0,
                    "patience": 100,
                    "early_stopping": True,
                    "batch_size": deepsurv_by_dataset[ds]["batch_size"],
                },
            )
        )
        exps.append(
            Experiment(
                row_name="DSM",
                dataset=ds,
                model="dsm",
                param={
                    "learning_rate": dsm_by_dataset[ds]["learning_rate"],
                    "layers": dsm_by_dataset[ds]["layers"],
                    "k": 2,
                    "iters": 2000,
                    "distribution": "Weibull",
                    "discount": dsm_by_dataset[ds]["discount"],
                    "patience": 100,
                    "early_stopping": True,
                    "batch_size": dsm_by_dataset[ds]["batch_size"],
                },
            )
        )
        exps.append(
            Experiment(
                row_name="DCSM",
                dataset=ds,
                model="dcsm",
                param={
                    "learning_rate": dcsm_by_dataset[ds]["learning_rate"],
                    "layers": dcsm_by_dataset[ds]["layers"],
                    "k": 2,
                    "iters": 2000,
                    "distribution": "Weibull",
                    "discount": dcsm_by_dataset[ds]["discount"],
                    "patience": 100,
                    "early_stopping": True,
                    "batch_size": 100,
                    "ranking_loss_lambda": 0.0,
                    "calibration_loss_lambda": 0.0,
                    "composite_beta": 0.5,
                    "metrics_extra_every": 0,
                    "report_val_mean_ipcw_td_cindex": False,
                },
            )
        )

    # AdaCSM dense (dataset-specific settings currently used in your table)
    exps += [
        Experiment(
            row_name="AdaCSM",
            dataset="flchain",
            model="adacsm",
            param={
                "learning_rate": 0.0001909,
                "layers": [50],
                "k": 2,
                "iters": 2000,
                "distribution": "Weibull",
                "discount": 0.7104,
                "patience": 100,
                "early_stopping": True,
                "batch_size": 100,
                "ranking_loss_lambda": 0.0,
                "calibration_loss_lambda": 0.0,
                "composite_beta": 0.5,
                "metrics_extra_every": 0,
                "report_val_mean_ipcw_td_cindex": False,
            },
            moe={
                "num_experts": 4,
                "top_k": 4,
                "moe_dropout": 0.0,
                "gate_dropout": 0.0,
                "gate_temperature": 1.0,
                "routing_noise_std": 0.0,
                "weight_decay": 0.0,
                "load_balance_lambda": 0.0,
            },
        ),
        Experiment(
            row_name="AdaCSM",
            dataset="support",
            model="adacsm",
            param={
                "learning_rate": 0.00477113094506427,
                "layers": [50],
                "k": 2,
                "iters": 2000,
                "distribution": "Weibull",
                "discount": 0.31682116379639425,
                "patience": 200,
                "early_stopping": True,
                "batch_size": 100,
                "ranking_loss_lambda": 0.0,
                "calibration_loss_lambda": 0.0,
                "composite_beta": 0.5,
                "metrics_extra_every": 0,
                "report_val_mean_ipcw_td_cindex": False,
            },
            moe={
                "num_experts": 8,
                "top_k": 8,
                "moe_dropout": 0.30638922522356743,
                "gate_dropout": 0.15697705701560705,
                "gate_temperature": 2.5768531752323436,
                "routing_noise_std": 0.0,
                "weight_decay": 0.0,
                "load_balance_lambda": 0.04668643329128504,
            },
        ),
        Experiment(
            row_name="AdaCSM",
            dataset="PBC",
            model="adacsm",
            param={
                "learning_rate": 7.742723038451758e-05,
                "layers": [100],
                "k": 2,
                "iters": 2000,
                "distribution": "Weibull",
                "discount": 0.9959954334497532,
                "patience": 100,
                "early_stopping": True,
                "batch_size": 100,
                "ranking_loss_lambda": 0.0,
                "calibration_loss_lambda": 0.0,
                "composite_beta": 0.5,
                "metrics_extra_every": 0,
                "report_val_mean_ipcw_td_cindex": False,
            },
            moe={
                "num_experts": 8,
                "top_k": 8,
                "moe_dropout": 0.2408151213430406,
                "gate_dropout": 0.12267624436904544,
                "gate_temperature": 1.9900485718306669,
                "routing_noise_std": 0.0,
                "weight_decay": 0.0,
                "load_balance_lambda": 0.08364655211504328,
            },
        ),
        Experiment(
            row_name="AdaCSM",
            dataset="FRAMINGHAM",
            model="adacsm",
            param={
                "learning_rate": 0.0005485761541080329,
                "layers": [100],
                "k": 2,
                "iters": 2000,
                "distribution": "Weibull",
                "discount": 0.34802786573933875,
                "patience": 200,
                "early_stopping": True,
                "batch_size": 100,
                "ranking_loss_lambda": 0.0,
                "calibration_loss_lambda": 0.0,
                "composite_beta": 0.5,
                "metrics_extra_every": 0,
                "report_val_mean_ipcw_td_cindex": False,
            },
            moe={
                "num_experts": 16,
                "top_k": 16,
                "moe_dropout": 0.1637838975948782,
                "gate_dropout": 0.01387447881670858,
                "gate_temperature": 3.7526445047224355,
                "routing_noise_std": 0.0,
                "weight_decay": 0.0,
                "load_balance_lambda": 0.06060964549941431,
            },
        ),
    ]

    # AdaCSM sparse top-2
    exps += [
        Experiment(
            row_name="AdaCSM (Sparse top-2)",
            dataset="flchain",
            model="adacsm",
            param={
                "learning_rate": 0.0001909,
                "layers": [50],
                "k": 2,
                "iters": 2000,
                "distribution": "Weibull",
                "discount": 0.7104,
                "patience": 100,
                "early_stopping": True,
                "batch_size": 100,
                "ranking_loss_lambda": 0.0,
                "calibration_loss_lambda": 0.0,
                "composite_beta": 0.5,
                "metrics_extra_every": 0,
                "report_val_mean_ipcw_td_cindex": False,
            },
            moe={
                "num_experts": 4,
                "top_k": 2,
                "moe_dropout": 0.0,
                "gate_dropout": 0.0,
                "gate_temperature": 1.0,
                "routing_noise_std": 0.0,
                "weight_decay": 0.0,
                "load_balance_lambda": 0.0,
            },
        ),
        Experiment(
            row_name="AdaCSM (Sparse top-2)",
            dataset="support",
            model="adacsm",
            param={
                "learning_rate": 0.00477113094506427,
                "layers": [50],
                "k": 2,
                "iters": 2000,
                "distribution": "Weibull",
                "discount": 0.31682116379639425,
                "patience": 200,
                "early_stopping": True,
                "batch_size": 100,
                "ranking_loss_lambda": 0.0,
                "calibration_loss_lambda": 0.0,
                "composite_beta": 0.5,
                "metrics_extra_every": 0,
                "report_val_mean_ipcw_td_cindex": False,
            },
            moe={
                "num_experts": 8,
                "top_k": 2,
                "moe_dropout": 0.30638922522356743,
                "gate_dropout": 0.15697705701560705,
                "gate_temperature": 2.5768531752323436,
                "routing_noise_std": 0.0,
                "weight_decay": 0.0,
                "load_balance_lambda": 0.04668643329128504,
            },
        ),
        Experiment(
            row_name="AdaCSM (Sparse top-2)",
            dataset="PBC",
            model="adacsm",
            param={
                "learning_rate": 0.0005485761541080329,
                "layers": [100],
                "k": 2,
                "iters": 2000,
                "distribution": "Weibull",
                "discount": 0.34802786573933875,
                "patience": 200,
                "early_stopping": True,
                "batch_size": 100,
                "ranking_loss_lambda": 0.0,
                "calibration_loss_lambda": 0.0,
                "composite_beta": 0.5,
                "metrics_extra_every": 0,
                "report_val_mean_ipcw_td_cindex": False,
            },
            moe={
                "num_experts": 16,
                "top_k": 2,
                "moe_dropout": 0.1637838975948782,
                "gate_dropout": 0.01387447881670858,
                "gate_temperature": 3.7526445047224355,
                "routing_noise_std": 0.0,
                "weight_decay": 0.0,
                "load_balance_lambda": 0.06060964549941431,
            },
        ),
        Experiment(
            row_name="AdaCSM (Sparse top-2)",
            dataset="FRAMINGHAM",
            model="adacsm",
            param={
                "learning_rate": 0.0005485761541080329,
                "layers": [100],
                "k": 2,
                "iters": 2000,
                "distribution": "Weibull",
                "discount": 0.34802786573933875,
                "patience": 200,
                "early_stopping": True,
                "batch_size": 100,
                "ranking_loss_lambda": 0.0,
                "calibration_loss_lambda": 0.0,
                "composite_beta": 0.5,
                "metrics_extra_every": 0,
                "report_val_mean_ipcw_td_cindex": False,
            },
            moe={
                "num_experts": 16,
                "top_k": 2,
                "moe_dropout": 0.1637838975948782,
                "gate_dropout": 0.01387447881670858,
                "gate_temperature": 3.7526445047224355,
                "routing_noise_std": 0.0,
                "weight_decay": 0.0,
                "load_balance_lambda": 0.06060964549941431,
            },
        ),
    ]
    return exps


def _make_args_for_load(dataset: str) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=dataset,
        is_generate_sim=False,
        num_inst=200,
        num_feat=10,
        is_save_sim=False,
    )


def build_full_cohort(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    args = _make_args_for_load(dataset)
    X_tr, X_va, X_te, y_tr, y_va, y_te, _ = load_data(args, random_state=42)
    X_full = np.vstack([X_tr, X_va, X_te]).astype(float)
    y_full = np.concatenate([y_tr, y_va, y_te], axis=0)
    return X_full, y_full


def fit_model(exp: Experiment, X_full: np.ndarray, y_full: np.ndarray):
    n = X_full.shape[0]
    val_n = max(16, int(0.1 * n))
    rng = np.random.default_rng(20260416)
    perm = rng.permutation(n)
    val_idx = perm[:val_n]
    X_val, y_val = X_full[val_idx], y_full[val_idx]

    if exp.model == "coxph":
        model, *_ = train_test_CoxPH(
            exp.param, X_full, X_full, y_full, y_full, seed=42, X_val=X_val, y_val=y_val
        )
        return model
    if exp.model == "deepcoxph":
        model, *_ = train_test_DPCH(
            exp.param, X_full, X_full, y_full, y_full, seed=42, X_val=X_val, y_val=y_val
        )
        return model
    if exp.model == "dsm":
        model, *_ = train_test_DSM(
            exp.param, X_full, X_full, y_full, y_full, seed=42, X_val=X_val, y_val=y_val
        )
        return model
    if exp.model == "dcsm":
        model, *_ = train_test_DCSM(
            exp.param, X_full, X_full, y_full, y_full, seed=42, X_val=X_val, y_val=y_val
        )
        return model
    if exp.model == "adacsm":
        m = exp.moe or {}
        model, *_ = train_test_AdaCSM(
            exp.param,
            X_full,
            X_full,
            y_full,
            y_full,
            seed=42,
            num_experts=int(m.get("num_experts", 4)),
            top_k=m.get("top_k", None),
            moe_dropout=float(m.get("moe_dropout", 0.0)),
            gate_dropout=float(m.get("gate_dropout", 0.0)),
            gate_temperature=float(m.get("gate_temperature", 1.0)),
            routing_noise_std=float(m.get("routing_noise_std", 0.0)),
            weight_decay=float(m.get("weight_decay", 0.0)),
            load_balance_lambda=float(m.get("load_balance_lambda", 0.0)),
            X_val=X_val,
            y_val=y_val,
        )
        return model
    raise ValueError(f"Unsupported model {exp.model}")


def risk_scores(model, exp: Experiment, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    t = np.asarray([item[1] for item in y], dtype=float)
    if exp.model == "coxph":
        d = X.shape[1]
        import pandas as pd

        feat_cols = [f"x{i}" for i in range(d)]
        df = pd.DataFrame(X, columns=feat_cols)
        r = model.predict_partial_hazard(df[feat_cols]).to_numpy().reshape(-1)
        return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    horizon = float(np.percentile(t, 90))
    r = model.predict_risk(np.asarray(X, dtype=float), horizon)
    r = np.asarray(r).reshape(-1)
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def _apply_adacsm_expert_weight_mode(model, exp: Experiment):
    mode = "learned"
    if exp.model == "adacsm":
        mode = str((exp.moe or {}).get("expert_weight_mode", "learned")).lower().strip()
        if hasattr(model, "set_expert_weight_mode"):
            model.set_expert_weight_mode(mode)
    return mode


def evaluate_metrics_95(model, exp: Experiment, X_full: np.ndarray, y_full: np.ndarray) -> Tuple[List[float], List[float]]:
    n = X_full.shape[0]
    k = max(2, int(np.floor(0.95 * n)))
    stats = []
    cidxs = []
    _apply_adacsm_expert_weight_mode(model, exp)
    for s in SEEDS:
        rng = np.random.default_rng(s)
        idx = rng.choice(n, size=k, replace=False)
        X_sub = X_full[idx]
        y_sub = y_full[idx]
        times = np.asarray([item[1] for item in y_sub], dtype=float)
        events = np.asarray([item[0] for item in y_sub], dtype=int)
        risks = risk_scores(model, exp, X_sub, y_sub)
        lr_stat, _ = _logrank_median_risk_split(times, events, risks)
        stats.append(float(lr_stat))
        cidx = concordance_index_censored(events.astype(bool), times, risks)[0]
        cidxs.append(float(cidx))
    return stats, cidxs


def fmt_mean_std(x: List[float]) -> str:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return "nan ± nan"
    return f"{arr.mean():.2f} ± {arr.std(ddof=1):.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=str, default="all", help="comma-separated row names to run, or all")
    parser.add_argument("--datasets", type=str, default="all", help="comma-separated datasets to run, or all")
    parser.add_argument("--override_num_experts", type=int, default=None,
                        help="Override MoE num_experts for AdaCSM rows")
    parser.add_argument("--override_top_k", type=int, default=None,
                        help="Override MoE top_k for AdaCSM rows")
    parser.add_argument(
        "--adacsm_expert_weight_mode",
        type=str,
        default="learned",
        choices=("learned", "equal", "random"),
        help="AdaCSM-only expert-weight mode for ablation: learned/equal/random.",
    )
    args = parser.parse_args()

    exps = build_experiments()
    if args.rows != "all":
        keep_rows = {x.strip() for x in args.rows.split(",") if x.strip()}
        exps = [e for e in exps if e.row_name in keep_rows]
    if args.datasets != "all":
        keep_ds = {x.strip() for x in args.datasets.split(",") if x.strip()}
        exps = [e for e in exps if e.dataset in keep_ds]

    if args.override_num_experts is not None or args.override_top_k is not None:
        for e in exps:
            if e.model == "adacsm" and e.moe is not None:
                if args.override_num_experts is not None:
                    e.moe["num_experts"] = int(args.override_num_experts)
                if args.override_top_k is not None:
                    e.moe["top_k"] = int(args.override_top_k)

    for e in exps:
        if e.model == "adacsm":
            if e.moe is None:
                e.moe = {}
            e.moe["expert_weight_mode"] = args.adacsm_expert_weight_mode

    results_logrank = {}
    results_cindex = {}
    for exp in exps:
        key = (exp.row_name, exp.dataset)
        print(f"\n=== Running {exp.row_name} | {exp.dataset} ===")
        if exp.model == "adacsm":
            print(f"  AdaCSM expert_weight_mode: {(exp.moe or {}).get('expert_weight_mode', 'learned')}")
        X_full, y_full = build_full_cohort(exp.dataset)
        model = fit_model(exp, X_full, y_full)
        lr_stats, cidx_stats = evaluate_metrics_95(model, exp, X_full, y_full)
        results_logrank[key] = lr_stats
        results_cindex[key] = cidx_stats
        print(f"logrank_95 ({exp.row_name}, {exp.dataset}): {fmt_mean_std(lr_stats)}")
        print(f"cindex_95 ({exp.row_name}, {exp.dataset}): {fmt_mean_std(cidx_stats)}")

    print("\n=== Summary (logrank mean ± std over 95% subsamples) ===")
    row_order = [
        "Cox PH",
        "DeepSurv",
        "DSM",
        "DCSM",
        "AdaCSM",
        "AdaCSM (Sparse top-2)",
    ]
    ds_order = ["flchain", "support", "PBC", "FRAMINGHAM"]
    for row in row_order:
        vals = []
        for ds in ds_order:
            vals.append(fmt_mean_std(results_logrank.get((row, ds), [])))
        print(f"{row}: {vals}")

    print("\n=== Summary (c-index mean ± std over 95% subsamples) ===")
    for row in row_order:
        vals = []
        for ds in ds_order:
            vals.append(fmt_mean_std(results_cindex.get((row, ds), [])))
        print(f"{row}: {vals}")


if __name__ == "__main__":
    main()
