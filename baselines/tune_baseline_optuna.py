#!/usr/bin/env python3
"""
Optuna tuning for baseline DCSM (non-MoE), fully contained in baselines/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import optuna
from lifelines.statistics import multivariate_logrank_test
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("ADACSM_NO_SHOW", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.data_utils import load_data
from utils.general_utils import train_test_DCSM


def _bool_event(y: np.ndarray) -> np.ndarray:
    return np.asarray([bool(item[0]) for item in y], dtype=bool)


def _time(y: np.ndarray) -> np.ndarray:
    return np.asarray([float(item[1]) for item in y], dtype=float)


def _logrank_from_risk(y_test: np.ndarray, risk: np.ndarray) -> float:
    t = _time(y_test)
    e = _bool_event(y_test)
    g = (risk.ravel() >= np.median(risk.ravel())).astype(int)
    out = multivariate_logrank_test(t, g, e)
    return float(out.test_statistic)


def _load_split(ds: str, seed: int):
    data_args = SimpleNamespace(
        dataset=ds,
        is_generate_sim=True,
        is_save_sim=False,
        num_inst=200,
        num_feat=10,
    )
    return load_data(data_args, random_state=seed)


def _configure_torch_device(cuda_device: int) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(cuda_device)
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=str, default="flchain", choices=["support", "PBC", "FRAMINGHAM", "flchain"])
    ap.add_argument("--tune_trials", type=int, default=50)
    ap.add_argument("--tune_epochs", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--progress_every", type=int, default=0)
    ap.add_argument("--cuda_device", type=int, default=0)
    ap.add_argument("--seeds", type=str, default="42,73,666,777,1009")
    ap.add_argument("--select_metric", type=str, default="val_cindex", choices=["val_cindex", "test_cindex", "logrank"])
    ap.add_argument("--out_base", type=str, default=None)
    ap.add_argument("--storage", type=str, default=None)
    ap.add_argument("--study_name", type=str, default=None)
    return ap.parse_args()


def run() -> None:
    args = parse_args()
    seed_list = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seed_list:
        raise ValueError("--seeds must include at least one integer")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = out_dir / "optuna_baseline_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ADACSM_FIGURE_DIR"] = str(figure_dir.resolve())
    _configure_torch_device(args.cuda_device)

    if args.out_base is None:
        args.out_base = str(out_dir / f"optuna_{args.dataset}_baseline")
    if args.storage is None:
        args.storage = f"sqlite:///{out_dir}/optuna_{args.dataset}_baseline.db"
    if args.study_name is None:
        args.study_name = f"dcsm_{args.dataset}_baseline"

    direction = "minimize" if args.select_metric == "logrank" else "maximize"

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        discount = trial.suggest_float("discount", 0.3, 1.0)
        layers_str = trial.suggest_categorical("layers", ["[50]", "[100]", "[50,50]"])
        layers = [int(x) for x in layers_str.strip("[]").split(",") if x.strip()]
        cidx_scores = []
        logrank_scores = []

        for seed in seed_list:
            os.environ["ADACSM_PLOT_TAG"] = f"optuna_{args.dataset}_trial{trial.number}_seed{seed}"
            x_train, x_test, y_train, y_test, _ = _load_split(args.dataset, seed)
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            param = {
                "learning_rate": float(lr),
                "layers": list(layers),
                "k": 2,
                "iters": int(args.tune_epochs),
                "distribution": "Weibull",
                "discount": float(discount),
                "patience": int(args.patience),
                "early_stopping": True,
            }
            _, cidx, pred, _, _, _ = train_test_DCSM(
                param, x_train, x_test, y_train, y_test,
                seed=seed, fix=True, method="DCSM", use_moe=False,
                progress_every=args.progress_every,
            )
            cidx_scores.append(float(cidx))
            logrank_scores.append(_logrank_from_risk(y_test, np.asarray(pred)[:, 0]))

        metrics = {
            "c_index_test_mean": float(np.mean(cidx_scores)),
            "c_index_test_std": float(np.std(cidx_scores)),
            "logrank_mean": float(np.mean(logrank_scores)),
            "logrank_std": float(np.std(logrank_scores)),
            "n_seeds": len(seed_list),
        }
        trial.set_user_attr("metrics", metrics)

        if args.select_metric == "test_cindex":
            return metrics["c_index_test_mean"]
        if args.select_metric == "logrank":
            return metrics["logrank_mean"]
        return metrics["c_index_test_mean"]

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction=direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=args.tune_trials, show_progress_bar=True)

    best = study.best_trial
    output_json = {
        "study_name": args.study_name,
        "dataset": args.dataset,
        "best_trial_number": best.number,
        "best_value": best.value,
        "best_params": best.params,
        "best_metrics": best.user_attrs.get("metrics", {}),
        "select_metric": args.select_metric,
        "n_trials": len(study.trials),
    }
    with open(f"{args.out_base}_best_trial.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
    study.trials_dataframe().to_csv(f"{args.out_base}_all_trials.csv", index=False)
    print(f"Saved best trial to {args.out_base}_best_trial.json")
    print(f"Saved all trials to {args.out_base}_all_trials.csv")


if __name__ == "__main__":
    run()
