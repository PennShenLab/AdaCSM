#!/usr/bin/env python3
"""
Run baseline survival models with one log file per (dataset, model).

Supported baselines:
  - coxph (lifelines)
  - deepcoxph (auton_survival)
  - dsm (auton_survival)
  - dcsm (in-repo non-MoE DCSM)
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Baseline runs should never block on interactive figure windows.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("ADACSM_NO_SHOW", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

from utils.data_utils import load_data
from utils.general_utils import train_test_DCSM

DCSM_HP = {
    "support": (1.33e-4, 0.4092, [50], 100),
    "PBC": (1.60e-4, 0.7662, [50], 100),
    "FRAMINGHAM": (4.01e-4, 0.6950, [50, 50], 100),
    "flchain": (8.07e-3, 0.7760, [100], 100),
}

DEEPSURV = {
    "support": (1.31e-4, [100], 100),
    "PBC": (2.14e-4, [50, 50], 100),
    "FRAMINGHAM": (7.06e-4, [100], 16),
    "flchain": (5.88e-3, [50], 100),
}

DSM_HP = {
    "support": (1.67e-4, [50, 50], 32, 0.8990),
    "PBC": (9.04e-3, [50, 50], 32, 0.4433),
    "FRAMINGHAM": (1.54e-3, [50, 50], 16, 0.4655),
    "flchain": (3.79e-5, [50], 128, 0.6438),
}

COX_BY_DATASET = {
    "support": (0.0096, 0.3594),
    "PBC": (0.0894, 0.6936),
    "FRAMINGHAM": (0.0019, 0.4612),
    "flchain": (0.0097, 0.3861),
}


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


def _auton_available() -> bool:
    try:
        import auton_survival  # noqa: F401
        return True
    except Exception:
        return False


def _configure_torch_device(cuda_device: int) -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(cuda_device)
    except Exception:
        pass


def _load_split(ds: str, seed: int):
    data_args = SimpleNamespace(
        dataset=ds,
        is_generate_sim=True,
        is_save_sim=False,
        num_inst=200,
        num_feat=10,
    )
    return load_data(data_args, random_state=seed)


def _run_coxph_for_dataset(ds: str, args, log_path: Path) -> None:
    pen, l1 = COX_BY_DATASET[ds]
    if args.cox_penalizer is not None:
        pen = args.cox_penalizer
    if args.cox_l1_ratio is not None:
        l1 = args.cox_l1_ratio

    cidx_scores = []
    logrank_scores = []
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# model: coxph\n# dataset: {ds}\n# penalizer: {pen}\n# l1_ratio: {l1}\n")
        lf.write("seed,c_index_test,logrank\n")
        for seed in args.seed_list:
            x_train, x_test, y_train, y_test, _ = _load_split(ds, seed)
            t_train = _time(y_train)
            e_train = _bool_event(y_train).astype(int)
            t_test = _time(y_test)
            e_test_bool = _bool_event(y_test)

            cols = [f"x{i}" for i in range(x_train.shape[1])]
            df_tr = pd.DataFrame(x_train, columns=cols)
            df_tr["time"] = t_train
            df_tr["event"] = e_train
            df_te = pd.DataFrame(x_test, columns=cols)

            cph = CoxPHFitter(penalizer=float(pen), l1_ratio=float(l1))
            cph.fit(df_tr, duration_col="time", event_col="event")
            pred = cph.predict_partial_hazard(df_te[cols]).to_numpy().reshape(-1, 1)
            pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            cidx = float(concordance_index_censored(e_test_bool, t_test, pred[:, 0])[0])
            lr = _logrank_from_risk(y_test, pred[:, 0])
            cidx_scores.append(cidx)
            logrank_scores.append(lr)
            lf.write(f"{seed},{cidx:.6f},{lr:.6f}\n")
        lf.write("\n")
        lf.write(f"c_index_mean,{np.mean(cidx_scores):.6f}\n")
        lf.write(f"c_index_std,{np.std(cidx_scores):.6f}\n")
        lf.write(f"logrank_mean,{np.mean(logrank_scores):.6f}\n")
        lf.write(f"logrank_std,{np.std(logrank_scores):.6f}\n")


def _run_auton_for_dataset(ds: str, args, log_path: Path, model_kind: str) -> None:
    from auton_survival.models.cph import DeepCoxPH
    from auton_survival.models.dsm import DeepSurvivalMachines

    cidx_scores = []
    logrank_scores = []
    if model_kind == "deepcoxph":
        lr, layers, bs = DEEPSURV[ds]
        discount = None
    else:
        lr, layers, bs, discount = DSM_HP[ds]

    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# model: {model_kind}\n# dataset: {ds}\n")
        lf.write(f"# learning_rate: {lr}\n# layers: {layers}\n# batch_size: {bs}\n")
        if discount is not None:
            lf.write(f"# discount: {discount}\n")
        lf.write("seed,c_index_test,logrank\n")
        with contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
            for seed in args.seed_list:
                x_train, x_test, y_train, y_test, _ = _load_split(ds, seed)
                t_train = _time(y_train)
                e_train = _bool_event(y_train).astype(int)
                t_test = _time(y_test)
                e_test_bool = _bool_event(y_test)

                x_tr, x_val, t_tr, t_val, e_tr, e_val = train_test_split(
                    x_train, t_train, e_train,
                    test_size=0.15, random_state=seed, stratify=e_train,
                )
                if model_kind == "deepcoxph":
                    model = DeepCoxPH(layers=layers, random_seed=int(seed))
                else:
                    model = DeepSurvivalMachines(
                        k=2, layers=layers, distribution="Weibull",
                        discount=float(discount), random_seed=int(seed),
                    )
                model.fit(
                    np.asarray(x_tr, dtype=float), np.asarray(t_tr, dtype=float), np.asarray(e_tr, dtype=int),
                    val_data=(np.asarray(x_val, dtype=float), np.asarray(t_val, dtype=float), np.asarray(e_val, dtype=int)),
                    iters=args.iters, learning_rate=float(lr), batch_size=int(bs),
                )
                horizon = float(np.max(t_train))
                pred = model.predict_risk(np.asarray(x_test, dtype=float), horizon)
                pred = np.nan_to_num(np.asarray(pred, dtype=float).reshape(-1, 1), nan=0.0, posinf=0.0, neginf=0.0)
                cidx = float(concordance_index_censored(e_test_bool, t_test, pred[:, 0])[0])
                lr_stat = _logrank_from_risk(y_test, pred[:, 0])
                cidx_scores.append(cidx)
                logrank_scores.append(lr_stat)
                lf.write(f"{seed},{cidx:.6f},{lr_stat:.6f}\n")
        lf.write("\n")
        lf.write(f"c_index_mean,{np.mean(cidx_scores):.6f}\n")
        lf.write(f"c_index_std,{np.std(cidx_scores):.6f}\n")
        lf.write(f"logrank_mean,{np.mean(logrank_scores):.6f}\n")
        lf.write(f"logrank_std,{np.std(logrank_scores):.6f}\n")


def _run_dcsm_for_dataset(ds: str, args, log_path: Path) -> None:
    lr, disc, layers, _ = DCSM_HP[ds]
    param = {
        "learning_rate": float(lr),
        "layers": list(layers),
        "k": 2,
        "iters": int(args.iters),
        "distribution": "Weibull",
        "discount": float(disc),
        "patience": int(args.patience),
        "early_stopping": True,
    }
    cidx_scores = []
    logrank_scores = []
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# model: dcsm\n# dataset: {ds}\n")
        lf.write(f"# learning_rate: {lr}\n# discount: {disc}\n# layers: {layers}\n")
        lf.write("seed,c_index_test,logrank\n")
        with contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
            _configure_torch_device(args.cuda_device)
            for seed in args.seed_list:
                os.environ["ADACSM_PLOT_TAG"] = f"{ds}_dcsm_seed{seed}"
                x_train, x_test, y_train, y_test, _ = _load_split(ds, seed)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                _, cidx, pred, _, _, _ = train_test_DCSM(
                    param, x_train, x_test, y_train, y_test,
                    seed=seed, fix=True, method="DCSM", use_moe=False, progress_every=0,
                )
                lr_stat = _logrank_from_risk(y_test, np.asarray(pred)[:, 0])
                cidx_scores.append(float(cidx))
                logrank_scores.append(float(lr_stat))
                lf.write(f"{seed},{float(cidx):.6f},{float(lr_stat):.6f}\n")
        lf.write("\n")
        lf.write(f"c_index_mean,{np.mean(cidx_scores):.6f}\n")
        lf.write(f"c_index_std,{np.std(cidx_scores):.6f}\n")
        lf.write(f"logrank_mean,{np.mean(logrank_scores):.6f}\n")
        lf.write(f"logrank_std,{np.std(logrank_scores):.6f}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, required=True, help="directory for per-run .log files")
    ap.add_argument("--figure-dir", type=Path, default=None, help="directory for saved figures (default: <out-dir>/figures)")
    ap.add_argument("--cuda-device", type=int, default=0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--seeds", type=str, default="42,73,666,777,1009")
    ap.add_argument("--datasets", type=str, default="support,PBC,FRAMINGHAM,flchain")
    ap.add_argument("--models", type=str, default="coxph,deepcoxph,dsm,dcsm")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cox-penalizer", type=float, default=None)
    ap.add_argument("--cox-l1-ratio", type=float, default=None)
    args = ap.parse_args()

    args.seed_list = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not args.seed_list:
        sys.exit("--seeds must list at least one integer")
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    models = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    valid_ds = {"support", "PBC", "FRAMINGHAM", "flchain"}
    for d in datasets:
        if d not in valid_ds:
            sys.exit(f"unknown dataset {d!r}; expected one of {sorted(valid_ds)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.figure_dir is None:
        args.figure_dir = args.out_dir / "figures"
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ADACSM_FIGURE_DIR"] = str(args.figure_dir.resolve())
    print(f"Figures will be saved under: {args.figure_dir}")
    for ds in datasets:
        for model in models:
            name = f"{ds}_{model}"
            log_path = args.out_dir / f"{name}.log"
            print(f"=== {name} -> {log_path.name} ===")
            if args.dry_run:
                print("  [dry-run]")
                continue
            if model == "coxph":
                _run_coxph_for_dataset(ds, args, log_path)
            elif model in {"deepcoxph", "dsm"}:
                if not _auton_available():
                    print("  skipped (auton_survival not installed)")
                    continue
                _run_auton_for_dataset(ds, args, log_path, model)
            elif model == "dcsm":
                _run_dcsm_for_dataset(ds, args, log_path)
            else:
                print(f"  skipped unknown model: {model}")
                continue
            print(f"  wrote {log_path}")

    if not args.dry_run:
        print(f"\nAggregate metrics:\n  python scripts/parse_ranking_exp_logs.py {args.out_dir}")


if __name__ == "__main__":
    main()
