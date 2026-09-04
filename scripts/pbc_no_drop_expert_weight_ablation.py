#!/usr/bin/env python3
"""PBC expert-weight ablation with no row dropping and 60/10/30 split.

This runner:
- loads PBC via ``load_pbc_dataset`` (current data_utils behavior: no row dropping),
- enforces stratified 60/10/30 train/val/test split per seed,
- trains AdaCSM once per (seed, architecture),
- evaluates learned/equal/random expert weights from the same checkpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import load_pbc_dataset
from utils.general_utils import combine_t_e, logrank_multivariate_from_stratified_y, train_test_AdaCSM


SEEDS = [42, 73, 666, 777, 1009]
MODES = ["learned", "equal", "random"]


@dataclass
class ArchConfig:
    name: str
    num_experts: int
    top_k: int
    param: Dict[str, float]
    moe: Dict[str, float]


ARCHS = [
    ArchConfig(
        name="AdaCSM_pbc_table_hp",
        num_experts=32,
        top_k=32,
        param={
            "learning_rate": 4.68e-4,
            "layers": [100],
            "k": 2,
            "iters": 2000,
            "distribution": "Weibull",
            "discount": 0.8316,
            "patience": 200,
            "early_stopping": True,
            "batch_size": 16,
            "ranking_loss_lambda": 0.0,
            "calibration_loss_lambda": 0.0,
            "composite_beta": 0.5,
            "metrics_extra_every": 0,
            "report_val_mean_ipcw_td_cindex": False,
        },
        moe={
            "moe_dropout": 0.0477,
            "gate_dropout": 0.2708,
            "gate_temperature": 0.1191,
            "routing_noise_std": 0.0,
            "weight_decay": 0.0,
            "load_balance_lambda": 0.0952,
        },
    ),
    ArchConfig(
        name="AdaCSM_single_expert",
        num_experts=1,
        top_k=1,
        param={
            "learning_rate": 4.68e-4,
            "layers": [100],
            "k": 2,
            "iters": 2000,
            "distribution": "Weibull",
            "discount": 0.8316,
            "patience": 200,
            "early_stopping": True,
            "batch_size": 16,
            "ranking_loss_lambda": 0.0,
            "calibration_loss_lambda": 0.0,
            "composite_beta": 0.5,
            "metrics_extra_every": 0,
            "report_val_mean_ipcw_td_cindex": False,
        },
        moe={
            "moe_dropout": 0.0477,
            "gate_dropout": 0.2708,
            "gate_temperature": 0.1191,
            "routing_noise_std": 0.0,
            "weight_decay": 0.0,
            "load_balance_lambda": 0.0952,
        },
    ),
]


def split_60_10_30(event: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(event))
    rs2 = (int(seed) + 7919) % (2**31 - 1)
    idx_train, idx_temp = train_test_split(
        idx, test_size=0.4, random_state=seed, stratify=event,
    )
    e_temp = event[idx_temp]
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.75, random_state=rs2, stratify=e_temp,
    )
    return idx_train, idx_val, idx_test


def make_y(t: np.ndarray, e: np.ndarray):
    return combine_t_e(np.asarray(t, dtype=float), np.asarray(e, dtype=int))


def evaluate_mode(model, x_test: np.ndarray, y_test, mode: str):
    model.set_expert_weight_mode(mode)
    t_test = np.asarray([row[1] for row in y_test], dtype=float)
    e_test = np.asarray([row[0] for row in y_test], dtype=int)
    horizon = float(np.percentile(t_test, 90))
    risks = model.predict_risk(np.asarray(x_test, dtype=float), horizon)
    risks = np.asarray(risks).reshape(-1)
    risks = np.nan_to_num(risks, nan=0.0, posinf=0.0, neginf=0.0)
    cidx = float(concordance_index_censored(e_test.astype(bool), t_test, risks)[0])

    tags, _, _ = model.predict_phenotype(np.asarray(x_test, dtype=np.float64))
    y_groups = []
    for k in range(model.k):
        sel = np.where(tags == k)[0]
        y_groups.append(y_test[sel].tolist())
    lr_stat, _ = logrank_multivariate_from_stratified_y(y_groups)
    return cidx, float(lr_stat)


def summarize(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1))}


def _run_seed_for_arch(seed: int, arch: ArchConfig):
    x_all, t_all, e_all, _ = load_pbc_dataset()
    t_all = np.asarray(t_all, dtype=float)
    e_all = np.asarray(e_all, dtype=int)

    idx_tr, idx_va, idx_te = split_60_10_30(e_all, seed)
    x_tr, x_va, x_te = x_all[idx_tr], x_all[idx_va], x_all[idx_te]
    y_tr = make_y(t_all[idx_tr], e_all[idx_tr])
    y_va = make_y(t_all[idx_va], e_all[idx_va])
    y_te = make_y(t_all[idx_te], e_all[idx_te])

    model, _, _, _, _, _ = train_test_AdaCSM(
        arch.param,
        x_tr,
        x_te,
        y_tr,
        y_te,
        seed=seed,
        fix=True,
        num_experts=arch.num_experts,
        top_k=arch.top_k,
        moe_dropout=float(arch.moe["moe_dropout"]),
        gate_dropout=float(arch.moe["gate_dropout"]),
        gate_temperature=float(arch.moe["gate_temperature"]),
        routing_noise_std=float(arch.moe["routing_noise_std"]),
        weight_decay=float(arch.moe["weight_decay"]),
        load_balance_lambda=float(arch.moe["load_balance_lambda"]),
        expert_weight_mode="learned",
        X_val=x_va,
        y_val=y_va,
    )

    out = []
    for mode in MODES:
        cidx, lr = evaluate_mode(model, x_te, y_te, mode)
        out.append(
            {
                "arch": arch.name,
                "seed": seed,
                "mode": mode,
                "c_index": cidx,
                "logrank": lr,
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results/pbc_no_drop_expert_weight_ablation_tablehp_20260505.json")
    parser.add_argument(
        "--arch_names",
        type=str,
        default="all",
        help="Comma-separated arch names to run, or 'all'.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=len(SEEDS),
        help="Parallel workers for per-seed runs.",
    )
    args = parser.parse_args()

    selected_archs = ARCHS
    if args.arch_names.strip().lower() != "all":
        keep = {x.strip() for x in args.arch_names.split(",") if x.strip()}
        selected_archs = [a for a in ARCHS if a.name in keep]
        if not selected_archs:
            raise ValueError(f"No matching arch names in {sorted(keep)}")

    rows = []
    agg = defaultdict(lambda: defaultdict(lambda: {"c_index": [], "logrank": []}))

    for arch in selected_archs:
        max_workers = max(1, int(args.max_workers))
        if max_workers == 1:
            for seed in SEEDS:
                seed_rows = _run_seed_for_arch(seed, arch)
                for row in seed_rows:
                    rows.append(row)
                    agg[row["arch"]][row["mode"]]["c_index"].append(float(row["c_index"]))
                    agg[row["arch"]][row["mode"]]["logrank"].append(float(row["logrank"]))
                    print(
                        f"{row['arch']} seed={row['seed']} mode={row['mode']} "
                        f"c_index={row['c_index']:.4f} logrank={row['logrank']:.4f}",
                        flush=True,
                    )
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_run_seed_for_arch, seed, arch): seed for seed in SEEDS}
                for fut in as_completed(futs):
                    seed_rows = fut.result()
                    for row in seed_rows:
                        rows.append(row)
                        agg[row["arch"]][row["mode"]]["c_index"].append(float(row["c_index"]))
                        agg[row["arch"]][row["mode"]]["logrank"].append(float(row["logrank"]))
                        print(
                            f"{row['arch']} seed={row['seed']} mode={row['mode']} "
                            f"c_index={row['c_index']:.4f} logrank={row['logrank']:.4f}",
                            flush=True,
                        )

    summary = {}
    for arch_name, mode_map in agg.items():
        summary[arch_name] = {}
        for mode, vals in mode_map.items():
            summary[arch_name][mode] = {
                "c_index": summarize(vals["c_index"]),
                "logrank": summarize(vals["logrank"]),
            }

    out = {"seeds": SEEDS, "rows": rows, "summary": summary}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== Summary (mean ± std over 5 seeds) ===")
    for arch in selected_archs:
        print(f"\n[{arch.name}]")
        for mode in MODES:
            s = summary[arch.name][mode]
            print(
                f"  {mode:7s} | "
                f"C-index {s['c_index']['mean']:.4f} ± {s['c_index']['std']:.4f} | "
                f"Logrank {s['logrank']['mean']:.2f} ± {s['logrank']['std']:.2f}",
            )


if __name__ == "__main__":
    main()

