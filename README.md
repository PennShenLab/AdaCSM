# AdaCSM

AdaCSM is a Mixture-of-Experts (MoE) survival modeling framework that combines representation learning, adaptive expert routing, and time-to-event prediction for heterogeneous clinical populations.

This repository provides the official AdaCSM codebase and reproducibility scripts for the ACM publication.

## 📄 Paper

- ACM publication: [https://dl.acm.org/doi/full/10.1145/3807503.3819574](https://dl.acm.org/doi/full/10.1145/3807503.3819574)

## 🎯 Motivation

Clinical survival cohorts are often heterogeneous: different subgroups follow different progression patterns and risk dynamics. A single global survival function can miss this structure.

AdaCSM addresses this by learning:

- A shared feature representation for survival prediction
- Multiple expert survival components that capture subgroup-specific behavior
- An adaptive MoE routing mechanism that selects relevant experts per patient

This design improves both predictive flexibility and interpretability of subgroup behavior.

## 🧠 Method At A Glance

AdaCSM models survival outcomes with an MoE architecture:

- **Encoder/Backbone** transforms patient covariates into latent representations
- **Expert survival heads** model distinct survival patterns
- **Gating network** produces instance-specific expert weights (or top-k routing)
- **Aggregated survival prediction** combines expert outputs into final risk/survival estimates

In practice, this lets AdaCSM capture non-uniform risk structure across populations while preserving a transparent expert-assignment view for analysis.
These expert-assignment patterns can also be used for subtype-style clustering and patient stratification.

## 🔢 Model Outputs

AdaCSM provides two primary outputs:

1. **Time-to-event prediction**: individualized survival risk/survival-time estimates.
2. **Subtype clustering**: expert-assignment-based patient subgrouping for stratification and interpretation.

## 🏗️ AdaCSM Architecture

<p align="center">
  <img src="docs/AdaCSM_schema.png" alt="AdaCSM Schema" width="500" />
</p>

## 📦 Repository Scope

- Training and evaluation code: `main.py`, `src/models/`, `baselines/models/`, `utils/`
- Hyperparameter search: `src/tune_adacsm_optuna_core.py`
- Reproducibility scripts: `scripts/` (including `scripts/plot_km.py`, `scripts/plot_pareto_frontier.py`, `scripts/visualize_moe_gates.py`)
- Baseline lane scripts: `baselines/run_baseline_models.sh`, `baselines/run_baseline_optuna.sh`
- AdaCSM lane scripts: `src/run_adacsm_model.sh`, `src/run_dense_experiments.sh`, `src/run_topk_experiments.sh`, `src/run_adacsm_optuna_tuning.sh`

## 🗂️ Project Layout

- `src/`: AdaCSM experiment/tuning entrypoints (`src/run_adacsm_model.sh`, `src/run_dense_experiments.sh`, `src/run_topk_experiments.sh`, `src/run_adacsm_optuna_tuning.sh`, `src/tune_adacsm_optuna_core.py`)
- `src/models/`: AdaCSM model implementations (`src/models/adacsm_api.py`, `src/models/adacsm_torch.py`)
- `baselines/`: baseline runners (`baselines/run_baselines.py`, `baselines/run_baseline_*.sh`)
- `baselines/models/`: DCSM baseline model implementations (`baselines/models/dcsm_api.py`, `baselines/models/dcsm_torch.py`)
- `main.py`: AdaCSM core training runner (MoE-enabled)

## 🔐 Data Availability

- Included open datasets:
  - `datasets/support2.csv`
  - `datasets/flchain.csv`
  - `datasets/pbc2.csv`
  - `datasets/framingham.csv`
- The repository includes the data files used in the released experiments.

## ⚙️ Environment Setup

```bash
conda create -n adacsm python=3.10 -y
conda activate adacsm
pip install -r requirements.txt
```

## 🚀 Quick Start

Single AdaCSM run:

```bash
bash src/run_adacsm_model.sh --dataset FRAMINGHAM --num_experts 32 --top_k 2
```

Supported datasets in this release include `support`, `flchain`, `PBC`, and `FRAMINGHAM`.

## 🧪 Reproducibility Workflows

Baseline lane:

```bash
bash baselines/run_baseline_models.sh
bash baselines/run_baseline_optuna.sh
```

You can choose a subset explicitly:

```bash
conda run -n adacsm python baselines/run_baselines.py \
  --out-dir logs/paper_baselines \
  --models coxph,deepcoxph,dsm,dcsm
```

Notes:
- Baseline runs are non-interactive and save training/KM figures under `<out-dir>/figures` by default.
- Quick overrides without editing scripts:
  - `OUT_DIR=logs/my_baselines bash baselines/run_baseline_models.sh --models coxph,dcsm`
  - `DATASET=PBC TUNE_TRIALS=20 bash baselines/run_baseline_optuna.sh`

AdaCSM lane:

```bash
bash src/run_dense_experiments.sh
bash src/run_topk_experiments.sh
bash src/run_adacsm_optuna_tuning.sh
```

Quick overrides without editing scripts:
- `DATASET=PBC bash src/run_dense_experiments.sh`
- `DATASET=FRAMINGHAM bash src/run_topk_experiments.sh`
- `DATASET=support TUNE_TRIALS=20 bash src/run_adacsm_optuna_tuning.sh`

## 📚 Citation

Please cite the ACM paper listed above when using this repository.

```bibtex
@inproceedings{zhuang2026expert,
  title={Expert-Driven Survival Machines: Improving Stratification and Interpretability in Multiple Clinical Cohorts},
  author={Zhuang, Farica and Wen, Zixuan and Davatzikos, Christos and Shen, Li},
  booktitle={Proceedings of the 17th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics},
  pages={1--10},
  year={2026}
}
```
