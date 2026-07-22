# Churn Benchmark

Leakage-safe benchmarking pipeline for bank customer churn prediction: 5 classical ML baselines (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost) and 3 deep tabular models (MLP, TabNet, TabR), evaluated with repeated stratified cross-validation.

## Requirements

- Python 3.10–3.13
- Windows / macOS / Linux

## 1. Setup

Clone or download this repo, then from the project root:

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> Note: `torch` and `pytorch-tabnet` are required only for the deep models (MLP/TabNet/TabR). If you don't need them, skip installing those two lines from `requirements.txt` and run with `--skip-deep` (see below).

## 2. Data

Make sure `data/churn.csv` exists (already included in this repo). The pipeline reads it via the path set in `config.py`.

## 3. Run the full pipeline

From the project root:

```bash
python main.py
```

This runs, in order:
1. Leakage audit on the raw dataset
2. Hyperparameter tuning + repeated CV for the 5 classical baselines
3. Repeated CV for the 3 deep models (MLP, TabNet, TabR)
4. Statistical significance tests (Friedman + pairwise Wilcoxon)
5. Imbalance-handling ablation (none / class weighting / SMOTE)
6. SHAP feature-importance analysis for every model

All tables are written to `outputs/tables/*.csv` and all figures to `outputs/figures/*.png`.

### Useful flags

```bash
# Quick smoke test (1 seed, small hyperparameter search, fewer epochs)
# — use this first to confirm everything runs before the full multi-hour run
python main.py --quick

# Skip the deep models (useful if torch / pytorch-tabnet aren't installed)
python main.py --skip-deep
```

Full run time depends on hardware; expect it to take from tens of minutes to a few hours on a laptop CPU due to hyperparameter search and repeated CV.

## 4. Leakage demonstration (optional, standalone)

To reproduce the before/after data-leakage comparison on its own (without running the whole pipeline):

```bash
python leakage_demo.py
```

This writes `outputs/tables/leakage_before_after.csv`.

## Project structure

```
churn-benchmark/
├── main.py                # Full end-to-end pipeline (entry point)
├── leakage_demo.py         # Standalone leakage before/after demo
├── config.py               # Central configuration (paths, seeds, hyperparameters)
├── requirements.txt
├── data/
│   └── churn.csv
├── src/
│   ├── data_pipeline.py    # Loading, leakage audit, preprocessing
│   ├── baseline_models.py  # Classical ML models + tuning
│   ├── deep_models.py      # MLP / TabNet / TabR
│   ├── evaluate.py         # Metrics, CV, statistical tests
│   └── shap_analysis.py    # SHAP explainability
└── outputs/
    ├── tables/              # Generated CSV/JSON results
    └── figures/             # Generated PNG figures
```

## Troubleshooting

- **`ImportError` for `torch` or `pytorch_tabnet`**: install them (`pip install torch pytorch-tabnet`) or run with `--skip-deep`.
- **Long runtime**: use `python main.py --quick` first to validate the setup before committing to a full run.
