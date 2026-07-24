"""
Hyperparameter tuning for the three deep tabular models (MLP, TabR, TabNet)
via Optuna, matching the RandomizedSearchCV budget already applied to the
classical baselines in src/baseline_models.py.

Addresses reviewer/advisor comment:
  "Classical ML gets RandomizedSearchCV, but MLP/TabNet/TabR use a fixed
   configuration. Why were deep models not optimized using the same
   search budget?"

Design decisions (documented here so they can be copied into the
Methodology section):
  - Tuning is performed ONCE on a single representative seed (42) with an
    inner train/validation split, NOT across all 5 evaluation seeds --
    tuning on the same seeds used for final evaluation would leak model
    selection into the reported cross-validation results.
  - MLP and TabR: 30 trials each (relatively cheap to train).
  - TabNet: 15 trials, with a reduced max_epochs budget during search
    (40 instead of 100) and unchanged early-stopping patience, because
    TabNet training time is both the highest and the most variable of the
    three models (see Table 8 / table_computational_efficiency.csv,
    std >> mean for training time) -- an unconstrained 30-50 trial search
    risks multi-hour runtime on a CPU-only machine.
  - The best hyperparameters found here are then used to re-run the full
    5-seed x 5-fold repeated cross-validation (edit config.py's
    MLP_PARAMS / TABR_PARAMS / TABNET_PARAMS with the values printed
    below, then re-run main.py) so the headline results in Table 4 use
    tuned, not fixed, deep-model configurations.

Run:
    python tune_deep_models.py                  # all 3 models
    python tune_deep_models.py --model mlp       # just one
    python tune_deep_models.py --quick           # 3 trials each, smoke test

Writes:
    outputs/tables/table_hyperparameter_tuning.csv   (all models, one row per trial)
    outputs/tables/table_hyperparameter_best.csv      (best params per model)
"""
import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_pipeline import load_raw_data, get_feature_target, build_preprocessor
from imblearn.over_sampling import SMOTE

optuna.logging.set_verbosity(optuna.logging.WARNING)

TUNING_SEED = 42


def _prepare_data(random_state=TUNING_SEED):
    df = load_raw_data()
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=random_state
    )
    # inner split: tune on X_tr/X_val only, X_test is never touched here
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=random_state
    )
    pre = build_preprocessor()
    Xtr = pre.fit_transform(X_tr)
    Xval = pre.transform(X_val)

    sm = SMOTE(random_state=random_state)
    Xtr_res, ytr_res = sm.fit_resample(Xtr, y_tr)
    return Xtr_res, ytr_res, Xval, y_val.values


def objective_mlp(trial, Xtr, ytr, Xval, yval, input_dim):
    from src.deep_models import MLPClassifier
    n_layers = trial.suggest_int("n_layers", 1, 2)
    hidden_layers = [trial.suggest_categorical(f"units_l{i}", [16, 32, 64, 128]) for i in range(n_layers)]
    params = dict(
        hidden_layers=hidden_layers,
        dropout=trial.suggest_float("dropout", 0.1, 0.5),
        batch_norm=trial.suggest_categorical("batch_norm", [True, False]),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        max_epochs=60,
        patience=8,
    )
    model = MLPClassifier(input_dim, params=params, random_state=TUNING_SEED)
    model.fit(Xtr, ytr, Xval, yval)
    proba = model.predict_proba(Xval)[:, 1]
    return roc_auc_score(yval, proba)


def objective_tabr(trial, Xtr, ytr, Xval, yval, input_dim):
    from src.deep_models import TabRClassifier
    params = dict(
        embed_dim=trial.suggest_categorical("embed_dim", [16, 32, 64]),
        n_neighbors=trial.suggest_categorical("n_neighbors", [8, 16, 32]),
        dropout=trial.suggest_float("dropout", 0.1, 0.4),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256]),
        max_epochs=60,
        patience=8,
    )
    model = TabRClassifier(input_dim, params=params, random_state=TUNING_SEED)
    model.fit(Xtr, ytr, Xval, yval)
    proba = model.predict_proba(Xval)[:, 1]
    return roc_auc_score(yval, proba)


def objective_tabnet(trial, Xtr, ytr, Xval, yval, input_dim):
    from src.deep_models import TabNetClassifierWrapper
    nd_na = trial.suggest_categorical("n_d_n_a", [8, 16, 24])
    params = dict(
        n_d=nd_na, n_a=nd_na,
        n_steps=trial.suggest_int("n_steps", 3, 6),
        gamma=trial.suggest_float("gamma", 1.0, 2.0),
        lambda_sparse=trial.suggest_float("lambda_sparse", 1e-4, 1e-2, log=True),
        max_epochs=40,   # reduced budget during search (see module docstring)
        patience=10,
        batch_size=trial.suggest_categorical("batch_size", [128, 256, 512]),
    )
    model = TabNetClassifierWrapper(input_dim, params=params, random_state=TUNING_SEED)
    model.fit(Xtr, ytr, Xval, yval)
    proba = model.predict_proba(Xval)[:, 1]
    return roc_auc_score(yval, proba)


MODEL_CONFIG = {
    "mlp": {"objective": objective_mlp, "n_trials": 30},
    "tabr": {"objective": objective_tabr, "n_trials": 30},
    "tabnet": {"objective": objective_tabnet, "n_trials": 15},
}


def tune_one_model(name, n_trials, Xtr, ytr, Xval, yval, input_dim):
    print(f"\n{'='*60}\nTuning {name} ({n_trials} trials)\n{'='*60}")
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=TUNING_SEED))
    objective_fn = MODEL_CONFIG[name]["objective"]

    t0 = time.time()
    study.optimize(
        lambda trial: objective_fn(trial, Xtr, ytr, Xval, yval, input_dim),
        n_trials=n_trials, show_progress_bar=False,
    )
    elapsed = time.time() - t0

    print(f"Best {name} val ROC-AUC: {study.best_value:.4f}  ({elapsed:.1f}s total, "
          f"{elapsed/n_trials:.1f}s/trial)")
    print(f"Best params: {study.best_params}")

    trials_df = study.trials_dataframe()
    trials_df["model"] = name
    return trials_df, study.best_params, study.best_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIG.keys()), default=None,
                         help="Tune a single model only (default: all three)")
    parser.add_argument("--quick", action="store_true", help="3 trials per model, smoke test")
    args = parser.parse_args()

    try:
        import torch  # noqa
    except ImportError:
        print("[ERROR] torch is required for this script. Install with: pip install torch pytorch-tabnet")
        return

    print("Preparing data (single inner train/validation split, seed=42) ...")
    Xtr, ytr, Xval, yval = _prepare_data()
    input_dim = Xtr.shape[1]
    print(f"Train: {Xtr.shape}, Val: {Xval.shape}")

    models_to_tune = [args.model] if args.model else list(MODEL_CONFIG.keys())

    all_trials = []
    best_rows = []
    for name in models_to_tune:
        n_trials = 3 if args.quick else MODEL_CONFIG[name]["n_trials"]
        trials_df, best_params, best_value = tune_one_model(name, n_trials, Xtr, ytr, Xval, yval, input_dim)
        all_trials.append(trials_df)
        best_rows.append({"model": name, "best_val_roc_auc": round(best_value, 4),
                           "n_trials": n_trials, "best_params": best_params})

    combined_trials = pd.concat(all_trials, ignore_index=True)
    combined_trials.to_csv(os.path.join(config.TABLES_DIR, "table_hyperparameter_tuning.csv"), index=False)

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(os.path.join(config.TABLES_DIR, "table_hyperparameter_best.csv"), index=False)

    print("\n" + "=" * 60)
    print("DONE. Best hyperparameters per model:")
    print(best_df.to_string(index=False))
    print("\nNext step: copy these values into config.py (MLP_PARAMS / TABR_PARAMS / "
          "TABNET_PARAMS), then re-run 'python main.py' so Table 4 reflects tuned "
          "deep-model configurations.")


if __name__ == "__main__":
    main()
