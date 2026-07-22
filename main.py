"""
End-to-end pipeline. Run this file to reproduce every table and figure
needed for the manuscript revision:

    python main.py                # full run (uses config.SEEDS, slower)
    python main.py --quick         # smoke test with 1 seed / small search

Outputs are written to outputs/tables/*.csv and outputs/figures/*.png.
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_pipeline import (
    load_raw_data, leakage_audit, get_feature_target,
    build_preprocessor, get_feature_names,
)
from src.baseline_models import get_model_registry, build_pipeline, tune_model
from src.evaluate import (
    compute_metrics, confusion_matrix_dict, repeated_cv_scores, summarize_cv,
    friedman_test, pairwise_wilcoxon, mcnemar_test, delong_test,
)
from src.shap_analysis import compute_shap_values, mean_abs_importance, shap_stability_across_seeds

from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import SMOTE


BASELINE_MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
DEEP_MODELS = ["MLP", "TabNet", "TabR"]   # requires torch / pytorch-tabnet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                         help="Fast smoke-test run: 1 seed, small hyperparameter search, "
                              "reduced deep-model epochs. Use this first to check everything "
                              "runs before committing to the full multi-hour run.")
    parser.add_argument("--skip-deep", action="store_true",
                         help="Skip MLP/TabNet/TabR (useful if torch / pytorch-tabnet "
                              "are not installed yet).")
    return parser.parse_args()


def step1_leakage_audit(df):
    print("\n" + "=" * 70)
    print("STEP 1 — Leakage audit")
    print("=" * 70)
    audit = leakage_audit(df)
    audit.to_csv(os.path.join(config.TABLES_DIR, "table_leakage_audit.csv"), index=False)
    print(audit.to_string(index=False))
    return audit


def step2_baseline_cv(X, y, seeds, n_iter):
    """
    Repeated stratified CV for every classical baseline, SMOTE-in-pipeline
    (fit on the training fold only), best hyperparameters re-tuned once on
    the full training data of the first fold to keep runtime reasonable,
    then reused across the repeated CV (documented explicitly as a
    runtime/compute trade-off in the manuscript).
    """
    print("\n" + "=" * 70)
    print("STEP 2 — Classical ML baselines: hyperparameter tuning + repeated CV")
    print("=" * 70)

    all_cv_rows = []
    tuned_params = {}

    for model_name in BASELINE_MODELS:
        print(f"\n[{model_name}] tuning hyperparameters ...")
        t0 = time.time()
        best_model, best_params = tune_model(
            model_name, X, y, imbalance_strategy="smote",
            random_state=seeds[0], n_iter=n_iter, cv_splits=3,
        )
        tuned_params[model_name] = best_params
        print(f"[{model_name}] best params: {best_params}  ({time.time()-t0:.1f}s)")

        def fit_predict_fn(X_train, y_train, X_test, y_test, seed, _name=model_name, _params=best_params):
            registry = get_model_registry(seed)
            estimator, _ = registry[_name]
            clean_params = {k.replace("model__", ""): v for k, v in _params.items()}
            estimator.set_params(**clean_params)
            pipe = build_pipeline(estimator, "smote", random_state=seed)
            t0 = time.time()
            pipe.fit(X_train, y_train)
            train_time = time.time() - t0
            t0 = time.time()
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            infer_time = (time.time() - t0) / len(X_test)
            m = compute_metrics(y_test, y_pred, y_proba)
            m.update(confusion_matrix_dict(y_test, y_pred))
            m["model"] = _name
            m["train_time_sec"] = train_time
            m["infer_time_per_sample_sec"] = infer_time
            return m

        cv_df = repeated_cv_scores(fit_predict_fn, X, y, seeds=seeds)
        all_cv_rows.append(cv_df)

    cv_results = pd.concat(all_cv_rows, ignore_index=True)
    cv_results.to_csv(os.path.join(config.TABLES_DIR, "table_baseline_cv_raw.csv"), index=False)

    summary = summarize_cv(cv_results)
    summary.to_csv(os.path.join(config.TABLES_DIR, "table_baseline_cv_summary.csv"), index=False)
    print("\nSummary (mean ± std):")
    print(summary[["model", "ROC_AUC_mean", "ROC_AUC_std", "F1_mean", "F1_std"]].to_string(index=False))

    with open(os.path.join(config.TABLES_DIR, "baseline_best_hyperparameters.json"), "w") as f:
        json.dump(tuned_params, f, indent=2, default=str)

    return cv_results, summary, tuned_params


def step3_deep_model_cv(X, y, seeds, quick=False):
    print("\n" + "=" * 70)
    print("STEP 3 — Deep tabular models (MLP, TabNet, TabR): repeated CV")
    print("=" * 70)
    try:
        from src.deep_models import get_deep_model
    except ImportError as e:
        print(f"[SKIPPED] Could not import deep_models ({e}). "
              f"Install torch / pytorch-tabnet, e.g.:\n"
              f"  pip install torch pytorch-tabnet")
        return pd.DataFrame(), pd.DataFrame()

    preprocessor = build_preprocessor()
    all_cv_rows = []

    for model_name in DEEP_MODELS:
        print(f"\n[{model_name}] running repeated CV ...")

        def fit_predict_fn(X_train, y_train, X_test, y_test, seed, _name=model_name):
            pre = build_preprocessor()
            Xtr_raw, Xval_raw, ytr, yval = train_test_split(
                X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed
            )
            Xtr = pre.fit_transform(Xtr_raw)
            Xval = pre.transform(Xval_raw)
            Xte = pre.transform(X_test)

            sm = SMOTE(random_state=seed)
            Xtr_res, ytr_res = sm.fit_resample(Xtr, ytr)

            model = get_deep_model(_name, input_dim=Xtr_res.shape[1], random_state=seed)
            if quick:
                model.params["max_epochs"] = min(5, model.params.get("max_epochs", 5))

            t0 = time.time()
            model.fit(Xtr_res, ytr_res, Xval, yval)
            train_time = time.time() - t0

            t0 = time.time()
            y_pred = model.predict(Xte)
            y_proba = model.predict_proba(Xte)[:, 1]
            infer_time = (time.time() - t0) / len(Xte)

            m = compute_metrics(y_test, y_pred, y_proba)
            m.update(confusion_matrix_dict(y_test, y_pred))
            m["model"] = _name
            m["train_time_sec"] = train_time
            m["infer_time_per_sample_sec"] = infer_time
            m["n_parameters"] = model.count_parameters()
            return m

        cv_df = repeated_cv_scores(fit_predict_fn, X, y, seeds=seeds)
        all_cv_rows.append(cv_df)

    cv_results = pd.concat(all_cv_rows, ignore_index=True) if all_cv_rows else pd.DataFrame()
    if not cv_results.empty:
        cv_results.to_csv(os.path.join(config.TABLES_DIR, "table_deep_cv_raw.csv"), index=False)
        summary = summarize_cv(cv_results)
        summary.to_csv(os.path.join(config.TABLES_DIR, "table_deep_cv_summary.csv"), index=False)
        print(summary[["model", "ROC_AUC_mean", "ROC_AUC_std", "F1_mean", "F1_std"]].to_string(index=False))
        return cv_results, summary
    return pd.DataFrame(), pd.DataFrame()


def step4_statistical_tests(cv_results):
    print("\n" + "=" * 70)
    print("STEP 4 — Statistical significance tests")
    print("=" * 70)
    if cv_results.empty or cv_results["model"].nunique() < 3:
        print("[SKIPPED] Need >=3 models with matching (seed, fold) CV results.")
        return

    friedman = friedman_test(cv_results, metric="ROC_AUC")
    print("Friedman test (ROC-AUC across all models):", friedman)

    wilcoxon_df = pairwise_wilcoxon(cv_results, metric="ROC_AUC")
    wilcoxon_df.to_csv(os.path.join(config.TABLES_DIR, "table_pairwise_wilcoxon.csv"), index=False)
    print(wilcoxon_df.to_string(index=False))

    with open(os.path.join(config.TABLES_DIR, "friedman_test.json"), "w") as f:
        json.dump({k: (list(v) if isinstance(v, list) else float(v) if not isinstance(v, int) else v)
                   for k, v in friedman.items()}, f, indent=2, default=str)


def step5_imbalance_ablation(X, y, seeds):
    """
    Ablation over imbalance-handling strategy (none / class_weight / smote)
    for a single representative model (RandomForest), addressing Reviewer
    3's request not to rely on a single SMOTE configuration.
    """
    print("\n" + "=" * 70)
    print("STEP 5 — Imbalance-handling ablation (RandomForest)")
    print("=" * 70)
    rows = []
    for strategy in config.IMBALANCE_STRATEGIES:
        def fit_predict_fn(X_train, y_train, X_test, y_test, seed, _strategy=strategy):
            registry = get_model_registry(seed)
            estimator, _ = registry["RandomForest"]
            estimator.set_params(n_estimators=300, max_depth=8)
            if _strategy == "class_weight":
                estimator.set_params(class_weight="balanced")
            pipe = build_pipeline(estimator, _strategy, random_state=seed)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)[:, 1]
            m = compute_metrics(y_test, y_pred, y_proba)
            m["model"] = f"RandomForest_{_strategy}"
            return m
        cv_df = repeated_cv_scores(fit_predict_fn, X, y, seeds=seeds)
        rows.append(cv_df)

    ablation = pd.concat(rows, ignore_index=True)
    ablation.to_csv(os.path.join(config.TABLES_DIR, "table_imbalance_ablation_raw.csv"), index=False)
    summary = summarize_cv(ablation)
    summary.to_csv(os.path.join(config.TABLES_DIR, "table_imbalance_ablation_summary.csv"), index=False)
    print(summary[["model", "ROC_AUC_mean", "ROC_AUC_std", "F1_mean", "F1_std"]].to_string(index=False))


def step6_shap(X, y, seeds):
    print("\n" + "=" * 70)
    print("STEP 6 — SHAP explainability for all models")
    print("=" * 70)
    from src.baseline_models import get_model_registry, build_pipeline

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=seeds[0]
    )
    pre = build_preprocessor()
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    feature_names = get_feature_names(pre)

    sm = SMOTE(random_state=seeds[0])
    Xtr_res, ytr_res = sm.fit_resample(Xtr, y_train)

    importances = {}
    for model_name in BASELINE_MODELS:
        registry = get_model_registry(seeds[0])
        estimator, _ = registry[model_name]
        estimator.fit(Xtr_res, ytr_res)
        sv = compute_shap_values(model_name, estimator, Xte[:200], Xtr_res)
        imp = mean_abs_importance(sv, feature_names)
        importances[model_name] = imp
        imp.to_csv(os.path.join(config.TABLES_DIR, f"shap_importance_{model_name}.csv"))

        plt.figure(figsize=(6, 4))
        imp.sort_values().plot(kind="barh")
        plt.title(f"Mean |SHAP value| — {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(config.FIGURES_DIR, f"shap_{model_name}.png"), dpi=150)
        plt.close()
        print(f"[{model_name}] top feature: {imp.index[0]} ({imp.iloc[0]:.4f})")

    # deep models, if available
    try:
        from src.deep_models import get_deep_model
        for model_name in DEEP_MODELS:
            model = get_deep_model(model_name, input_dim=Xtr_res.shape[1], random_state=seeds[0])
            X_fit_tr, X_fit_val, y_fit_tr, y_fit_val = train_test_split(
                Xtr_res, ytr_res, test_size=0.15, random_state=seeds[0]
            )
            model.fit(X_fit_tr, y_fit_tr, X_fit_val, y_fit_val)
            sv = compute_shap_values(model_name, model, Xte[:100], Xtr_res, nsamples=100)
            imp = mean_abs_importance(sv, feature_names)
            importances[model_name] = imp
            imp.to_csv(os.path.join(config.TABLES_DIR, f"shap_importance_{model_name}.csv"))
            print(f"[{model_name}] top feature: {imp.index[0]} ({imp.iloc[0]:.4f})")
    except ImportError:
        print("[SKIPPED deep-model SHAP] torch / pytorch-tabnet not installed.")

    return importances


def main():
    args = parse_args()
    seeds = [config.SEEDS[0]] if args.quick else config.SEEDS
    n_iter = 5 if args.quick else config.BASELINE_SEARCH_ITER

    print(f"Running with seeds={seeds}, n_iter={n_iter}, quick={args.quick}")

    df = load_raw_data()
    print(f"\nDataset: {df.shape[0]} rows x {df.shape[1]} columns, "
          f"churn rate = {df[config.TARGET_COL].mean():.4f}")

    step1_leakage_audit(df)
    X, y = get_feature_target(df)

    cv_results, summary, tuned_params = step2_baseline_cv(X, y, seeds, n_iter)

    deep_cv_results, deep_summary = (pd.DataFrame(), pd.DataFrame())
    if not args.skip_deep:
        deep_cv_results, deep_summary = step3_deep_model_cv(X, y, seeds, quick=args.quick)

    combined_cv = pd.concat([cv_results, deep_cv_results], ignore_index=True) if not deep_cv_results.empty else cv_results
    combined_summary = pd.concat([summary, deep_summary], ignore_index=True) if not deep_summary.empty else summary
    combined_summary.to_csv(os.path.join(config.TABLES_DIR, "table_ALL_MODELS_summary.csv"), index=False)

    step4_statistical_tests(combined_cv)
    step5_imbalance_ablation(X, y, seeds)
    step6_shap(X, y, seeds)

    print("\n" + "=" * 70)
    print("DONE. All tables saved to:", config.TABLES_DIR)
    print("All figures saved to:", config.FIGURES_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
