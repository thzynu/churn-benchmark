"""
Two ablation studies not yet covered by main.py:

  (a) Feature-leakage ablation: "naive full-feature" pipeline (keeping the
      raw identifier columns RowNumber, CustomerId, and a label-encoded
      Surname) vs. the leakage-safe feature set used throughout this study
      (identifiers dropped, per Table 1). This directly answers the
      advisor's request for a Feature Leakage ablation group, and, since
      this 14-column dataset has no Complain-style proxy variable,
      demonstrates whether naively keeping identifier columns inflates
      performance here (a useful negative result if it does not).

  (b) MLP architecture ablation: baseline MLP vs. +Dropout vs. +BatchNorm
      vs. +Both (the current tuned configuration), isolating the
      contribution of each regularization component -- addresses the
      advisor's explicit request for a Dropout/BatchNorm ablation.

Run AFTER main.py (uses the same seeds/folds convention):
    python ablation_extra.py

Writes:
    outputs/tables/table_ablation_feature_leakage.csv
    outputs/tables/table_ablation_mlp_architecture.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_pipeline import load_raw_data
from src.baseline_models import get_model_registry, build_pipeline
from src.evaluate import compute_metrics, repeated_cv_scores, summarize_cv


# ---------------------------------------------------------------------------
# (a) Feature-leakage ablation
# ---------------------------------------------------------------------------
def build_full_feature_preprocessor() -> ColumnTransformer:
    """
    'Naive' preprocessor that keeps RowNumber and CustomerId as raw numeric
    features and label-encodes Surname (arbitrary ordinal, exactly what a
    careless practitioner might do by passing every column to
    pd.factorize/OrdinalEncoder without first auditing for leakage).
    """
    naive_numeric = ["RowNumber", "CustomerId"] + config.NUMERIC_COLS
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), naive_numeric),
            ("surname", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["Surname"]),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), config.CATEGORICAL_COLS),
            ("bin", "passthrough", config.BINARY_COLS),
        ],
        remainder="drop",
    )


def get_feature_target_full(df: pd.DataFrame):
    """Like src.data_pipeline.get_feature_target, but keeps RowNumber/
    CustomerId/Surname instead of dropping them (for the 'full-feature'
    ablation arm only)."""
    y = df[config.TARGET_COL].copy()
    X = df.drop(columns=[config.TARGET_COL])
    return X, y


def feature_leakage_ablation(model_name="CatBoost", seeds=None):
    print("=" * 60)
    print("(a) Feature-leakage ablation")
    print("=" * 60)
    df = load_raw_data()
    seeds = seeds or config.SEEDS

    # --- Scenario A: leakage-safe (current default, identifiers dropped) ---
    X_safe, y_safe = df.drop(columns=config.ID_COLS + [config.TARGET_COL]), df[config.TARGET_COL]

    def fit_predict_safe(X_train, y_train, X_test, y_test, seed):
        registry = get_model_registry(seed)
        estimator, _ = registry[model_name]
        pipe = build_pipeline(estimator, "smote", random_state=seed)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        m = compute_metrics(y_test, y_pred, y_proba)
        m["scenario"] = "Leakage-safe (IDs dropped)"
        return m

    cv_safe = repeated_cv_scores(fit_predict_safe, X_safe, y_safe, seeds=seeds)
    cv_safe["scenario"] = "Leakage-safe (IDs dropped)"

    # --- Scenario B: naive full features (RowNumber, CustomerId, Surname kept) ---
    X_full, y_full = get_feature_target_full(df)

    def fit_predict_full(X_train, y_train, X_test, y_test, seed):
        registry = get_model_registry(seed)
        estimator, _ = registry[model_name]
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        pre = build_full_feature_preprocessor()
        pipe = ImbPipeline([
            ("preprocessor", pre),
            ("smote", SMOTE(random_state=seed)),
            ("model", estimator),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        m = compute_metrics(y_test, y_pred, y_proba)
        m["scenario"] = "Naive full features (IDs kept)"
        return m

    cv_full = repeated_cv_scores(fit_predict_full, X_full, y_full, seeds=seeds)
    cv_full["scenario"] = "Naive full features (IDs kept)"

    combined = pd.concat([cv_safe, cv_full], ignore_index=True)
    combined.to_csv(os.path.join(config.TABLES_DIR, "table_ablation_feature_leakage_raw.csv"), index=False)

    summary = combined.groupby("scenario")[
        ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "MCC"]
    ].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(config.TABLES_DIR, "table_ablation_feature_leakage.csv"), index=False)

    print(f"\nModel: {model_name}")
    print(summary[["scenario", "ROC_AUC_mean", "ROC_AUC_std", "F1_mean", "F1_std"]].to_string(index=False))

    delta_auc = (summary.loc[summary.scenario.str.contains("Naive"), "ROC_AUC_mean"].values[0] -
                 summary.loc[summary.scenario.str.contains("Leakage-safe"), "ROC_AUC_mean"].values[0])
    print(f"\nDelta ROC-AUC (naive - leakage-safe): {delta_auc:+.4f}")
    if abs(delta_auc) < 0.005:
        print("-> Negligible difference: including RowNumber/CustomerId/Surname does NOT "
              "meaningfully change performance on this dataset (expected, since a proper "
              "train/test split prevents these identifiers from acting as memorization keys).")
    else:
        print("-> Non-negligible difference: worth investigating further before concluding "
              "identifiers are harmless here.")

    return summary


# ---------------------------------------------------------------------------
# (b) MLP architecture ablation: Dropout / BatchNorm
# ---------------------------------------------------------------------------
def mlp_architecture_ablation(seeds=None):
    print("\n" + "=" * 60)
    print("(b) MLP architecture ablation (Dropout / BatchNorm)")
    print("=" * 60)
    try:
        from src.deep_models import get_deep_model
    except ImportError as e:
        print(f"[SKIPPED] torch not installed ({e}). Install torch to run this ablation.")
        return None

    from src.data_pipeline import load_raw_data, get_feature_target, build_preprocessor
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split

    df = load_raw_data()
    X, y = get_feature_target(df)
    seeds = seeds or config.SEEDS

    variants = {
        "Baseline (no Dropout, no BatchNorm)": dict(dropout=0.0, batch_norm=False),
        "+Dropout only":                        dict(dropout=config.MLP_PARAMS["dropout"], batch_norm=False),
        "+BatchNorm only":                      dict(dropout=0.0, batch_norm=True),
        "+Both (current tuned config)":         dict(dropout=config.MLP_PARAMS["dropout"], batch_norm=True),
    }

    rows = []
    for variant_name, overrides in variants.items():
        print(f"\nRunning: {variant_name} ...")

        def fit_predict_fn(X_train, y_train, X_test, y_test, seed, _overrides=overrides, _name=variant_name):
            pre = build_preprocessor()
            Xtr_raw, Xval_raw, ytr, yval = train_test_split(
                X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed
            )
            Xtr = pre.fit_transform(Xtr_raw)
            Xval = pre.transform(Xval_raw)
            Xte = pre.transform(X_test)

            sm = SMOTE(random_state=seed)
            Xtr_res, ytr_res = sm.fit_resample(Xtr, ytr)

            params = dict(config.MLP_PARAMS)
            params.update(_overrides)
            model = get_deep_model("MLP", input_dim=Xtr_res.shape[1], random_state=seed)
            model.params = params
            from src.deep_models import set_seed as _set_seed
            _set_seed(seed)  # re-seed so all 4 variants start from comparable initial weights
            model.model = type(model.model)(
                Xtr_res.shape[1], params["hidden_layers"], params["dropout"], params["batch_norm"]
            ).to(model.device)

            model.fit(Xtr_res, ytr_res, Xval, yval)
            y_pred = model.predict(Xte)
            y_proba = model.predict_proba(Xte)[:, 1]
            m = compute_metrics(y_test, y_pred, y_proba)
            m["variant"] = _name
            return m

        cv_df = repeated_cv_scores(fit_predict_fn, X, y, seeds=seeds)
        cv_df["variant"] = variant_name
        rows.append(cv_df)

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(os.path.join(config.TABLES_DIR, "table_ablation_mlp_architecture_raw.csv"), index=False)

    summary = combined.groupby("variant")[
        ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "MCC"]
    ].agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    # keep a sensible display order
    order = list(variants.keys())
    summary["variant"] = pd.Categorical(summary["variant"], categories=order, ordered=True)
    summary = summary.sort_values("variant")
    summary.to_csv(os.path.join(config.TABLES_DIR, "table_ablation_mlp_architecture.csv"), index=False)

    print("\n" + summary[["variant", "ROC_AUC_mean", "ROC_AUC_std", "F1_mean", "F1_std"]].to_string(index=False))
    return summary


def main():
    feature_leakage_ablation(model_name="CatBoost")
    mlp_architecture_ablation()
    print("\nSaved: table_ablation_feature_leakage.csv, table_ablation_mlp_architecture.csv "
          "(+ _raw.csv versions with per-fold detail)")


if __name__ == "__main__":
    main()
