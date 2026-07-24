"""
Robustness evaluation: geographic distribution shift, missing-feature
tolerance, and input noise sensitivity.

Addresses reviewer comment:
  "The manuscript does not assess model robustness under distribution
   shift, noisy inputs, or missing features, limiting confidence in
   deployment within real banking environments." (Reviewer 3)

Run:
    python robustness_analysis.py

Writes:
    outputs/tables/table_robustness_geography_shift.csv
    outputs/tables/table_robustness_missing_features.csv
    outputs/tables/table_robustness_noise.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_pipeline import load_raw_data, get_feature_target, build_preprocessor
from src.baseline_models import get_model_registry
from src.evaluate import compute_metrics
from imblearn.over_sampling import SMOTE


def fit_model(model_name, X_train, y_train, random_state=42):
    pre = build_preprocessor()
    Xtr = pre.fit_transform(X_train)
    sm = SMOTE(random_state=random_state)
    Xtr_res, ytr_res = sm.fit_resample(Xtr, y_train)
    registry = get_model_registry(random_state)
    estimator, _ = registry[model_name]
    estimator.fit(Xtr_res, ytr_res)
    return estimator, pre


# ---------------------------------------------------------------------------
# 1. Leave-one-geography-out distribution shift
# ---------------------------------------------------------------------------
def leave_one_geography_out(df, model_name="CatBoost", random_state=42):
    rows = []
    for held_out in ["France", "Germany", "Spain"]:
        train_df = df[df["Geography"] != held_out]
        test_df = df[df["Geography"] == held_out]

        X_train, y_train = get_feature_target(train_df)
        X_test, y_test = get_feature_target(test_df)

        model, pre = fit_model(model_name, X_train, y_train, random_state)
        Xte = pre.transform(X_test)
        y_pred = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]

        m = compute_metrics(y_test, y_pred, y_proba)
        m["held_out_country"] = held_out
        m["n_test"] = len(y_test)
        rows.append(m)
    return pd.DataFrame(rows).set_index("held_out_country")


# ---------------------------------------------------------------------------
# 2. Missing-feature robustness (randomly mask X% of cells with column mean/mode)
# ---------------------------------------------------------------------------
def missing_feature_robustness(X_test_raw, y_test, model, pre, missing_fracs=(0.0, 0.05, 0.10, 0.20),
                                random_state=42):
    rng = np.random.RandomState(random_state)
    rows = []
    for frac in missing_fracs:
        X_corrupted = X_test_raw.copy()
        # upcast numeric columns to float so mean-imputation doesn't hit
        # "Invalid value for dtype int64" on integer columns
        for col in X_corrupted.select_dtypes(include=[np.number]).columns:
            X_corrupted[col] = X_corrupted[col].astype(float)
        if frac > 0:
            for col in X_corrupted.columns:
                n_mask = int(len(X_corrupted) * frac)
                idx = rng.choice(X_corrupted.index, size=n_mask, replace=False)
                if pd.api.types.is_numeric_dtype(X_corrupted[col]):
                    fill_val = X_corrupted[col].mean()
                else:
                    fill_val = X_corrupted[col].mode()[0]
                X_corrupted.loc[idx, col] = fill_val

        Xte = pre.transform(X_corrupted)
        y_pred = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]
        m = compute_metrics(y_test, y_pred, y_proba)
        m["missing_fraction"] = frac
        rows.append(m)
    return pd.DataFrame(rows).set_index("missing_fraction")


# ---------------------------------------------------------------------------
# 3. Gaussian noise robustness (numeric features only, after scaling)
# ---------------------------------------------------------------------------
def noise_robustness(X_test_raw, y_test, model, pre, noise_stds=(0.0, 0.1, 0.25, 0.5), random_state=42):
    rng = np.random.RandomState(random_state)
    rows = []
    Xte_base = pre.transform(X_test_raw)
    n_numeric = len(config.NUMERIC_COLS)  # numeric block is first n columns after ColumnTransformer
    for std in noise_stds:
        Xte = Xte_base.copy()
        if std > 0:
            noise = rng.normal(0, std, size=(Xte.shape[0], n_numeric))
            Xte[:, :n_numeric] = Xte[:, :n_numeric] + noise
        y_pred = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:, 1]
        m = compute_metrics(y_test, y_pred, y_proba)
        m["noise_std"] = std
        rows.append(m)
    return pd.DataFrame(rows).set_index("noise_std")


def main(model_name="CatBoost", random_state=42):
    df = load_raw_data()
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=random_state
    )
    model, pre = fit_model(model_name, X_train, y_train, random_state)

    print(f"Robustness evaluation using {model_name} (seed={random_state})\n")

    print("=== 1. Leave-one-geography-out distribution shift ===")
    geo_df = leave_one_geography_out(df, model_name, random_state)
    geo_df.to_csv(os.path.join(config.TABLES_DIR, "table_robustness_geography_shift.csv"))
    print(geo_df[["Accuracy", "F1", "ROC_AUC", "n_test"]].to_string())

    print("\n=== 2. Missing-feature robustness ===")
    missing_df = missing_feature_robustness(X_test, y_test, model, pre, random_state=random_state)
    missing_df.to_csv(os.path.join(config.TABLES_DIR, "table_robustness_missing_features.csv"))
    print(missing_df[["Accuracy", "F1", "ROC_AUC"]].to_string())

    print("\n=== 3. Gaussian noise robustness ===")
    noise_df = noise_robustness(X_test, y_test, model, pre, random_state=random_state)
    noise_df.to_csv(os.path.join(config.TABLES_DIR, "table_robustness_noise.csv"))
    print(noise_df[["Accuracy", "F1", "ROC_AUC"]].to_string())

    print("\nSaved: table_robustness_geography_shift.csv, "
          "table_robustness_missing_features.csv, table_robustness_noise.csv")


if __name__ == "__main__":
    main()
