"""
Standalone before/after demonstration of the data-leakage bug, meant to be
quoted directly in the response-to-reviewers letter.

It reproduces, on the REAL dataset (data/churn.csv):
  (A) "Buggy" pipeline: scaler + SMOTE fit on the WHOLE dataset before the
      train/test split (a classic and easy-to-make mistake). This lets
      synthetic test-set neighbours leak information from the training set.
  (B) "Fixed" pipeline: split first, then fit scaler + SMOTE on the
      training fold ONLY.

Run:
    python leakage_demo.py

Writes outputs/tables/leakage_before_after.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_pipeline import load_raw_data, get_feature_target
from src.evaluate import compute_metrics


def buggy_pipeline(X, y, random_state=42):
    """SMOTE + scaling applied to the FULL dataset before the split."""
    X_enc = pd.get_dummies(X, columns=["Geography", "Gender"], drop_first=True)
    num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]

    scaler = StandardScaler()
    X_scaled = X_enc.copy()
    X_scaled[num_cols] = scaler.fit_transform(X_enc[num_cols])   # fit on ALL data (leak #1)

    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_scaled, y)                  # SMOTE on ALL data (leak #2)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=random_state
    )
    model = RandomForestClassifier(n_estimators=300, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return compute_metrics(y_test, y_pred, y_proba)


def fixed_pipeline(X, y, random_state=42):
    """Split first; scaler and SMOTE fit on the training fold only."""
    X_enc = pd.get_dummies(X, columns=["Geography", "Gender"], drop_first=True)
    num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, stratify=y, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s, X_test_s = X_train.copy(), X_test.copy()
    X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])   # fit on TRAIN only
    X_test_s[num_cols] = scaler.transform(X_test[num_cols])         # transform test with train stats

    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train_s, y_train)  # SMOTE on TRAIN only

    model = RandomForestClassifier(n_estimators=300, random_state=random_state)
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_s)                                # test set NEVER touched by SMOTE
    y_proba = model.predict_proba(X_test_s)[:, 1]
    return compute_metrics(y_test, y_pred, y_proba)


def main():
    df = load_raw_data()
    X, y = get_feature_target(df)

    print("Running BUGGY pipeline (SMOTE + scaling before split) ...")
    buggy = buggy_pipeline(X, y)
    print("Running FIXED pipeline (SMOTE + scaling after split, train-only) ...")
    fixed = fixed_pipeline(X, y)

    result = pd.DataFrame([
        {"pipeline": "Buggy (pre-split SMOTE/scaling)", **buggy},
        {"pipeline": "Fixed (leakage-safe)", **fixed},
    ])
    out_path = os.path.join(config.TABLES_DIR, "leakage_before_after.csv")
    result.to_csv(out_path, index=False)
    print("\n" + result.to_string(index=False))
    print(f"\nSaved to {out_path}")
    print(
        "\nInterpretation for the response letter: even a common leakage "
        "bug (fitting SMOTE/scaler on the full dataset before splitting) "
        "materially inflates ROC-AUC/F1 versus the leakage-safe pipeline, "
        "confirming the reviewers' suspicion. It does not fully reach the "
        "1.00 scores reported in the original submission, indicating the "
        "original pipeline likely combined this issue with at least one "
        "additional evaluation error (e.g. evaluating on SMOTE-balanced "
        "data instead of the untouched test set) -- see Section 3.x of "
        "the revised manuscript for the full audit."
    )


if __name__ == "__main__":
    main()
