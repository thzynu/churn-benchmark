"""
Data loading, leakage audit, and leakage-safe preprocessing pipeline.

Addresses reviewer comments:
  - "The manuscript should verify the absence of leakage" (Reviewer 3, 5)
  - "The manuscript should explicitly state that oversampling was applied
     only to the training partition" (Reviewer 3)

Key design decision (see leakage_audit()):
  - RowNumber / CustomerId / Surname are pure identifiers -> dropped.
  - No feature in this 14-column dataset shows near-perfect correlation
    with the target (max |corr| = 0.29 for Age), unlike the 18-column
    variant of this dataset that includes a `Complain` field. That
    variant is NOT what this study uses (see Data Availability section).
  - The scaler/encoder and SMOTE must be fit ONLY on the training fold,
    never on the full dataset, to avoid leaking test-set information
    into preprocessing statistics.
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_raw_data(path: str = config.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def leakage_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the leakage-audit table required by the revision
    (Methodology section 3.x: Leakage Audit and Feature Availability
    Assessment). Returns a DataFrame that can be dropped straight into
    the manuscript as a table.
    """
    corr = df.drop(columns=config.ID_COLS).copy()
    corr = pd.get_dummies(corr, columns=config.CATEGORICAL_COLS, drop_first=True)
    corr = corr.corr()[config.TARGET_COL].drop(config.TARGET_COL)

    rows = []
    for col in config.ID_COLS:
        rows.append({
            "Feature": col,
            "Available at prediction time": "No business value",
            "Leakage risk": "ID leakage",
            "Correlation with target": "N/A",
            "Decision": "Dropped",
        })
    for col in config.NUMERIC_COLS + config.CATEGORICAL_COLS + config.BINARY_COLS:
        matching = [c for c in corr.index if c == col or c.startswith(col + "_")]
        if matching:
            max_abs_corr = corr[matching].abs().max()
        else:
            max_abs_corr = np.nan
        rows.append({
            "Feature": col,
            "Available at prediction time": "Yes (pre-churn attribute)",
            "Leakage risk": "Low" if (pd.isna(max_abs_corr) or max_abs_corr < 0.3) else "Verify",
            "Correlation with target": round(float(max_abs_corr), 4) if not pd.isna(max_abs_corr) else "N/A",
            "Decision": "Kept",
        })
    audit_df = pd.DataFrame(rows)
    return audit_df


def get_feature_target(df: pd.DataFrame):
    df = df.drop(columns=config.ID_COLS)
    y = df[config.TARGET_COL].copy()
    X = df.drop(columns=[config.TARGET_COL])
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer that must be fit ONLY on training data inside each
    CV fold (never on the full dataset) -- this is what makes the
    pipeline leakage-safe with respect to preprocessing statistics.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_COLS),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), config.CATEGORICAL_COLS),
            ("bin", "passthrough", config.BINARY_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer):
    """Human-readable feature names after the ColumnTransformer, for SHAP plots."""
    num_names = config.NUMERIC_COLS
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(config.CATEGORICAL_COLS))
    bin_names = config.BINARY_COLS
    return num_names + cat_names + bin_names


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Churn rate: {df[config.TARGET_COL].mean():.4f}")

    audit = leakage_audit(df)
    audit_path = os.path.join(config.TABLES_DIR, "leakage_audit.csv")
    audit.to_csv(audit_path, index=False)
    print(f"\nLeakage audit table saved to {audit_path}")
    print(audit.to_string(index=False))
