"""
Fairness / bias audit across demographic subgroups (Gender, Geography).

Addresses reviewer comment:
  "Fairness and bias analysis are not conducted despite discussing
   explainable AI in regulated financial applications. Performance should
   be analyzed across sensitive demographic groups to evaluate potential
   algorithmic bias." (Reviewer 3)

This is presented as an EXPLORATORY audit, not a certification of
fairness -- the Kaggle dataset is not guaranteed representative of any
real population, and this should be stated explicitly in the manuscript.

Run:
    python fairness_analysis.py

Writes:
    outputs/tables/table_fairness_gender.csv
    outputs/tables/table_fairness_geography.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_pipeline import load_raw_data, get_feature_target, build_preprocessor
from src.baseline_models import get_model_registry
from imblearn.over_sampling import SMOTE


def subgroup_metrics(y_true, y_pred, y_proba) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    recall = recall_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = np.nan  # only one class present in this subgroup's test fold
    positive_rate = y_pred.mean()  # for demographic parity
    return {
        "n": len(y_true), "churn_rate": round(y_true.mean(), 4),
        "ROC_AUC": round(auc, 4) if not np.isnan(auc) else np.nan,
        "Recall": round(recall, 4), "FPR": round(fpr, 4), "FNR": round(fnr, 4),
        "Positive_rate": round(positive_rate, 4),
    }


def evaluate_subgroups(model, pre, X_test_raw, y_test, group_col, random_state=42):
    Xte = pre.transform(X_test_raw)
    y_pred = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]

    rows = []
    for group_val in sorted(X_test_raw[group_col].unique()):
        mask = (X_test_raw[group_col] == group_val).values
        if mask.sum() < 10:
            continue
        m = subgroup_metrics(y_test.values[mask], y_pred[mask], y_proba[mask])
        m[group_col] = group_val
        rows.append(m)
    df = pd.DataFrame(rows).set_index(group_col)

    # Demographic parity difference: max - min positive-prediction rate across groups
    dp_diff = df["Positive_rate"].max() - df["Positive_rate"].min()
    # Equal opportunity difference: max - min Recall (TPR) across groups
    eo_diff = df["Recall"].max() - df["Recall"].min()
    df.loc["Max group gap"] = np.nan
    df.loc["Max group gap", "Positive_rate"] = round(dp_diff, 4)
    df.loc["Max group gap", "Recall"] = round(eo_diff, 4)
    return df


def main(model_name="CatBoost", random_state=42):
    df = load_raw_data()
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=random_state
    )

    pre = build_preprocessor()
    Xtr = pre.fit_transform(X_train)
    sm = SMOTE(random_state=random_state)
    Xtr_res, ytr_res = sm.fit_resample(Xtr, y_train)

    registry = get_model_registry(random_state)
    estimator, _ = registry[model_name]
    estimator.fit(Xtr_res, ytr_res)

    print(f"Fairness audit using {model_name} (single representative split, seed={random_state})\n")

    for group_col, fname in [("Gender", "table_fairness_gender.csv"),
                              ("Geography", "table_fairness_geography.csv")]:
        result = evaluate_subgroups(estimator, pre, X_test, y_test, group_col, random_state)
        result.to_csv(os.path.join(config.TABLES_DIR, fname))
        print(f"=== Fairness by {group_col} ===")
        print(result.to_string())
        print()

    print(
        "NOTE for manuscript: this is an EXPLORATORY fairness audit on a single "
        "public Kaggle dataset, which is not guaranteed to represent any real "
        "population. Results should not be interpreted as a certification of "
        "model fairness; see Limitations."
    )


if __name__ == "__main__":
    main()
