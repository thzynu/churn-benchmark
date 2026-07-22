"""
Evaluation utilities: metrics, repeated stratified CV, statistical
significance tests, and calibration.

Addresses reviewer comments:
  - "A single 80/20 train-test split ... Repeated stratified k-fold
     cross-validation ... should be performed" (Reviewer 3)
  - "No statistical significance analysis ... Appropriate statistical
     tests should be included" (Reviewer 3)
  - "The justification for selecting MLP ... is weak because the
     reported ROC-AUC improvements ... are marginal and may not be
     statistically significant" (Reviewer 3)
  - "Additional measures such as PR-AUC, MCC, Balanced Accuracy, Brier
     Score, calibration curves, and expected calibration error should be
     included" (Reviewer 3)
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, brier_score_loss, confusion_matrix,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------
def expected_calibration_error(y_true, y_proba, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_proba >= lo) & (y_proba < hi) if hi < 1.0 else (y_proba >= lo) & (y_proba <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = y_proba[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return ece


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
        "PR_AUC": average_precision_score(y_true, y_proba),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Brier_Score": brier_score_loss(y_true, y_proba),
        "ECE": expected_calibration_error(y_true, y_proba),
    }


def confusion_matrix_dict(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


# ---------------------------------------------------------------------------
# Repeated stratified cross-validation
# ---------------------------------------------------------------------------
def repeated_cv_scores(fit_predict_fn, X, y, seeds=None, n_splits=None) -> pd.DataFrame:
    """
    fit_predict_fn(X_train, y_train, X_test, y_test) -> dict of metrics
    (the caller is responsible for fitting a fresh model per fold, and for
    making sure any preprocessing / SMOTE is fit on X_train only).

    Returns one row per (seed, fold) so mean/std/CI can be computed
    downstream, and so that per-fold predictions can be paired up for
    McNemar / Wilcoxon tests between models.
    """
    seeds = seeds or config.SEEDS
    n_splits = n_splits or config.N_SPLITS
    rows = []
    is_frame = isinstance(X, pd.DataFrame)
    y_arr = np.asarray(y)
    if is_frame:
        X_idx = X.reset_index(drop=True)
    else:
        X_idx = np.asarray(X)

    for seed in seeds:
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=1, random_state=seed)
        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_idx, y_arr)):
            if is_frame:
                X_train, X_test = X_idx.iloc[train_idx], X_idx.iloc[test_idx]
            else:
                X_train, X_test = X_idx[train_idx], X_idx[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]
            metrics = fit_predict_fn(X_train, y_train, X_test, y_test, seed=seed)
            metrics["seed"] = seed
            metrics["fold"] = fold_i
            rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_cv(df: pd.DataFrame, metric_cols=None) -> pd.DataFrame:
    metric_cols = metric_cols or [
        "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC",
        "MCC", "Balanced_Accuracy", "Brier_Score", "ECE",
    ]
    summary = df.groupby("model")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]

    # 95% CI (normal approximation) for the headline metric ROC_AUC
    n = df.groupby("model").size()
    ci = 1.96 * summary["ROC_AUC_std"] / np.sqrt(n)
    summary["ROC_AUC_95CI"] = ci
    return summary.reset_index()


# ---------------------------------------------------------------------------
# Statistical significance tests
# ---------------------------------------------------------------------------
def friedman_test(cv_results: pd.DataFrame, metric="ROC_AUC"):
    """
    Friedman test across >=3 models using per-fold scores. Requires the
    same (seed, fold) index to exist for every model (paired comparison).
    """
    pivot = cv_results.pivot_table(index=["seed", "fold"], columns="model", values=metric)
    pivot = pivot.dropna()
    stat, p = friedmanchisquare(*[pivot[c].values for c in pivot.columns])
    return {"statistic": stat, "p_value": p, "models": list(pivot.columns), "n_pairs": len(pivot)}


def pairwise_wilcoxon(cv_results: pd.DataFrame, metric="ROC_AUC", correction="holm"):
    """
    Pairwise Wilcoxon signed-rank test between every pair of models on the
    matched (seed, fold) scores, with Holm-Bonferroni correction for
    multiple comparisons -- answers Reviewer 3's request to verify whether
    the reported ROC-AUC differences between models are meaningful.
    """
    pivot = cv_results.pivot_table(index=["seed", "fold"], columns="model", values=metric).dropna()
    models = list(pivot.columns)
    pairs, pvals, deltas = [], [], []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = pivot[models[i]].values, pivot[models[j]].values
            try:
                stat, p = wilcoxon(a, b)
            except ValueError:
                stat, p = np.nan, 1.0
            pairs.append(f"{models[i]} vs {models[j]}")
            pvals.append(p)
            deltas.append(np.mean(a - b))

    reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method=correction)
    return pd.DataFrame({
        "comparison": pairs,
        "mean_diff": deltas,
        "p_value_raw": pvals,
        "p_value_adjusted": p_adj,
        "significant_at_0.05": reject,
    })


def mcnemar_test(y_true, pred_a, pred_b):
    """McNemar's test on a single shared held-out test set for two models'
    class predictions -- complements the CV-based Wilcoxon test with a
    test that only needs discordant-pair counts."""
    from statsmodels.stats.contingency_tables import mcnemar
    y_true = np.asarray(y_true)
    correct_a = (np.asarray(pred_a) == y_true)
    correct_b = (np.asarray(pred_b) == y_true)
    n01 = int(((~correct_a) & correct_b).sum())  # a wrong, b right
    n10 = int((correct_a & (~correct_b)).sum())  # a right, b wrong
    table = [[0, n01], [n10, 0]]
    result = mcnemar(table, exact=(n01 + n10 < 25))
    return {"statistic": result.statistic, "p_value": result.pvalue, "n01": n01, "n10": n10}


def delong_roc_variance(y_true, y_proba):
    """
    Fast DeLong variance estimate for a single ROC-AUC (Sun & Xu, 2014
    algorithm), used as a building block for delong_test below.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-y_proba)
    y_sorted = y_true[order]
    proba_sorted = y_proba[order]

    pos = proba_sorted[y_sorted == 1]
    neg = proba_sorted[y_sorted == 0]
    m, n = len(pos), len(neg)

    def midrank(x):
        sorted_idx = np.argsort(x)
        ranks = np.empty(len(x))
        sorted_x = x[sorted_idx]
        i = 0
        while i < len(x):
            j = i
            while j < len(x) - 1 and sorted_x[j + 1] == sorted_x[i]:
                j += 1
            ranks[sorted_idx[i:j + 1]] = 0.5 * (i + j) + 1
            i = j + 1
        return ranks

    tx = midrank(pos)
    ty = midrank(neg)
    tz = midrank(np.concatenate([pos, neg]))
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    auc = tz[:m].sum() / (m * n) - (m + 1) / (2 * n)
    s01 = np.var(v01, ddof=1) / m
    s10 = np.var(v10, ddof=1) / n
    return auc, s01 + s10


def delong_test(y_true, proba_a, proba_b):
    """
    DeLong's test for comparing two correlated ROC-AUCs computed on the
    SAME held-out test set -- exactly what Reviewer 3 asked for
    ("DeLong test for ROC-AUC comparison").
    """
    auc_a, var_a = delong_roc_variance(y_true, np.asarray(proba_a))
    auc_b, var_b = delong_roc_variance(y_true, np.asarray(proba_b))
    var_diff = var_a + var_b
    if var_diff <= 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "z": np.nan, "p_value": 1.0}
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"auc_a": auc_a, "auc_b": auc_b, "z": z, "p_value": p}
