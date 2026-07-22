"""
SHAP-based explainability for ALL benchmarked models.

Addresses reviewer comments:
  - "SHAP explanations are presented only for MLP, although the paper
     claims explainability for all benchmarked models. Equivalent
     interpretability analyses should be provided for TabNet and TabR"
     (Reviewer 3)
  - "Quantitative evaluation of SHAP consistency, stability, and
     agreement across different deep learning models is needed" (Reviewer 3)

Design:
  - Tree models (RandomForest, XGBoost, LightGBM, CatBoost) -> shap.TreeExplainer
    (fast, exact).
  - LogisticRegression -> shap.LinearExplainer.
  - MLP / TabNet / TabR -> shap.KernelExplainer on a background sample
    (model-agnostic; slower, so we explain a subsample of the test set,
    which is standard practice and should be stated in the manuscript).
  - Stability across seeds is assessed with Spearman rank correlation and
    top-k Jaccard overlap of mean(|SHAP|) feature rankings.
"""
import os
import sys
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


TREE_MODEL_NAMES = {"RandomForest", "XGBoost", "LightGBM", "CatBoost"}
LINEAR_MODEL_NAMES = {"LogisticRegression"}


def get_explainer(model_name, model, background_X):
    if model_name in TREE_MODEL_NAMES:
        return shap.TreeExplainer(model)
    elif model_name in LINEAR_MODEL_NAMES:
        return shap.LinearExplainer(model, background_X)
    else:
        # MLP / TabNet / TabR: model-agnostic KernelExplainer on
        # predict_proba's positive-class column.
        predict_fn = lambda X: model.predict_proba(X)[:, 1]
        background = shap.sample(background_X, min(100, len(background_X)))
        return shap.KernelExplainer(predict_fn, background)


def compute_shap_values(model_name, model, X_explain, background_X, nsamples="auto"):
    explainer = get_explainer(model_name, model, background_X)
    if model_name in TREE_MODEL_NAMES:
        sv = explainer.shap_values(X_explain)
        if isinstance(sv, list):        # some tree explainers return [class0, class1]
            sv = sv[1]
        elif sv.ndim == 3:              # (n, features, classes)
            sv = sv[:, :, 1]
    elif model_name in LINEAR_MODEL_NAMES:
        sv = explainer.shap_values(X_explain)
    else:
        sv = explainer.shap_values(X_explain, nsamples=nsamples)
    return np.asarray(sv)


def mean_abs_importance(shap_values, feature_names) -> pd.Series:
    return pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)


def shap_stability_across_seeds(importances_by_seed: dict, top_k=5) -> dict:
    """
    importances_by_seed: {seed: pd.Series(feature -> mean|SHAP|)}
    Returns Spearman rank correlation (avg pairwise) and top-k Jaccard
    overlap across seeds -- the quantitative stability metrics Reviewer 3
    asked for, instead of a purely descriptive SHAP discussion.
    """
    seeds = list(importances_by_seed.keys())
    features = importances_by_seed[seeds[0]].index

    rank_matrix = pd.DataFrame({
        s: importances_by_seed[s].reindex(features).rank(ascending=False) for s in seeds
    })

    spearman_vals = []
    jaccard_vals = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            rho, _ = spearmanr(rank_matrix[seeds[i]], rank_matrix[seeds[j]])
            spearman_vals.append(rho)

            top_i = set(importances_by_seed[seeds[i]].nlargest(top_k).index)
            top_j = set(importances_by_seed[seeds[j]].nlargest(top_k).index)
            jaccard_vals.append(len(top_i & top_j) / len(top_i | top_j))

    std_importance = pd.DataFrame(importances_by_seed).std(axis=1).mean()

    return {
        "mean_spearman_rank_corr": float(np.mean(spearman_vals)),
        "mean_top{}_jaccard".format(top_k): float(np.mean(jaccard_vals)),
        "mean_std_of_mean_abs_shap": float(std_importance),
    }
