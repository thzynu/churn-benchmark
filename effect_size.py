"""
Effect-size analysis to complement the p-value-only statistical testing.

Addresses reviewer/advisor comment:
  "Statistical Analysis ... has Friedman, Wilcoxon, Holm-Bonferroni, but
   not Effect Size, Kendall's W, Cliff's Delta." Statistically significant
   differences are not automatically practically meaningful; effect sizes
   quantify the magnitude, not just the existence, of a difference.

Run AFTER main.py (reads outputs/tables/table_baseline_cv_raw.csv and
table_deep_cv_raw.csv):

    python effect_size.py

Writes:
    outputs/tables/table_cliffs_delta.csv   (pairwise, matches Wilcoxon pairs)
    outputs/tables/table_kendalls_w.csv     (single value, overall agreement)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cliff's Delta: proportion of pairs where a > b minus proportion where
    a < b, ranging from -1 to 1. Common interpretation thresholds
    (Romano et al. 2006): |d| < 0.147 negligible, < 0.33 small,
    < 0.474 medium, else large.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    n_gt = np.sum(a[:, None] > b[None, :])
    n_lt = np.sum(a[:, None] < b[None, :])
    return (n_gt - n_lt) / (len(a) * len(b))


def interpret_cliffs_delta(d: float) -> str:
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    elif ad < 0.33:
        return "small"
    elif ad < 0.474:
        return "medium"
    return "large"


def kendalls_w(rank_matrix: pd.DataFrame) -> float:
    """
    Kendall's W (coefficient of concordance) across raters (here: CV folds)
    ranking multiple models. 0 = no agreement, 1 = perfect agreement on
    which model is best/worst.
    rank_matrix: rows = folds/seeds, columns = models, values = ranks.
    """
    n, k = rank_matrix.shape  # n = number of folds, k = number of models
    rank_sums = rank_matrix.sum(axis=0).values
    mean_rank_sum = rank_sums.mean()
    ss = np.sum((rank_sums - mean_rank_sum) ** 2)
    w = 12 * ss / (n ** 2 * (k ** 3 - k))
    return float(w)


def interpret_kendalls_w(w: float) -> str:
    if w < 0.1:
        return "very weak agreement"
    elif w < 0.3:
        return "weak agreement"
    elif w < 0.5:
        return "moderate agreement"
    elif w < 0.7:
        return "strong agreement"
    return "very strong agreement"


def main():
    frames = []
    for f in ["table_baseline_cv_raw.csv", "table_deep_cv_raw.csv"]:
        path = os.path.join(config.TABLES_DIR, f)
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        print("[ERROR] No CV raw result files found. Run main.py first.")
        return
    df = pd.concat(frames, ignore_index=True)

    metric = "ROC_AUC"
    pivot = df.pivot_table(index=["seed", "fold"], columns="model", values=metric).dropna()
    models = list(pivot.columns)

    # --- Cliff's Delta for every pair (mirrors the Wilcoxon comparison table) ---
    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            d = cliffs_delta(pivot[models[i]].values, pivot[models[j]].values)
            rows.append({
                "comparison": f"{models[i]} vs {models[j]}",
                "cliffs_delta": round(d, 4),
                "magnitude": interpret_cliffs_delta(d),
            })
    delta_df = pd.DataFrame(rows).sort_values("cliffs_delta", key=abs, ascending=False)
    delta_df.to_csv(os.path.join(config.TABLES_DIR, "table_cliffs_delta.csv"), index=False)
    print("=== Cliff's Delta (effect size, pairwise ROC-AUC) ===")
    print(delta_df.to_string(index=False))

    # --- Kendall's W across all 8 models (overall ranking agreement) ---
    rank_matrix = pivot.rank(axis=1, ascending=False)  # rank 1 = best (highest ROC-AUC)
    w = kendalls_w(rank_matrix)
    w_df = pd.DataFrame([{
        "metric": metric,
        "kendalls_w": round(w, 4),
        "interpretation": interpret_kendalls_w(w),
        "n_folds": len(rank_matrix),
        "n_models": len(models),
    }])
    w_df.to_csv(os.path.join(config.TABLES_DIR, "table_kendalls_w.csv"), index=False)
    print(f"\n=== Kendall's W (overall model-ranking agreement across {len(rank_matrix)} CV folds) ===")
    print(w_df.to_string(index=False))

    print("\nSaved: table_cliffs_delta.csv, table_kendalls_w.csv")


if __name__ == "__main__":
    main()
