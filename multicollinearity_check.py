"""
Multicollinearity assessment: correlation matrix + Variance Inflation Factor
(VIF) for numeric features, and Cramer's V for categorical associations.

Addresses reviewer comment:
  "The manuscript does not investigate feature redundancy or
   multicollinearity, which may substantially influence SHAP importance
   values and feature attribution reliability." (Reviewer 3)

Run:
    python multicollinearity_check.py

Writes:
    outputs/tables/table_vif.csv
    outputs/tables/table_correlation_matrix.csv
    outputs/tables/table_cramers_v.csv
    outputs/figures/fig_correlation_heatmap.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_pipeline import load_raw_data, get_feature_target


def compute_vif(X_numeric: pd.DataFrame) -> pd.DataFrame:
    """VIF > 5 is commonly flagged as moderate concern, > 10 as severe."""
    X = X_numeric.copy()
    X.insert(0, "const", 1.0)
    rows = []
    for i, col in enumerate(X.columns):
        if col == "const":
            continue
        vif = variance_inflation_factor(X.values, i)
        rows.append({"Feature": col, "VIF": round(vif, 3)})
    df = pd.DataFrame(rows).sort_values("VIF", ascending=False)
    df["Concern"] = pd.cut(
        df["VIF"], bins=[-np.inf, 5, 10, np.inf],
        labels=["Low (<5)", "Moderate (5-10)", "Severe (>10)"],
    )
    return df


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V association measure for two categorical variables (0-1)."""
    confusion = pd.crosstab(x, y)
    chi2 = _chi2_stat(confusion)
    n = confusion.values.sum()
    r, k = confusion.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def _chi2_stat(confusion: pd.DataFrame) -> float:
    from scipy.stats import chi2_contingency
    chi2, _, _, _ = chi2_contingency(confusion)
    return chi2


def main():
    df = load_raw_data()
    X, y = get_feature_target(df)

    numeric_cols = config.NUMERIC_COLS + config.BINARY_COLS
    categorical_cols = config.CATEGORICAL_COLS

    # 1. Correlation matrix (numeric features)
    corr = X[numeric_cols].corr()
    corr.to_csv(os.path.join(config.TABLES_DIR, "table_correlation_matrix.csv"))

    # 2. VIF
    vif_df = compute_vif(X[numeric_cols].astype(float))
    vif_df.to_csv(os.path.join(config.TABLES_DIR, "table_vif.csv"), index=False)
    print("=== Variance Inflation Factor (numeric features) ===")
    print(vif_df.to_string(index=False))
    max_vif = vif_df["VIF"].max()
    print(f"\nMax VIF = {max_vif:.2f} -> "
          f"{'no severe multicollinearity detected' if max_vif < 5 else 'review flagged features'}")

    # 3. Cramer's V (categorical-categorical + categorical-numeric via binned)
    cramers_rows = []
    cat_all = categorical_cols + config.BINARY_COLS
    for i in range(len(cat_all)):
        for j in range(i + 1, len(cat_all)):
            v = cramers_v(X[cat_all[i]], X[cat_all[j]])
            cramers_rows.append({"Feature A": cat_all[i], "Feature B": cat_all[j], "Cramers_V": round(v, 4)})
    cramers_df = pd.DataFrame(cramers_rows).sort_values("Cramers_V", ascending=False)
    cramers_df.to_csv(os.path.join(config.TABLES_DIR, "table_cramers_v.csv"), index=False)
    print("\n=== Cramer's V (categorical associations) ===")
    print(cramers_df.to_string(index=False))

    # 4. Heatmap figure
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_yticklabels(numeric_cols, fontsize=8)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, ax=ax, label="Pearson correlation")
    ax.set_title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, "fig_correlation_heatmap.png"), dpi=150)
    plt.close()

    print("\nSaved: table_vif.csv, table_correlation_matrix.csv, table_cramers_v.csv, fig_correlation_heatmap.png")


if __name__ == "__main__":
    main()
