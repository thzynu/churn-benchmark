"""
Extended SHAP stability analysis: adds Kendall's Tau (Reviewer 3 named this
metric explicitly) alongside the Spearman/Jaccard metrics already computed
in main.py, using the shap_importance_*.csv files main.py already produced
(no retraining needed).

Addresses reviewer comment:
  "Reviewer wants ... Spearman rank correlation, Top-k overlap or Jaccard
   similarity, Kendall's tau between models, std of mean |SHAP| across
   folds." (Reviewer 3 / advisor notes)

Run AFTER main.py:
    python shap_stability_extra.py

Writes:
    outputs/tables/table_shap_stability_full.csv   (pairwise Spearman + Kendall + Jaccard)
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

ALL_MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost",
              "MLP", "TabNet", "TabR"]


def main(top_k=3):
    importances = {}
    for model in ALL_MODELS:
        path = os.path.join(config.TABLES_DIR, f"shap_importance_{model}.csv")
        if os.path.exists(path):
            importances[model] = pd.read_csv(path, index_col=0).iloc[:, 0]

    if len(importances) < 2:
        print("[ERROR] Need at least 2 shap_importance_*.csv files. Run main.py first.")
        return

    models = list(importances.keys())
    features = importances[models[0]].index
    rank_df = pd.DataFrame({m: importances[m].reindex(features).rank(ascending=False) for m in models})

    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            rho, _ = spearmanr(rank_df[models[i]], rank_df[models[j]])
            tau, _ = kendalltau(rank_df[models[i]], rank_df[models[j]])
            top_i = set(importances[models[i]].nlargest(top_k).index)
            top_j = set(importances[models[j]].nlargest(top_k).index)
            jaccard = len(top_i & top_j) / len(top_i | top_j)
            rows.append({
                "comparison": f"{models[i]} vs {models[j]}",
                "spearman_rho": round(rho, 4),
                "kendalls_tau": round(tau, 4),
                f"top{top_k}_jaccard": round(jaccard, 4),
            })
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(config.TABLES_DIR, "table_shap_stability_full.csv"), index=False)

    print(f"=== Pairwise SHAP feature-ranking agreement across {len(models)} models ===")
    print(result.to_string(index=False))
    print(f"\nMean Spearman rho: {result['spearman_rho'].mean():.4f}")
    print(f"Mean Kendall's tau: {result['kendalls_tau'].mean():.4f}")
    print(f"Mean top-{top_k} Jaccard: {result[f'top{top_k}_jaccard'].mean():.4f}")

    # Std of mean |SHAP| across models, per feature (how much attribution varies by architecture)
    imp_matrix = pd.DataFrame(importances).reindex(features)
    std_per_feature = imp_matrix.std(axis=1).sort_values(ascending=False)
    print("\n=== Std of mean |SHAP| across models (per feature) ===")
    print(std_per_feature.to_string())

    print("\nSaved: table_shap_stability_full.csv")


if __name__ == "__main__":
    main()
