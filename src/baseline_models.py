"""
Classical machine-learning baselines, added per reviewer requirement:
"Model comparison is incomplete because strong tabular-learning baselines
(e.g., XGBoost, LightGBM, CatBoost, Random Forest) are omitted" (Reviewer 3).

Each model is wrapped in an imblearn Pipeline: preprocessor -> [SMOTE] -> model,
so that SMOTE (when used) is only ever applied to the training fold of a
cross-validation split, never to validation/test data. This directly answers
Reviewer 3 & 5's data-leakage concerns.
"""
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_pipeline import build_preprocessor


# ---------------------------------------------------------------------------
# Model registry: name -> (estimator, hyperparameter search space)
# ---------------------------------------------------------------------------
def get_model_registry(random_state: int = 42):
    registry = {
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, random_state=random_state),
            {
                "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [4, 6, 8, 12, None],
                "model__min_samples_leaf": [1, 2, 5, 10],
            },
        ),
        "XGBoost": (
            xgb.XGBClassifier(
                random_state=random_state, eval_metric="logloss",
                use_label_encoder=False, n_jobs=-1,
            ),
            {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [3, 4, 5, 6],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__subsample": [0.7, 0.8, 1.0],
                "model__colsample_bytree": [0.7, 0.8, 1.0],
            },
        ),
        "LightGBM": (
            lgb.LGBMClassifier(random_state=random_state, verbosity=-1, n_jobs=-1),
            {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [-1, 4, 6, 8],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__num_leaves": [15, 31, 63],
            },
        ),
        "CatBoost": (
            CatBoostClassifier(random_state=random_state, verbose=0),
            {
                "model__depth": [4, 6, 8],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__iterations": [200, 300, 500],
            },
        ),
    }
    return registry


def build_pipeline(estimator, imbalance_strategy: str, random_state: int = 42):
    """
    imbalance_strategy in {"none", "class_weight", "smote"}.
    "class_weight" only takes effect for estimators that expose the
    class_weight / scale_pos_weight parameter; for tree-boosting libraries
    we set it explicitly below in tune_and_evaluate().
    """
    preprocessor = build_preprocessor()
    steps = [("preprocessor", preprocessor)]
    if imbalance_strategy == "smote":
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("model", estimator))
    return ImbPipeline(steps)


def tune_model(model_name, X_train, y_train, imbalance_strategy="smote",
               random_state=42, n_iter=None, cv_splits=3):
    """
    Runs RandomizedSearchCV *inside* an imblearn Pipeline so that SMOTE is
    refit on every inner training fold only (never touching the inner
    validation fold) -- this is the leakage-safe hyperparameter tuning
    Reviewer 3 asked for ("no hyperparameter optimization details are
    provided, making the experiments irreproducible").
    """
    n_iter = n_iter or config.BASELINE_SEARCH_ITER
    registry = get_model_registry(random_state)
    estimator, param_dist = registry[model_name]

    if imbalance_strategy == "class_weight":
        if hasattr(estimator, "class_weight"):
            estimator.set_params(class_weight="balanced")
        elif model_name == "XGBoost":
            neg, pos = np.bincount(y_train)
            estimator.set_params(scale_pos_weight=neg / pos)
        elif model_name in ("LightGBM",):
            estimator.set_params(class_weight="balanced")
        elif model_name == "CatBoost":
            neg, pos = np.bincount(y_train)
            estimator.set_params(class_weights=[1.0, neg / pos])

    pipeline = build_pipeline(estimator, imbalance_strategy, random_state)

    inner_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=n_iter,
        scoring="roc_auc", cv=inner_cv, random_state=random_state,
        n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


if __name__ == "__main__":
    from src.data_pipeline import load_raw_data, get_feature_target
    from sklearn.model_selection import train_test_split

    df = load_raw_data()
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=42
    )

    for name in ["LogisticRegression", "RandomForest"]:
        best_model, best_params = tune_model(
            name, X_train, y_train, imbalance_strategy="smote", n_iter=5, cv_splits=3
        )
        print(name, "->", best_params)
