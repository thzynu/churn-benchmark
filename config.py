"""
Central configuration for the churn-prediction benchmarking pipeline.
Edit values here to change behaviour of the whole project without touching
the rest of the code.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "data", "churn.csv")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Multiple seeds used for repeated cross-validation (reviewer requirement:
# repeated stratified k-fold rather than a single 80/20 split).
SEEDS = [42, 52, 62, 72, 82]

# Set to a shorter list (e.g. [42]) for a fast smoke-test run before
# committing to the full repeated-CV experiment.
# SEEDS = [42]

N_SPLITS = 5          # folds per repeat
TEST_SIZE = 0.2        # held-out test size (stratified, untouched by SMOTE/tuning)

# ---------------------------------------------------------------------------
# Target / columns
# ---------------------------------------------------------------------------
TARGET_COL = "Exited"

# Columns that carry no predictive/business meaning and must be dropped
# (leakage audit: pure identifiers).
ID_COLS = ["RowNumber", "CustomerId", "Surname"]

CATEGORICAL_COLS = ["Geography", "Gender"]
NUMERIC_COLS = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "EstimatedSalary",
]
BINARY_COLS = ["HasCrCard", "IsActiveMember"]

# ---------------------------------------------------------------------------
# Imbalance handling strategies to compare (Reviewer 3 requirement:
# do not rely on a single SMOTE configuration)
# ---------------------------------------------------------------------------
IMBALANCE_STRATEGIES = ["none", "class_weight", "smote"]

# ---------------------------------------------------------------------------
# Model hyperparameters (kept modest so the whole pipeline runs in a
# reasonable time on a laptop CPU; increase for the final paper-quality run)
# ---------------------------------------------------------------------------
# Tuned via Optuna on seed=42 with an inner train/validation split
# (see tune_deep_models.py and outputs/tables/table_hyperparameter_best.csv).
# max_epochs/patience kept at the full search-time budget (not the reduced
# budget used during the search itself) for the final reported runs.
MLP_PARAMS = dict(
    hidden_layers=[32, 16],
    dropout=0.1097,
    batch_norm=True,
    learning_rate=0.001266,
    batch_size=32,
    max_epochs=100,
    patience=10,
)

TABNET_PARAMS = dict(
    n_d=16,
    n_a=16,
    n_steps=5,
    gamma=1.156,
    lambda_sparse=0.0002051,
    max_epochs=100,
    patience=15,
    batch_size=256,
)

TABR_PARAMS = dict(
    embed_dim=32,
    n_neighbors=16,
    dropout=0.3909,
    learning_rate=0.001904,
    batch_size=64,
    max_epochs=100,
    patience=10,
)

# Optuna-style search budget for classical baselines (RandomizedSearchCV
# n_iter, used instead of a full Optuna dependency to keep requirements
# lighter; swap in Optuna easily if preferred).
BASELINE_SEARCH_ITER = 25
