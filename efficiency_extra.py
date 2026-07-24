"""
Extended computational efficiency: adds peak RAM usage and serialized
model size (on disk) to the training-time/inference-latency table
generate_figures.py already produces.

Addresses reviewer comment:
  "Computational Efficiency ... has Training Time, Latency, Parameter
   Count, but not Peak RAM, Memory, Model Size."

Run:
    python efficiency_extra.py

Writes:
    outputs/tables/table_computational_efficiency_full.csv
"""
import os
import sys
import pickle
import tempfile
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from src.data_pipeline import load_raw_data, get_feature_target, build_preprocessor
from src.baseline_models import get_model_registry
from imblearn.over_sampling import SMOTE

BASELINE_MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "CatBoost"]
DEEP_MODELS = ["MLP", "TabNet", "TabR"]


def model_size_bytes(obj) -> int:
    with tempfile.NamedTemporaryFile(delete=False) as f:
        pickle.dump(obj, f)
        path = f.name
    size = os.path.getsize(path)
    os.remove(path)
    return size


def measure_baseline(model_name, X_train, y_train, random_state=42):
    pre = build_preprocessor()
    Xtr = pre.fit_transform(X_train)
    sm = SMOTE(random_state=random_state)
    Xtr_res, ytr_res = sm.fit_resample(Xtr, y_train)

    registry = get_model_registry(random_state)
    estimator, _ = registry[model_name]

    tracemalloc.start()
    estimator.fit(Xtr_res, ytr_res)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    size_bytes = model_size_bytes(estimator)
    return {"peak_ram_mb": peak / 1e6, "model_size_kb": size_bytes / 1e3}


def measure_deep(model_name, X_train, y_train, random_state=42):
    from src.deep_models import get_deep_model
    pre = build_preprocessor()
    Xtr_raw, Xval_raw, ytr, yval = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=random_state
    )
    Xtr = pre.fit_transform(Xtr_raw)
    Xval = pre.transform(Xval_raw)
    sm = SMOTE(random_state=random_state)
    Xtr_res, ytr_res = sm.fit_resample(Xtr, ytr)

    model = get_deep_model(model_name, input_dim=Xtr_res.shape[1], random_state=random_state)

    tracemalloc.start()
    model.fit(Xtr_res, ytr_res, Xval, yval)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # torch state_dict pickled as a proxy for on-disk model size
    try:
        size_bytes = model_size_bytes(model.model.state_dict())
    except AttributeError:
        size_bytes = model_size_bytes(model.model.network.state_dict())
    return {"peak_ram_mb": peak / 1e6, "model_size_kb": size_bytes / 1e3}


def main(random_state=42):
    df = load_raw_data()
    X, y = get_feature_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=random_state
    )

    rows = []
    for model_name in BASELINE_MODELS:
        print(f"Measuring {model_name} ...")
        m = measure_baseline(model_name, X_train, y_train, random_state)
        m["model"] = model_name
        rows.append(m)

    try:
        for model_name in DEEP_MODELS:
            print(f"Measuring {model_name} ...")
            m = measure_deep(model_name, X_train, y_train, random_state)
            m["model"] = model_name
            rows.append(m)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"[SKIPPED deep models] torch/pytorch-tabnet not available ({e})")

    result = pd.DataFrame(rows).set_index("model")

    # merge with existing training-time/inference-latency table if present
    eff_path = os.path.join(config.TABLES_DIR, "table_computational_efficiency.csv")
    if os.path.exists(eff_path):
        existing = pd.read_csv(eff_path).set_index("model")
        result = existing.join(result, how="outer")

    result.to_csv(os.path.join(config.TABLES_DIR, "table_computational_efficiency_full.csv"))
    print("\n=== Computational efficiency (extended: + RAM, model size) ===")
    print(result.to_string())
    print("\nSaved: table_computational_efficiency_full.csv")


if __name__ == "__main__":
    main()
