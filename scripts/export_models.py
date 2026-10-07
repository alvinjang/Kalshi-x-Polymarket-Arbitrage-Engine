"""Train LR + XGBoost on train.parquet and export as pickle files.

Produces:
    models/deployed/linear_regression.pkl
    models/deployed/xgboost.pkl
    models/deployed/feature_columns.json

These pickled models are loaded by the adaptive trading system on the
Oracle VM (1 GB RAM) to avoid retraining every cycle.

Usage:
    python scripts/export_models.py
    python -m src.live.trading_cycle --export-models   (same logic)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.run_baselines import (
    _build_split,
    _feature_columns,
    load_train_test,
    prepare_xy,
)
from src.models.base import BasePredictor
from src.models.linear_regression import LinearRegressionPredictor
from src.models.xgboost_model import XGBoostPredictor

MODEL_DIR = Path("models/deployed")
DATA_DIR = Path("data/processed")


def export_models(
    data_dir: Path = DATA_DIR,
    model_dir: Path = MODEL_DIR,
) -> None:
    """Train LR + XGBoost and save as pickles."""
    print(f"Loading training data from {data_dir} ...")
    train_raw, _test_raw = load_train_test(data_dir)
    train = _build_split(train_raw)

    feature_cols = _feature_columns(train)
    X_train, y_train = prepare_xy(train, feature_cols)
    print(f"  {len(train)} rows, {len(feature_cols)} features")

    model_dir.mkdir(parents=True, exist_ok=True)

    # --- Linear Regression ---
    lr = LinearRegressionPredictor()
    lr.fit(X_train, y_train)
    lr_path = model_dir / "linear_regression.pkl"
    lr.save(lr_path)
    print(f"  Saved {lr.name} -> {lr_path}")

    # --- XGBoost (same hyperparams as run_baselines.py) ---
    xgb = XGBoostPredictor(n_estimators=200, max_depth=4, learning_rate=0.05)
    xgb.fit(X_train, y_train)
    xgb_path = model_dir / "xgboost.pkl"
    xgb.save(xgb_path)
    print(f"  Saved {xgb.name} -> {xgb_path}")

    # --- Feature columns ---
    fc_path = model_dir / "feature_columns.json"
    with open(fc_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  Saved {len(feature_cols)} feature columns -> {fc_path}")

    # --- Verification ---
    print("\nVerification:")
    lr_loaded = BasePredictor.load(lr_path)
    xgb_loaded = BasePredictor.load(xgb_path)

    # Quick prediction test on 5 rows
    X_test_batch = X_train.iloc[:5]
    lr_preds = lr_loaded.predict(X_test_batch)
    xgb_preds = xgb_loaded.predict(X_test_batch)

    assert lr_preds.shape == (5,), f"LR shape mismatch: {lr_preds.shape}"
    assert xgb_preds.shape == (5,), f"XGB shape mismatch: {xgb_preds.shape}"
    assert np.all(np.isfinite(lr_preds)), "LR produced non-finite predictions"
    assert np.all(np.isfinite(xgb_preds)), "XGB produced non-finite predictions"

    print(f"  {lr_loaded.name}: predictions OK (shape={lr_preds.shape})")
    print(f"  {xgb_loaded.name}: predictions OK (shape={xgb_preds.shape})")
    print("\nAll models exported successfully.")


if __name__ == "__main__":
    export_models()
