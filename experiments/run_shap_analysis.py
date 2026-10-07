"""SHAP feature importance analysis for XGBoost (TreeSHAP).

TreeSHAP provides exact Shapley values for tree-based models in polynomial
time. This script trains XGBoost with the same hyperparameters as
run_baselines.py, computes SHAP values on the test set, and produces:

  - Summary beeswarm plot (experiments/figures/shap_summary_plot.png)
  - Bar plot of mean |SHAP| per feature (experiments/figures/shap_bar_plot.png)
  - Feature importance CSV (experiments/results/shap/xgboost_feature_importance.csv)

Run:
    python -m experiments.run_shap_analysis
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # noqa: E402 — must be set before shap imports matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pathlib import Path

from experiments.run_baselines import (
    _build_split,
    _feature_columns,
    load_train_test,
    prepare_xy,
    DEFAULT_DATA_DIR,
)
from src.models.xgboost_model import XGBoostPredictor


FIGURES_DIR = Path("experiments/figures")
SHAP_RESULTS_DIR = Path("experiments/results/shap")


def main() -> int:
    """Run TreeSHAP analysis on XGBoost and save outputs."""
    # --- Load data (same pipeline as run_baselines) ---
    train_raw, test_raw = load_train_test(DEFAULT_DATA_DIR)
    train = _build_split(train_raw)
    test = _build_split(test_raw)

    feature_cols = _feature_columns(train)
    X_train, y_train = prepare_xy(train, feature_cols)
    X_test, y_test = prepare_xy(test, feature_cols)

    n_features = len(feature_cols)
    print(f"Loaded {len(train)} train rows, {len(test)} test rows, {n_features} features.")

    # --- Train XGBoost with same hyperparams as run_baselines ---
    model = XGBoostPredictor(n_estimators=200, max_depth=4, learning_rate=0.05)
    model.fit(X_train, y_train)
    print("XGBoost trained.")

    # --- Compute TreeSHAP values ---
    explainer = shap.TreeExplainer(model._model)
    shap_values = explainer.shap_values(X_test.values)
    print(f"SHAP values computed: shape {shap_values.shape}")

    # --- Create output directories ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Summary beeswarm plot ---
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.tight_layout()
    summary_path = FIGURES_DIR / "shap_summary_plot.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {summary_path}")

    # --- Bar plot (mean |SHAP| per feature) ---
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    bar_path = FIGURES_DIR / "shap_bar_plot.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {bar_path}")

    # --- Feature importance CSV ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature_name": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    csv_path = SHAP_RESULTS_DIR / "xgboost_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # --- Print top 10 features ---
    print("\n" + "=" * 60)
    print("Top 10 Features by Mean |SHAP| Value")
    print("=" * 60)
    for i, row in importance_df.head(10).iterrows():
        print(f"  {i + 1:>2}. {row['feature_name']:<40s} {row['mean_abs_shap']:.6f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
