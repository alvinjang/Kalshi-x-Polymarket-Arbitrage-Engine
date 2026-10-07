"""Experiment 3: Spread threshold ablation.

Re-evaluates ALL 8 trained models at trading thresholds {0.0, 0.02, 0.05, 0.10}
WITHOUT retraining. Each model is trained once, raw predictions are obtained,
then ``simulate_profit`` is called 4 times (once per threshold).

This tests how the minimum predicted-spread threshold affects trading frequency
and profitability. At threshold=0.0 all predictions trigger trades; at
threshold=0.10 only large predicted moves trigger trades.

Output:
  - 32 JSON files in experiments/results/ablation_threshold/
  - experiments/figures/experiment3_threshold_heatmap.png
  - experiments/figures/experiment3_threshold_pnl.png
  - Summary table printed to stdout

Run:
    python -m experiments.run_experiment3_threshold
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.run_baselines import (
    NON_FEATURE_COLUMNS,
    _MODEL_ORDER,
    _build_split,
    _feature_columns,
    load_train_test,
    prepare_xy,
    prepare_xy_for_seq,
)
from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.profit_sim import simulate_profit
from src.evaluation.results_store import save_results
from src.models.autoencoder import AnomalyDetectorAutoencoder
from src.models.gru import GRUPredictor
from src.models.linear_regression import LinearRegressionPredictor
from src.models.lstm import LSTMPredictor
from src.models.naive import NaivePredictor
from src.models.ppo_filtered import PPOFilteredPredictor
from src.models.ppo_raw import PPORawPredictor
from src.models.volume import VolumePredictor
from src.models.xgboost_model import XGBoostPredictor
from src.utils.seed import set_all_seeds

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

THRESHOLDS = [0.0, 0.02, 0.05, 0.10]
RESULTS_DIR = Path("experiments/results/ablation_threshold")
FIGURES_DIR = Path("experiments/figures")
DATA_DIR = Path("data/processed")

# Apple Silicon thread safety handled by set_all_seeds() in main()


def _slug(model_name: str) -> str:
    """Filesystem-safe slug for a model name."""
    import re
    slug = model_name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_") or "unnamed"


def _save_threshold_result(
    model_name: str,
    threshold: float,
    reg_metrics: dict,
    trading_metrics: dict,
    n_train: int,
    n_test: int,
    n_features: int,
) -> Path:
    """Save a single (model, threshold) result JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pnl_series = trading_metrics.pop("pnl_series", [])
    metrics = {**reg_metrics, **trading_metrics}

    payload = {
        "model_name": model_name,
        "threshold": threshold,
        "metrics": metrics,
        "n_train_rows": n_train,
        "n_test_rows": n_test,
        "n_features": n_features,
        "pnl_series": pnl_series,
    }

    filename = f"{_slug(model_name)}_threshold_{threshold:.2f}.json"
    path = RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def train_and_predict_all(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, np.ndarray]:
    """Train all 8 models once, return {model_name: predictions} dict."""
    X_train, y_train = prepare_xy(train, feature_cols)
    X_test, y_test = prepare_xy(test, feature_cols)
    X_train_seq, y_train_seq = prepare_xy_for_seq(train, feature_cols)
    X_test_seq, y_test_seq = prepare_xy_for_seq(test, feature_cols)

    n_features = len(feature_cols)
    predictions: dict[str, np.ndarray] = {}

    # ---- Tier 1: Flat models ----
    tier1_models = [
        NaivePredictor(),
        VolumePredictor(),
        LinearRegressionPredictor(),
        XGBoostPredictor(n_estimators=200, max_depth=4, learning_rate=0.05),
    ]
    for model in tier1_models:
        print(f"  Training {model.name}...", end=" ", flush=True)
        t0 = time.time()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        predictions[model.name] = preds

    # ---- Tier 2: Sequence models (single seed=42 for ablation) ----
    tier2_classes = [
        (GRUPredictor, "GRU"),
        (LSTMPredictor, "LSTM"),
    ]
    for cls, display_name in tier2_classes:
        print(f"  Training {display_name}...", end=" ", flush=True)
        t0 = time.time()
        model = cls(random_state=42)
        model.fit(X_train_seq, y_train_seq)
        preds = model.predict(X_test_seq)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")
        predictions[display_name] = preds

    # ---- Tier 3: RL models (single seed=42 for ablation) ----

    # PPO-Raw
    print("  Training PPO-Raw...", end=" ", flush=True)
    t0 = time.time()
    ppo_raw = PPORawPredictor(random_state=42, total_timesteps=100_000)
    ppo_raw.fit(X_train_seq, y_train_seq)
    preds = ppo_raw.predict(X_test_seq)
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")
    predictions["PPO-Raw"] = preds

    # Autoencoder + PPO-Filtered
    print("  Training Autoencoder for PPO-Filtered...", end=" ", flush=True)
    t0 = time.time()
    autoencoder = AnomalyDetectorAutoencoder(
        input_dim=n_features, random_state=42,
    )
    autoencoder.fit(train[feature_cols], feature_cols)
    flagging_rate = float(autoencoder.flag_anomalies(train[feature_cols]).mean())
    elapsed_ae = time.time() - t0
    print(f"done ({elapsed_ae:.1f}s, flagging rate={flagging_rate:.1%})")

    print("  Training PPO-Filtered...", end=" ", flush=True)
    t0 = time.time()
    ppo_filtered = PPOFilteredPredictor(
        anomaly_detector=autoencoder,
        random_state=42,
        total_timesteps=100_000,
    )
    ppo_filtered.fit(X_train_seq, y_train_seq)
    preds = ppo_filtered.predict(X_test_seq)
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")
    predictions["PPO-Filtered"] = preds

    return predictions


def evaluate_at_thresholds(
    predictions: dict[str, np.ndarray],
    y_test: np.ndarray,
    test_timestamps: np.ndarray,
    n_train: int,
    n_test: int,
    n_features: int,
) -> list[dict]:
    """Evaluate all models at all thresholds. Returns list of result rows."""
    all_results: list[dict] = []

    for model_name in _MODEL_ORDER:
        if model_name not in predictions:
            continue
        preds = predictions[model_name]
        reg_metrics = compute_regression_metrics(y_test, preds)

        for threshold in THRESHOLDS:
            trading = simulate_profit(
                preds, y_test,
                threshold=threshold,
                timestamps=test_timestamps,
            )

            # Save JSON
            _save_threshold_result(
                model_name=model_name,
                threshold=threshold,
                reg_metrics=reg_metrics.copy(),
                trading_metrics=trading.copy(),
                n_train=n_train,
                n_test=n_test,
                n_features=n_features,
            )

            all_results.append({
                "model_name": model_name,
                "threshold": threshold,
                **reg_metrics,
                **{k: v for k, v in trading.items() if k != "pnl_series"},
            })

            print(
                f"    {model_name} @ threshold={threshold:.2f}: "
                f"P&L={trading['total_pnl']:.4f}, "
                f"trades={trading['num_trades']}, "
                f"win_rate={trading['win_rate']:.4f}, "
                f"sharpe={trading['sharpe_ratio']:.4f}"
            )

    return all_results


def print_summary_table(results: list[dict]) -> None:
    """Print a formatted table of all 32 (model, threshold) combinations."""
    print("\n" + "=" * 100)
    print("EXPERIMENT 3: THRESHOLD ABLATION -- FULL RESULTS TABLE")
    print("=" * 100)

    model_width = max(len(r["model_name"]) for r in results)
    model_width = max(model_width, len("Model"))

    header = (
        f"{'Model':<{model_width}} | {'Thresh':>6} | {'RMSE':>7} | {'MAE':>7} "
        f"| {'Dir Acc':>7} | {'P&L':>8} | {'Trades':>6} "
        f"| {'Win Rate':>8} | {'Sharpe':>7}"
    )
    sep = (
        f"{'-' * model_width}-+-{'-' * 6}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}"
        f"-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 7}"
    )

    print()
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r['model_name']:<{model_width}} | "
            f"{r['threshold']:>6.2f} | "
            f"{r['rmse']:>7.4f} | "
            f"{r['mae']:>7.4f} | "
            f"{r['directional_accuracy']:>7.4f} | "
            f"{r['total_pnl']:>8.4f} | "
            f"{r['num_trades']:>6d} | "
            f"{r['win_rate']:>8.4f} | "
            f"{r['sharpe_ratio']:>7.4f}"
        )


def print_best_thresholds(results: list[dict]) -> None:
    """Print which threshold maximizes P&L and Sharpe for each model."""
    print("\n" + "=" * 80)
    print("BEST THRESHOLD PER MODEL")
    print("=" * 80)

    model_width = max(len(r["model_name"]) for r in results)
    model_width = max(model_width, len("Model"))

    header = (
        f"{'Model':<{model_width}} | "
        f"{'Best P&L Thresh':>15} | {'P&L':>8} | "
        f"{'Best Sharpe Thresh':>18} | {'Sharpe':>7}"
    )
    sep = (
        f"{'-' * model_width}-+-{'-' * 15}-+-{'-' * 8}-+-{'-' * 18}-+-{'-' * 7}"
    )

    print()
    print(header)
    print(sep)

    # Group by model
    by_model: dict[str, list[dict]] = {}
    for r in results:
        by_model.setdefault(r["model_name"], []).append(r)

    for model_name in _MODEL_ORDER:
        if model_name not in by_model:
            continue
        rows = by_model[model_name]

        best_pnl = max(rows, key=lambda r: r["total_pnl"])
        best_sharpe = max(rows, key=lambda r: r["sharpe_ratio"])

        print(
            f"{model_name:<{model_width}} | "
            f"{best_pnl['threshold']:>15.2f} | "
            f"{best_pnl['total_pnl']:>8.4f} | "
            f"{best_sharpe['threshold']:>18.2f} | "
            f"{best_sharpe['sharpe_ratio']:>7.4f}"
        )


def plot_heatmap(results: list[dict]) -> None:
    """Create a heatmap of total P&L by (model, threshold)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Build the matrix: rows=models, cols=thresholds
    model_names = [m for m in _MODEL_ORDER if any(r["model_name"] == m for r in results)]
    n_models = len(model_names)
    n_thresholds = len(THRESHOLDS)

    pnl_matrix = np.zeros((n_models, n_thresholds))
    for r in results:
        if r["model_name"] in model_names:
            row_idx = model_names.index(r["model_name"])
            col_idx = THRESHOLDS.index(r["threshold"])
            pnl_matrix[row_idx, col_idx] = r["total_pnl"]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Diverging colormap: red=negative, yellow=neutral, green=positive
    vmax = max(abs(pnl_matrix.min()), abs(pnl_matrix.max())) or 1.0
    im = ax.imshow(
        pnl_matrix,
        cmap="RdYlGn",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
    )

    # Annotate cells
    for i in range(n_models):
        for j in range(n_thresholds):
            val = pnl_matrix[i, j]
            color = "black" if abs(val) < vmax * 0.6 else "white"
            ax.text(
                j, i, f"{val:.3f}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color=color,
            )

    ax.set_xticks(range(n_thresholds))
    ax.set_xticklabels([f"{t:.2f}" for t in THRESHOLDS])
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Trading Threshold", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title("Experiment 3: Total P&L by Model and Threshold", fontsize=14, pad=15)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Total P&L", fontsize=11)

    plt.tight_layout()
    path = FIGURES_DIR / "experiment3_threshold_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHeatmap saved to {path}")


def plot_grouped_bar(results: list[dict]) -> None:
    """Create a grouped bar chart of P&L by threshold for each model."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    model_names = [m for m in _MODEL_ORDER if any(r["model_name"] == m for r in results)]
    n_models = len(model_names)
    n_thresholds = len(THRESHOLDS)

    # Build P&L values per (model, threshold)
    pnl_by_model_thresh: dict[str, list[float]] = {m: [] for m in model_names}
    for model_name in model_names:
        for threshold in THRESHOLDS:
            match = [r for r in results if r["model_name"] == model_name and r["threshold"] == threshold]
            pnl_by_model_thresh[model_name].append(match[0]["total_pnl"] if match else 0.0)

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(n_models)
    bar_width = 0.18
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]

    for i, threshold in enumerate(THRESHOLDS):
        values = [pnl_by_model_thresh[m][i] for m in model_names]
        offset = (i - (n_thresholds - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width,
            label=f"threshold={threshold:.2f}",
            color=colors[i], edgecolor="white", linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Total P&L", fontsize=12)
    ax.set_title("Experiment 3: P&L by Threshold", fontsize=14, pad=15)
    ax.legend(fontsize=10, loc="best")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "experiment3_threshold_pnl.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart saved to {path}")


def main() -> int:
    """Run Experiment 3: threshold ablation across all 8 models."""
    set_all_seeds(42)
    print("=" * 70)
    print("EXPERIMENT 3: SPREAD THRESHOLD ABLATION")
    print(f"Thresholds: {THRESHOLDS}")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    try:
        train_raw, test_raw = load_train_test(DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    train = _build_split(train_raw)
    test = _build_split(test_raw)
    feature_cols = _feature_columns(train)
    n_train = len(train)
    n_test = len(test)
    n_features = len(feature_cols)

    print(f"  Train: {n_train} rows, Test: {n_test} rows, Features: {n_features}")

    _, y_test = prepare_xy(test, feature_cols)
    test_timestamps = test["timestamp"].to_numpy()

    # Train all models once
    print("\n--- Training all 8 models (single seed=42 for Tier 2/3) ---\n")
    t_start = time.time()
    predictions = train_and_predict_all(train, test, feature_cols)
    t_train = time.time() - t_start
    print(f"\nAll models trained in {t_train:.1f}s")

    # Evaluate at all thresholds
    print("\n--- Evaluating at all thresholds ---\n")
    all_results = evaluate_at_thresholds(
        predictions=predictions,
        y_test=y_test,
        test_timestamps=test_timestamps,
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
    )

    # Count saved files
    json_count = len(list(RESULTS_DIR.glob("*.json")))
    print(f"\nSaved {json_count} JSON result files to {RESULTS_DIR}/")

    # Print tables
    print_summary_table(all_results)
    print_best_thresholds(all_results)

    # Generate figures
    print("\n--- Generating figures ---\n")
    plot_heatmap(all_results)
    plot_grouped_bar(all_results)

    print("\nExperiment 3 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
