"""Experiment 2: Lookback window ablation for GRU and LSTM.

Retrains GRU and LSTM at lookback values {2, 6, 12, 18}, mapping to
{8h, 24h, 48h, 72h} at 4-hour bars.  The current default is lookback=6
(24h).  This tests whether longer temporal context improves spread
prediction on the 6.8k-row matched-pairs dataset.

Produces:
  - 8 result JSONs in experiments/results/ablation_lookback/
  - RMSE line plot in experiments/figures/experiment2_lookback_rmse.png
  - P&L line plot in experiments/figures/experiment2_lookback_pnl.png

Run:
    python -m experiments.run_experiment2_lookback
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib.pyplot as plt

from src.utils.seed import set_all_seeds

from experiments.run_baselines import (
    load_train_test,
    _build_split,
    _feature_columns,
    prepare_xy_for_seq,
    NON_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.evaluation.results_store import save_results
from src.models.gru import GRUPredictor
from src.models.lstm import LSTMPredictor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOOKBACK_VALUES = [2, 6, 12, 18]
LOOKBACK_HOURS = {2: "8h", 6: "24h", 12: "48h", 18: "72h"}

SEED = 42  # Single seed for ablation (main multi-seed results live in tier2/)
THRESHOLD = 0.02

DEFAULT_DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("experiments/results/ablation_lookback")
FIGURES_DIR = Path("experiments/figures")

MODEL_CLASSES = [
    ("GRU", GRUPredictor),
    ("LSTM", LSTMPredictor),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    set_all_seeds(42)
    # Load data
    try:
        train_raw, test_raw = load_train_test(DEFAULT_DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    train = _build_split(train_raw)
    test = _build_split(test_raw)

    if len(train) == 0 or len(test) == 0:
        print(
            f"Error: no usable rows after target computation. "
            f"train={len(train)} test={len(test)}",
            file=sys.stderr,
        )
        return 1

    feature_cols = _feature_columns(train)
    n_train = len(train)
    n_test = len(test)
    n_features = len(feature_cols)

    print(
        f"Loaded {n_train} train rows, {n_test} test rows, "
        f"{n_features} features."
    )

    # Timestamps for panel-aware Sharpe
    test_timestamps = test["timestamp"].to_numpy()

    # Prepare sequence data (includes group_id)
    X_train_seq, y_train = prepare_xy_for_seq(train, feature_cols)
    X_test_seq, y_test = prepare_xy_for_seq(test, feature_cols)

    # Results storage: {model_name: {lookback: {metric: value}}}
    results: dict[str, dict[int, dict]] = {}

    for model_label, model_cls in MODEL_CLASSES:
        results[model_label] = {}

        for lookback in LOOKBACK_VALUES:
            hours = LOOKBACK_HOURS[lookback]
            print(f"\n{'='*60}")
            print(f"Training {model_label} with lookback={lookback} ({hours})")
            print(f"{'='*60}")

            model = model_cls(lookback=lookback, random_state=SEED)
            model.fit(X_train_seq, y_train)
            metrics = model.evaluate(
                X_test_seq, y_test,
                threshold=THRESHOLD,
                timestamps=test_timestamps,
            )

            # Extract pnl_series before saving (it's very long)
            pnl_series = metrics.pop("pnl_series")

            # Save result JSON
            result_name = f"{model_label}_lookback_{lookback}"
            extra = {
                "threshold": THRESHOLD,
                "n_train_rows": n_train,
                "n_test_rows": n_test,
                "n_features": n_features,
                "lookback": lookback,
                "lookback_hours": hours,
                "seed": SEED,
                "pnl_series": pnl_series,
            }
            save_results(result_name, metrics, RESULTS_DIR, extra=extra)

            # Store metrics for plotting
            results[model_label][lookback] = {
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "total_pnl": metrics["total_pnl"],
                "sharpe_ratio": metrics["sharpe_ratio"],
            }

            print(
                f"[{model_label}] lookback={lookback} ({hours}): "
                f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, "
                f"P&L={metrics['total_pnl']:.4f}, Sharpe={metrics['sharpe_ratio']:.4f}"
            )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("Experiment 2: Lookback Window Ablation Summary")
    print(f"{'='*72}")
    header = (
        f"{'Model':<6} | {'Lookback':>8} | {'Hours':>5} | "
        f"{'RMSE':>7} | {'MAE':>7} | {'P&L':>9} | {'Sharpe':>7}"
    )
    sep = (
        f"{'-'*6}-+-{'-'*8}-+-{'-'*5}-+-"
        f"{'-'*7}-+-{'-'*7}-+-{'-'*9}-+-{'-'*7}"
    )
    print(header)
    print(sep)

    for model_label, _ in MODEL_CLASSES:
        for lookback in LOOKBACK_VALUES:
            m = results[model_label][lookback]
            hours = LOOKBACK_HOURS[lookback]
            print(
                f"{model_label:<6} | {lookback:>8} | {hours:>5} | "
                f"{m['rmse']:>7.4f} | {m['mae']:>7.4f} | "
                f"{m['total_pnl']:>9.4f} | {m['sharpe_ratio']:>7.4f}"
            )

    # ------------------------------------------------------------------
    # RMSE line plot
    # ------------------------------------------------------------------
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    x_labels = [LOOKBACK_HOURS[lb] for lb in LOOKBACK_VALUES]

    gru_rmse = [results["GRU"][lb]["rmse"] for lb in LOOKBACK_VALUES]
    lstm_rmse = [results["LSTM"][lb]["rmse"] for lb in LOOKBACK_VALUES]

    ax.plot(
        x_labels, gru_rmse,
        color="blue", marker="o", linewidth=2, markersize=8,
        label="GRU",
    )
    ax.plot(
        x_labels, lstm_rmse,
        color="orange", marker="s", linewidth=2, markersize=8,
        label="LSTM",
    )

    ax.set_xlabel("Lookback Window (hours)")
    ax.set_ylabel("RMSE")
    ax.set_title("Experiment 2: RMSE vs Lookback Window")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "experiment2_lookback_rmse.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved RMSE plot: {FIGURES_DIR / 'experiment2_lookback_rmse.png'}")

    # ------------------------------------------------------------------
    # P&L line plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    gru_pnl = [results["GRU"][lb]["total_pnl"] for lb in LOOKBACK_VALUES]
    lstm_pnl = [results["LSTM"][lb]["total_pnl"] for lb in LOOKBACK_VALUES]

    ax.plot(
        x_labels, gru_pnl,
        color="blue", marker="o", linewidth=2, markersize=8,
        label="GRU",
    )
    ax.plot(
        x_labels, lstm_pnl,
        color="orange", marker="s", linewidth=2, markersize=8,
        label="LSTM",
    )

    ax.set_xlabel("Lookback Window (hours)")
    ax.set_ylabel("Total P&L")
    ax.set_title("Experiment 2: Total P&L vs Lookback Window")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "experiment2_lookback_pnl.png", dpi=150)
    plt.close(fig)
    print(f"Saved P&L plot: {FIGURES_DIR / 'experiment2_lookback_pnl.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
