"""Bootstrap 95% confidence intervals for all 8 models.

Trains each model once, collects test-set predictions, then resamples
(y_true, y_pred) pairs 1000 times to build nonparametric CIs on key
metrics.  This avoids retraining 8000 times -- the bootstrap loop is
purely on pre-computed predictions.

Outputs:
  - experiments/results/bootstrap_ci/bootstrap_results.json
  - experiments/results/bootstrap_ci/bootstrap_ci_table.txt
  - experiments/figures/bootstrap_ci_rmse.png

Run:
    python -m experiments.run_bootstrap_ci
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.run_baselines import (
    _build_split,
    _feature_columns,
    load_train_test,
    prepare_xy,
    prepare_xy_for_seq,
    NON_FEATURE_COLUMNS,
)
from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.profit_sim import simulate_profit
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
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
THRESHOLD = 0.02
SEED = 42
RESULTS_DIR = Path("experiments/results/bootstrap_ci")
FIGURES_DIR = Path("experiments/figures")
DATA_DIR = Path("data/processed")

# Fixed display order (same as run_baselines._MODEL_ORDER).
_MODEL_ORDER = [
    "Naive (Spread Closes)",
    "Volume (Higher Volume Correct)",
    "Linear Regression",
    "XGBoost",
    "GRU",
    "LSTM",
    "PPO-Raw",
    "PPO-Filtered",
]

# Tier colours for the forest plot.
_TIER_COLORS = {
    "Naive (Spread Closes)": "#7f7f7f",        # Tier 0: grey
    "Volume (Higher Volume Correct)": "#7f7f7f",
    "Linear Regression": "#1f77b4",             # Tier 1: blue
    "XGBoost": "#1f77b4",
    "GRU": "#2ca02c",                           # Tier 2: green
    "LSTM": "#2ca02c",
    "PPO-Raw": "#d62728",                       # Tier 3: red
    "PPO-Filtered": "#d62728",
}

_TIER_LABELS = {
    "Naive (Spread Closes)": "Naive baseline",
    "Volume (Higher Volume Correct)": "Naive baseline",
    "Linear Regression": "Tier 1 (Regression)",
    "XGBoost": "Tier 1 (Regression)",
    "GRU": "Tier 2 (Time Series)",
    "LSTM": "Tier 2 (Time Series)",
    "PPO-Raw": "Tier 3 (RL)",
    "PPO-Filtered": "Tier 3 (RL)",
}


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_all_models(
    df_train, df_test, feature_cols, test_timestamps,
) -> dict[str, np.ndarray]:
    """Train all 8 models and return {model_name: predictions_on_test}."""

    # Seed all RNG sources for deterministic training
    set_all_seeds(42)

    X_train_flat, y_train_flat = prepare_xy(df_train, feature_cols)
    X_test_flat, y_test_flat = prepare_xy(df_test, feature_cols)
    X_train_seq, y_train_seq = prepare_xy_for_seq(df_train, feature_cols)
    X_test_seq, y_test_seq = prepare_xy_for_seq(df_test, feature_cols)

    model_predictions: dict[str, np.ndarray] = {}

    # ---- Tier 1 (fast) ----
    tier1_models = [
        NaivePredictor(),
        VolumePredictor(),
        LinearRegressionPredictor(),
        XGBoostPredictor(n_estimators=200, max_depth=4, learning_rate=0.05),
    ]
    for model in tier1_models:
        t0 = time.time()
        model.fit(X_train_flat, y_train_flat)
        preds = model.predict(X_test_flat)
        elapsed = time.time() - t0
        print(f"  [{model.name}] trained + predicted in {elapsed:.1f}s")
        model_predictions[model.name] = np.asarray(preds, dtype=float)

    # ---- Tier 2 (moderate -- use max_epochs=50 for speed) ----
    tier2_classes = [
        ("GRU", GRUPredictor),
        ("LSTM", LSTMPredictor),
    ]
    for display_name, cls in tier2_classes:
        t0 = time.time()
        model = cls(random_state=SEED, max_epochs=50)
        model.fit(X_train_seq, y_train_seq)
        preds = model.predict(X_test_seq)
        elapsed = time.time() - t0
        print(f"  [{model.name}] trained + predicted in {elapsed:.1f}s")
        model_predictions[model.name] = np.asarray(preds, dtype=float)

    # ---- Tier 3 (use reduced timesteps for speed) ----
    n_features = len(feature_cols)

    # PPO-Raw
    t0 = time.time()
    ppo_raw = PPORawPredictor(random_state=SEED, total_timesteps=10_000)
    ppo_raw.fit(X_train_seq, y_train_seq)
    preds_raw = ppo_raw.predict(X_test_seq)
    elapsed = time.time() - t0
    print(f"  [{ppo_raw.name}] trained + predicted in {elapsed:.1f}s")
    model_predictions[ppo_raw.name] = np.asarray(preds_raw, dtype=float)

    # PPO-Filtered (train autoencoder first)
    t0 = time.time()
    autoencoder = AnomalyDetectorAutoencoder(
        input_dim=n_features, random_state=SEED,
    )
    autoencoder.fit(df_train[feature_cols], feature_cols)
    ppo_filt = PPOFilteredPredictor(
        anomaly_detector=autoencoder,
        random_state=SEED,
        total_timesteps=10_000,
    )
    ppo_filt.fit(X_train_seq, y_train_seq)
    preds_filt = ppo_filt.predict(X_test_seq)
    elapsed = time.time() - t0
    print(f"  [{ppo_filt.name}] trained + predicted in {elapsed:.1f}s")
    model_predictions[ppo_filt.name] = np.asarray(preds_filt, dtype=float)

    return model_predictions


# ---------------------------------------------------------------------------
# Bootstrap CI computation
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    y_test: np.ndarray,
    model_predictions: dict[str, np.ndarray],
    test_timestamps: np.ndarray | None,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
    threshold: float = THRESHOLD,
) -> dict[str, dict]:
    """Compute bootstrap CIs for each model.

    Returns:
        {model_name: {metric_name: {mean, ci_lower, ci_upper}}}
    """
    rng = np.random.RandomState(seed)
    n = len(y_test)

    # Pre-generate all bootstrap index arrays (reproducible).
    all_indices = rng.choice(n, size=(n_bootstrap, n), replace=True)

    metrics_keys = [
        "rmse", "mae", "directional_accuracy",
        "total_pnl", "num_trades", "win_rate", "sharpe_ratio",
    ]

    results: dict[str, dict] = {}

    for model_name, predictions in model_predictions.items():
        t0 = time.time()
        boot_metrics: dict[str, list[float]] = {k: [] for k in metrics_keys}

        for b in range(n_bootstrap):
            idx = all_indices[b]
            y_boot = y_test[idx]
            p_boot = predictions[idx]
            ts_boot = (
                test_timestamps[idx] if test_timestamps is not None else None
            )

            reg = compute_regression_metrics(y_boot, p_boot)
            trd = simulate_profit(
                p_boot, y_boot, threshold=threshold, timestamps=ts_boot,
            )

            for k in metrics_keys:
                val = reg.get(k) if k in reg else trd.get(k, 0.0)
                boot_metrics[k].append(float(val))

        # Compute CIs from collected bootstrap samples.
        alpha = (1.0 - CI_LEVEL) / 2.0  # 0.025 for 95% CI
        lower_pct = alpha * 100          # 2.5
        upper_pct = (1.0 - alpha) * 100  # 97.5

        model_ci: dict[str, dict] = {}
        for k, vals in boot_metrics.items():
            arr = np.array(vals)
            model_ci[k] = {
                "mean": float(np.mean(arr)),
                "ci_lower": float(np.percentile(arr, lower_pct)),
                "ci_upper": float(np.percentile(arr, upper_pct)),
            }

        results[model_name] = model_ci
        elapsed = time.time() - t0
        print(f"  [{model_name}] bootstrap CIs computed in {elapsed:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Output: JSON
# ---------------------------------------------------------------------------

def _save_json(
    results: dict[str, dict],
    n_test: int,
    out_path: Path,
) -> None:
    """Save bootstrap CI results to JSON."""
    payload = {
        "n_bootstrap": N_BOOTSTRAP,
        "ci_level": CI_LEVEL,
        "threshold": THRESHOLD,
        "n_test_rows": n_test,
        "models": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved JSON: {out_path}")


# ---------------------------------------------------------------------------
# Output: formatted text table
# ---------------------------------------------------------------------------

def _format_ci(ci_dict: dict) -> str:
    """Format a single CI entry as 'mean [lower-upper]'."""
    m = ci_dict["mean"]
    lo = ci_dict["ci_lower"]
    hi = ci_dict["ci_upper"]
    return f"{m:.4f} [{lo:.4f}-{hi:.4f}]"


def _save_table(
    results: dict[str, dict],
    out_path: Path,
) -> str:
    """Save a formatted text table of CIs and return the string."""
    # Sort models by _MODEL_ORDER.
    ordered = []
    for name in _MODEL_ORDER:
        if name in results:
            ordered.append((name, results[name]))
    # Any not in _MODEL_ORDER go at the end.
    for name in results:
        if name not in _MODEL_ORDER:
            ordered.append((name, results[name]))

    col_headers = ["Model", "RMSE", "MAE", "Dir Acc", "P&L", "Sharpe"]
    metric_keys = ["rmse", "mae", "directional_accuracy", "total_pnl", "sharpe_ratio"]

    # Compute column widths.
    rows: list[list[str]] = []
    for name, ci in ordered:
        row = [name]
        for mk in metric_keys:
            row.append(_format_ci(ci[mk]))
        rows.append(row)

    widths = [len(h) for h in col_headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            if i == 0:
                parts.append(f"{cell:<{widths[i]}}")
            else:
                parts.append(f"{cell:>{widths[i]}}")
        return " | ".join(parts)

    lines = [
        "Bootstrap 95% Confidence Intervals (1000 resamples)",
        "=" * 80,
        "",
        _fmt_row(col_headers),
        "-+-".join("-" * w for w in widths),
    ]
    for row in rows:
        lines.append(_fmt_row(row))

    # Overlap analysis for RMSE.
    lines.append("")
    lines.append("RMSE CI Overlap Analysis:")
    lines.append("-" * 40)
    names_list = [name for name, _ in ordered]
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            n1, ci1 = ordered[i]
            n2, ci2 = ordered[j]
            r1 = ci1["rmse"]
            r2 = ci2["rmse"]
            overlaps = r1["ci_lower"] <= r2["ci_upper"] and r2["ci_lower"] <= r1["ci_upper"]
            if overlaps:
                lines.append(
                    f"  {n1} vs {n2}: OVERLAPPING "
                    f"(indistinguishable at 95% level)"
                )

    table_str = "\n".join(lines) + "\n"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(table_str)
    print(f"Saved table: {out_path}")
    return table_str


# ---------------------------------------------------------------------------
# Output: RMSE forest plot
# ---------------------------------------------------------------------------

def _save_forest_plot(
    results: dict[str, dict],
    out_path: Path,
) -> None:
    """Create a horizontal forest plot of RMSE with 95% CI error bars."""
    # Order models.
    ordered = []
    for name in _MODEL_ORDER:
        if name in results:
            ordered.append(name)
    for name in results:
        if name not in _MODEL_ORDER:
            ordered.append(name)

    # Reverse so the first model appears at the top of the plot.
    ordered = list(reversed(ordered))

    names = []
    means = []
    lowers = []
    uppers = []
    colors = []

    for name in ordered:
        ci = results[name]["rmse"]
        names.append(name)
        means.append(ci["mean"])
        lowers.append(ci["mean"] - ci["ci_lower"])
        uppers.append(ci["ci_upper"] - ci["mean"])
        colors.append(_TIER_COLORS.get(name, "#333333"))

    y_pos = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        y_pos,
        means,
        xerr=[lowers, uppers],
        height=0.5,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=4,
        alpha=0.8,
        ecolor="black",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("RMSE", fontsize=12)
    ax.set_title(
        f"Bootstrap 95% CI for RMSE ({N_BOOTSTRAP} resamples)",
        fontsize=13,
        fontweight="bold",
    )
    ax.axvline(x=0, color="grey", linewidth=0.5, linestyle="--")

    # Legend for tier colours.
    from matplotlib.patches import Patch
    seen = set()
    legend_handles = []
    for name in reversed(ordered):
        label = _TIER_LABELS.get(name, "")
        color = _TIER_COLORS.get(name, "#333333")
        key = (label, color)
        if key not in seen and label:
            seen.add(key)
            legend_handles.append(Patch(facecolor=color, label=label, alpha=0.8))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    set_all_seeds(42)
    print("=" * 60)
    print("Bootstrap 95% Confidence Intervals")
    print(f"  N_BOOTSTRAP={N_BOOTSTRAP}, CI_LEVEL={CI_LEVEL}")
    print(f"  THRESHOLD={THRESHOLD}, SEED={SEED}")
    print("=" * 60)

    # Load data.
    train_raw, test_raw = load_train_test(DATA_DIR)
    train = _build_split(train_raw)
    test = _build_split(test_raw)
    feature_cols = _feature_columns(train)

    n_train = len(train)
    n_test = len(test)
    n_features = len(feature_cols)
    print(f"\nData: {n_train} train, {n_test} test, {n_features} features")

    test_timestamps = test["timestamp"].to_numpy()
    _, y_test = prepare_xy(test, feature_cols)

    # Step 1: Train all models and get predictions.
    print("\n--- Training all 8 models ---")
    model_predictions = _train_all_models(
        train, test, feature_cols, test_timestamps,
    )
    print(f"\nGot predictions for {len(model_predictions)} models.")

    # Step 2: Bootstrap CI computation.
    print(f"\n--- Computing {N_BOOTSTRAP} bootstrap CIs ---")
    ci_results = _bootstrap_ci(
        y_test, model_predictions, test_timestamps,
    )

    # Step 3: Save outputs.
    json_path = RESULTS_DIR / "bootstrap_results.json"
    table_path = RESULTS_DIR / "bootstrap_ci_table.txt"
    plot_path = FIGURES_DIR / "bootstrap_ci_rmse.png"

    _save_json(ci_results, n_test, json_path)
    table_str = _save_table(ci_results, table_path)
    _save_forest_plot(ci_results, plot_path)

    # Print table to stdout.
    print("\n" + table_str)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
