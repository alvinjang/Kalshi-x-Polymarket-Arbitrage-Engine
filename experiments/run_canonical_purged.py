"""Phase 18-08 leakage-free retraining of LR + XGBoost.

Sister script to ``experiments/run_canonical.py``. Uses the **purged
pair-stratified split** built by ``experiments/audit/build_purged_split.py``
(stored under ``data/processed/purged_split/``) instead of the canonical
80/20 row-index split. Every other knob is held identical:

    seed = 42
    position_size = $100
    threshold = 0.02 (signal gate, not fee — Tier 3 audit confirmed
        ``simulate_profit`` charges zero fees)
    feature pipeline = ``compute_derived_features`` +
        ``select_dtypes(['number'])`` minus NON_FEATURE_COLUMNS
        (51 numeric features)

Scope is intentionally tight: **LR + XGBoost only**. These are the
headline models per Phase 17 (LR wins 4 of 5 metrics in
``experiments/results/canonical/headline.json``); GRU/LSTM/TFT/PPO
retraining is out of scope for this plan because they are not
load-bearing for the leakage-free Sharpe headline that Plan 18-07
needs to consume.

Output:
    experiments/results/canonical_purged/headline.json

Usage:
    PYTHONPATH=. python experiments/run_canonical_purged.py

Implements requirement: AUDIT-07 (training side).
"""
# AI-assisted authorship: written with Anthropic Claude (Opus 4.7) as
# pair-programming assistant. All design decisions are the authors'.
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the canonical training helpers verbatim — do NOT duplicate
# preprocessing / metric computation. The whole point of this plan is to
# swap ONLY the data source.
from experiments.run_baselines import (
    NON_FEATURE_COLUMNS,
    TARGET_COLUMN,
    _build_split,
    _feature_columns,
    prepare_xy,
)
from experiments.run_canonical import (
    CANONICAL_POSITION_SIZE,
    CANONICAL_SEED,
    CANONICAL_THRESHOLD,
    evaluate_predictions,
)
from src.features.engineering import compute_derived_features
from src.models.linear_regression import LinearRegressionPredictor
from src.models.xgboost_model import XGBoostPredictor
from src.utils.seed import set_all_seeds


PURGED_TRAIN = Path("data/processed/purged_split/train.parquet")
PURGED_TEST = Path("data/processed/purged_split/test.parquet")
PURGED_METADATA = Path("data/processed/purged_split/split_metadata.json")
CANONICAL_HEADLINE = Path("experiments/results/canonical/headline.json")
OUTPUT_DIR = Path("experiments/results/canonical_purged")
OUTPUT_JSON = OUTPUT_DIR / "headline.json"


def load_purged_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read purged parquets and apply the canonical feature pipeline.

    Mirrors ``experiments.run_baselines.load_train_test`` but reads from
    the purged_split directory. Each DataFrame is run through
    ``compute_derived_features`` so the 51-feature numeric matrix matches
    the canonical pipeline.
    """
    if not PURGED_TRAIN.exists() or not PURGED_TEST.exists():
        raise FileNotFoundError(
            f"Purged split missing at {PURGED_TRAIN} / {PURGED_TEST}. "
            "Run experiments/audit/build_purged_split.py first."
        )
    train = compute_derived_features(pd.read_parquet(PURGED_TRAIN))
    test = compute_derived_features(pd.read_parquet(PURGED_TEST))
    return train, test


def train_linear_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    actuals: np.ndarray,
    timestamps: np.ndarray,
) -> dict:
    """Train LR on purged data, return canonical-shape metrics dict."""
    set_all_seeds(CANONICAL_SEED)
    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_test, _ = prepare_xy(test_df, feature_cols)
    model = LinearRegressionPredictor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return evaluate_predictions(
        "linear_regression", preds, actuals, timestamps, test_df
    )


def train_xgboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    actuals: np.ndarray,
    timestamps: np.ndarray,
) -> dict:
    """Train XGBoost on purged data, return canonical-shape metrics dict."""
    set_all_seeds(CANONICAL_SEED)
    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_test, _ = prepare_xy(test_df, feature_cols)
    model = XGBoostPredictor(n_estimators=200, max_depth=4, learning_rate=0.05)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return evaluate_predictions(
        "xgboost", preds, actuals, timestamps, test_df
    )


def main() -> int:
    set_all_seeds(CANONICAL_SEED)

    # ---- Load purged data + apply canonical feature pipeline ----
    train_raw, test_raw = load_purged_train_test()
    train_df = _build_split(train_raw)
    test_df = _build_split(test_raw)
    feature_cols = _feature_columns(train_df)

    actuals = test_df[TARGET_COLUMN].to_numpy(dtype=float)
    timestamps = test_df["timestamp"].astype("int64").to_numpy() // 10**9

    n_train_pairs = int(train_df["pair_id"].nunique())
    n_test_pairs = int(test_df["pair_id"].nunique())
    bridging = set(train_df["pair_id"]) & set(test_df["pair_id"])
    if bridging:
        raise RuntimeError(
            f"FATAL: purged split has {len(bridging)} bridging pairs — "
            f"split builder is broken; abort retrain."
        )

    print(
        f"Loaded purged split: {len(train_df)} train rows / {len(test_df)} "
        f"test rows | {n_train_pairs} train pairs / {n_test_pairs} test pairs "
        f"({len(feature_cols)} features)"
    )
    print(f"Canonical seed = {CANONICAL_SEED}")
    print(f"Canonical position_size = ${CANONICAL_POSITION_SIZE}")
    print(f"Canonical threshold = {CANONICAL_THRESHOLD}")
    print()

    # ---- Train LR + XGBoost ----
    print("[purged] Fitting linear_regression ...")
    lr_metrics = train_linear_regression(
        train_df, test_df, feature_cols, actuals, timestamps
    )
    print(
        f"  -> pnl={lr_metrics['total_pnl']:.4f} "
        f"trades={lr_metrics['num_trades']} "
        f"sharpe_pt={lr_metrics['sharpe_per_trade']:.4f} "
        f"alpha_bps={lr_metrics['alpha_bps_per_trade']:.2f}"
    )

    print("[purged] Fitting xgboost ...")
    xgb_metrics = train_xgboost(
        train_df, test_df, feature_cols, actuals, timestamps
    )
    print(
        f"  -> pnl={xgb_metrics['total_pnl']:.4f} "
        f"trades={xgb_metrics['num_trades']} "
        f"sharpe_pt={xgb_metrics['sharpe_per_trade']:.4f} "
        f"alpha_bps={xgb_metrics['alpha_bps_per_trade']:.2f}"
    )

    # ---- Mark each entry's source so the audit script can tell them apart ----
    lr_metrics["source"] = "retrained_in_canonical_purged_script"
    xgb_metrics["source"] = "retrained_in_canonical_purged_script"

    # ---- Assemble purged headline.json ----
    headline = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "experiments/run_canonical_purged.py",
        "phase": "18-08",
        "purpose": (
            "Leakage-free LR + XGBoost retraining on the pair-stratified "
            "train/test split built by experiments/audit/build_purged_split.py. "
            "Produces the headline numbers that replace the leaky canonical "
            "headline (per Plan 18-07 resumption)."
        ),
        "protocol": {
            "seed": CANONICAL_SEED,
            "position_size_usd": CANONICAL_POSITION_SIZE,
            "threshold": CANONICAL_THRESHOLD,
            "split_type": "pair_stratified_80_20",
            "split_seed": CANONICAL_SEED,
            "n_train_pairs": n_train_pairs,
            "n_test_pairs": n_test_pairs,
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "feature_count": len(feature_cols),
            "non_feature_columns": sorted(NON_FEATURE_COLUMNS),
            "purged_split_metadata": PURGED_METADATA.as_posix(),
            "source": "experiments/run_canonical_purged.py",
        },
        "models": {
            "linear_regression": lr_metrics,
            "xgboost": xgb_metrics,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(headline, indent=2))
    print(f"\nWrote {OUTPUT_JSON}")

    # ---- Side-by-side comparison vs leaky canonical headline ----
    if CANONICAL_HEADLINE.exists():
        canonical = json.loads(CANONICAL_HEADLINE.read_text())
        c_lr = canonical["models"]["linear_regression"]
        c_xgb = canonical["models"]["xgboost"]
        print()
        print("=" * 70)
        print("LEAKY canonical -> PURGED side-by-side")
        print("=" * 70)
        print(
            f"LR  sharpe_per_trade : {c_lr['sharpe_per_trade']:>+8.4f} -> "
            f"{lr_metrics['sharpe_per_trade']:>+8.4f}  "
            f"(delta {lr_metrics['sharpe_per_trade'] - c_lr['sharpe_per_trade']:+.4f})"
        )
        print(
            f"XGB sharpe_per_trade : {c_xgb['sharpe_per_trade']:>+8.4f} -> "
            f"{xgb_metrics['sharpe_per_trade']:>+8.4f}  "
            f"(delta {xgb_metrics['sharpe_per_trade'] - c_xgb['sharpe_per_trade']:+.4f})"
        )
        print(
            f"LR  total_pnl        : ${c_lr['total_pnl']:>+9.2f} -> "
            f"${lr_metrics['total_pnl']:>+9.2f}  "
            f"(delta {lr_metrics['total_pnl'] - c_lr['total_pnl']:+.2f})"
        )
        print(
            f"XGB total_pnl        : ${c_xgb['total_pnl']:>+9.2f} -> "
            f"${xgb_metrics['total_pnl']:>+9.2f}  "
            f"(delta {xgb_metrics['total_pnl'] - c_xgb['total_pnl']:+.2f})"
        )
        print(
            f"LR  win_rate         : {c_lr['win_rate']:>8.4f} -> "
            f"{lr_metrics['win_rate']:>8.4f}  "
            f"(delta {lr_metrics['win_rate'] - c_lr['win_rate']:+.4f})"
        )
        print(
            f"XGB win_rate         : {c_xgb['win_rate']:>8.4f} -> "
            f"{xgb_metrics['win_rate']:>8.4f}  "
            f"(delta {xgb_metrics['win_rate'] - c_xgb['win_rate']:+.4f})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
