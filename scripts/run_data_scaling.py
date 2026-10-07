"""Data-scaling experiment runner.

Trains each model tier on an increasing slice of the dataset and records
the results as one row in ``experiments/results/data_scaling/log.jsonl``.
The resulting log feeds the plot script and directly answers the central
research question from CLAUDE.md:

    *Does increasing model complexity improve arbitrage detection, and
    when is that complexity justified?*

Usage:
    # Run at a single checkpoint (for ad-hoc experimentation)
    python scripts/run_data_scaling.py --bars-per-pair 100

    # Run every uncompleted checkpoint (for cron / batch mode)
    python scripts/run_data_scaling.py --auto

    # List what's been recorded so far
    python scripts/run_data_scaling.py --show-log

Design notes:
  * We slice train.parquet by pair_id: take the first N bars of each pair
    to simulate "we only had N bars when we trained." This keeps the
    temporal ordering honest.
  * Test set (test.parquet) is held out across all runs so metrics are
    comparable between checkpoints.
  * Tier 1 (LR, XGBoost) runs unconditionally. Tier 2 (GRU, LSTM) only
    runs at checkpoints >= 100 bars/pair (below that is noise). Tier 3
    (PPO) is skipped here — it needs completed trajectory data from the
    position manager, not raw bars.
  * Each run appends a single JSONL row. The plot script reads the whole
    log and draws curves.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("run_data_scaling")

LOG_DIR = Path("experiments/results/data_scaling")
LOG_PATH = LOG_DIR / "log.jsonl"
STATE_PATH = LOG_DIR / "state.json"


def _load_state() -> dict:
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"last_checkpoint_ran": 0}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _append_log(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


def _show_log() -> int:
    if not LOG_PATH.exists():
        print("No scaling experiment runs yet.")
        return 0

    print(f"=== Data scaling experiment log ({LOG_PATH}) ===\n")
    print(f"{'bars/pair':>10} {'train rows':>11} {'timestamp':<25} {'best model':<25} {'best @2pp':>12}")
    print("-" * 90)

    with open(LOG_PATH) as f:
        for line in f:
            row = json.loads(line)
            metrics = row.get("metrics_by_model", {})
            if metrics:
                best_model = max(
                    metrics.items(),
                    key=lambda kv: kv[1].get("pnl_at_2pp", float("-inf")),
                )
                name, m = best_model
                best_pnl = m.get("pnl_at_2pp", float("nan"))
            else:
                name, best_pnl = "—", float("nan")
            print(
                f"{row['bars_per_pair']:>10} {row['training_rows']:>11} "
                f"{row['timestamp']:<25} {name:<25} {best_pnl:>12.3f}"
            )
    return 0


def _slice_train_by_bars_per_pair(train_df, bars_per_pair: int):
    """Return a copy of train_df containing only the first N bars per pair.

    The 'first N' is chosen by time_idx within each pair so that early
    bars always come first (temporal honesty: we never use future data).
    """
    train_df = train_df.sort_values(["pair_id", "time_idx"])
    return (
        train_df.groupby("pair_id", group_keys=False)
        .head(bars_per_pair)
        .reset_index(drop=True)
    )


def _load_train_test(data_dir: Path, include_category: bool = False):
    """Load train.parquet + test.parquet and compute the spread-change target.

    Mirrors experiments/run_baselines.py::_build_split without the torch
    dependency chain: per-pair next-bar minus current-bar spread, drop
    rows where the target or spread is NaN, fill remaining NaNs with 0.

    If ``include_category`` is True, derives a ``category`` column from
    each pair_id and one-hot encodes it. The resulting category indicator
    columns are prefixed ``cat_`` so the feature selector can pick them
    up with no extra config. Empty strings fall back to a dedicated
    ``cat_other`` column so models always see a complete one-hot.
    """
    import pandas as pd  # lazy

    train_path = data_dir / "train.parquet"
    test_path = data_dir / "test.parquet"
    missing = [p for p in (train_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Training data not found: {', '.join(str(p) for p in missing)}"
        )

    def _build_split(df):
        df = df.sort_values(["pair_id", "time_idx"]).reset_index(drop=True)
        df[_TARGET_COLUMN] = df.groupby("pair_id")["spread"].shift(-1) - df["spread"]
        df = df.dropna(subset=["spread", _TARGET_COLUMN]).reset_index(drop=True)
        df = df.fillna(0.0)
        return df

    train = _build_split(pd.read_parquet(train_path))
    test = _build_split(pd.read_parquet(test_path))

    if include_category:
        from src.features.category import (
            CATEGORIES,
            derive_category_from_pair_id,
        )

        train["category"] = train["pair_id"].apply(derive_category_from_pair_id)
        test["category"] = test["pair_id"].apply(derive_category_from_pair_id)

        # One-hot encode so linear models get per-category intercepts.
        # Tree models handle this fine too (they just learn splits on each
        # indicator). Force ALL categories into the column set even if
        # some are absent from this slice — makes columns identical
        # between train and test.
        for cat in CATEGORIES:
            col = f"cat_{cat}"
            train[col] = (train["category"] == cat).astype(float)
            test[col] = (test["category"] == cat).astype(float)
        # Drop the string column — only the one-hot indicators are used.
        train = train.drop(columns=["category"])
        test = test.drop(columns=["category"])

    return train, test


# Feature columns used by the Phase 5 harness. Mirrors experiments/run_baselines.py
# NON_FEATURE_COLUMNS exactly (without importing it, to avoid the torch chain).
# NOTE: 'spread' is KEPT as a feature — NaivePredictor and the spread-based
# baselines require it.
_TARGET_COLUMN = "spread_change_target"
_NON_FEATURE_COLUMNS = {
    "timestamp",
    "pair_id",
    "time_idx",
    "group_id",
    _TARGET_COLUMN,
    # Dropped at harness level (100% NaN / zero-variance upstream bugs)
    "kalshi_order_flow_imbalance",
    "kalshi_buy_volume",
    "kalshi_sell_volume",
    "kalshi_realized_spread",
}


def _feature_columns_lite(df) -> list[str]:
    """Return numeric feature columns, excluding non-feature metadata."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in numeric_cols if c not in _NON_FEATURE_COLUMNS]


def _prepare_xy(df, feature_cols: list[str]):
    import numpy as np  # lazy
    X = df[feature_cols].copy()
    y = df[_TARGET_COLUMN].to_numpy(dtype=float)
    return X, y


def _build_tier1_models():
    """Build Tier 1 models without triggering torch import chain."""
    from src.models.linear_regression import LinearRegressionPredictor
    from src.models.naive import NaivePredictor
    from src.models.volume import VolumePredictor
    from src.models.xgboost_model import XGBoostPredictor

    return [
        NaivePredictor(),
        VolumePredictor(),
        LinearRegressionPredictor(),
        XGBoostPredictor(n_estimators=200, max_depth=4, learning_rate=0.05),
    ]


def _run_tier1(train_sub, test_df, feature_cols):
    """Train Tier 1 models on the slice, evaluate on the held-out test set."""
    X_train, y_train = _prepare_xy(train_sub, feature_cols)
    X_test, y_test = _prepare_xy(test_df, feature_cols)

    results = {}
    timestamps = test_df["timestamp"].to_numpy() if "timestamp" in test_df.columns else None

    for model in _build_tier1_models():
        logger.info("Training %s on %d rows", model.name, len(X_train))
        model.fit(X_train, y_train)
        r = model.evaluate(X_test, y_test, threshold=0.02, timestamps=timestamps)
        # Compute simple fee-adjusted metric (P&L minus 2pp per trade)
        n_trades = r.get("num_trades", 0) or 0
        pnl_gross = r.get("total_pnl", 0.0) or 0.0
        results[model.name.lower().replace(" ", "_")] = {
            "rmse": float(r.get("rmse", float("nan"))),
            "mae": float(r.get("mae", float("nan"))),
            "dir_acc": float(r.get("directional_accuracy", float("nan"))),
            "num_trades": int(n_trades),
            "pnl_gross": float(pnl_gross),
            "pnl_at_2pp": float(pnl_gross - 0.02 * n_trades),
            "pnl_at_3pp": float(pnl_gross - 0.03 * n_trades),
            "sharpe": float(r.get("sharpe_ratio", float("nan"))),
        }
    return results


def _build_tier2_models():
    """Build Tier 2 models lazily (only imports torch when called)."""
    from src.models.gru import GRUPredictor
    from src.models.lstm import LSTMPredictor

    return [
        GRUPredictor(random_state=42),
        LSTMPredictor(random_state=42),
    ]


def _prepare_xy_for_seq(df, feature_cols: list[str]):
    """Sequence-model input needs group_id for per-pair windowing."""
    import numpy as np  # lazy
    X = df[feature_cols + ["group_id"]].copy()
    y = df[_TARGET_COLUMN].to_numpy(dtype=float)
    return X, y


def _run_tier2(train_sub, test_df, feature_cols):
    """Train Tier 2 sequence models (GRU/LSTM) on the slice."""
    X_train, y_train = _prepare_xy_for_seq(train_sub, feature_cols)
    X_test, y_test = _prepare_xy_for_seq(test_df, feature_cols)

    results = {}
    timestamps = test_df["timestamp"].to_numpy() if "timestamp" in test_df.columns else None

    for model in _build_tier2_models():
        logger.info("Training %s on %d rows", model.name, len(X_train))
        try:
            model.fit(X_train, y_train)
            r = model.evaluate(X_test, y_test, threshold=0.02, timestamps=timestamps)
        except Exception as e:
            logger.warning("Tier 2 model %s failed: %s", model.name, e)
            continue
        n_trades = r.get("num_trades", 0) or 0
        pnl_gross = r.get("total_pnl", 0.0) or 0.0
        results[model.name.lower().replace(" ", "_")] = {
            "rmse": float(r.get("rmse", float("nan"))),
            "mae": float(r.get("mae", float("nan"))),
            "dir_acc": float(r.get("directional_accuracy", float("nan"))),
            "num_trades": int(n_trades),
            "pnl_gross": float(pnl_gross),
            "pnl_at_2pp": float(pnl_gross - 0.02 * n_trades),
            "pnl_at_3pp": float(pnl_gross - 0.03 * n_trades),
            "sharpe": float(r.get("sharpe_ratio", float("nan"))),
        }
    return results


def run_checkpoint(
    bars_per_pair: int,
    data_dir: Path = Path("data/processed"),
    include_tier2: bool | None = None,
    include_category: bool = False,
) -> dict:
    """Execute one checkpoint: slice data, train all eligible models, log.

    Args:
        bars_per_pair: Slice to first N bars per pair.
        data_dir: Where to find train.parquet / test.parquet.
        include_tier2: If True, also train GRU/LSTM. Default auto-enables
            at 100+ bars/pair (see retraining_policy).
        include_category: If True, derive a ``category`` column from
            pair_id and add one-hot indicator features. Task #25.
    """
    if include_tier2 is None:
        include_tier2 = bars_per_pair >= 100  # from retraining_policy

    train_df, test_df = _load_train_test(data_dir, include_category=include_category)
    full_rows = len(train_df)

    train_sub = _slice_train_by_bars_per_pair(train_df, bars_per_pair)
    slice_rows = len(train_sub)

    if slice_rows == 0:
        raise ValueError(f"No training rows after slicing to {bars_per_pair} bars/pair")

    # Drop pairs that have too few bars to contribute
    pair_counts = train_sub.groupby("pair_id").size()
    valid_pairs = pair_counts[pair_counts >= min(bars_per_pair, 5)].index
    train_sub = train_sub[train_sub["pair_id"].isin(valid_pairs)].reset_index(drop=True)

    feature_cols = _feature_columns_lite(train_sub)
    logger.info(
        "checkpoint %d bars/pair: %d/%d training rows, %d features, %d pairs",
        bars_per_pair, slice_rows, full_rows, len(feature_cols),
        train_sub["pair_id"].nunique(),
    )

    metrics: dict[str, dict] = {}
    metrics.update(_run_tier1(train_sub, test_df, feature_cols))

    if include_tier2:
        try:
            metrics.update(_run_tier2(train_sub, test_df, feature_cols))
        except ImportError as e:
            logger.warning("Tier 2 imports unavailable (%s); skipping", e)

    row = {
        "bars_per_pair": bars_per_pair,
        "training_rows": slice_rows,
        "full_training_rows": full_rows,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "include_tier2": include_tier2,
        "include_category": include_category,
        "n_features": len(feature_cols),
        "metrics_by_model": metrics,
    }
    _append_log(row)
    logger.info("Checkpoint %d logged to %s", bars_per_pair, LOG_PATH)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the data-scaling experiment at one or more checkpoints",
        prog="python scripts/run_data_scaling.py",
    )
    parser.add_argument(
        "--bars-per-pair",
        type=int,
        help="Checkpoint to run (e.g. 50, 100, 250). Mutually exclusive with --auto.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run all checkpoints that haven't been run yet (cron mode)",
    )
    parser.add_argument(
        "--show-log",
        action="store_true",
        help="Print the existing scaling experiment log and exit",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing train.parquet + test.parquet",
    )
    parser.add_argument(
        "--include-tier2",
        action="store_true",
        help="Force tier-2 (GRU/LSTM) training even below 100 bars/pair",
    )
    parser.add_argument(
        "--include-category",
        action="store_true",
        help=(
            "Task #25: add one-hot category features derived from "
            "pair_id (crypto, oil, inflation, ...). Use to A/B test "
            "whether category-conditional rules lift performance."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.show_log:
        return _show_log()

    data_dir = Path(args.data_dir)

    if args.bars_per_pair is not None:
        run_checkpoint(
            args.bars_per_pair,
            data_dir=data_dir,
            include_tier2=True if args.include_tier2 else None,
            include_category=args.include_category,
        )
        return 0

    if args.auto:
        from src.experiments.retraining_policy import SCALING_CHECKPOINTS

        state = _load_state()
        last = state.get("last_checkpoint_ran", 0)

        # Determine current bars-per-pair from live data + train slice.
        # Use the number of PAIRS that have enough bars, not the median.
        # Discovery adds ~1000 new pairs/day at 1 bar each, which drags
        # the median down forever. Instead: a checkpoint is "ready" when
        # at least MIN_PAIRS_FOR_CHECKPOINT pairs have that many bars.
        MIN_PAIRS_FOR_CHECKPOINT = 20
        train_df, _ = _load_train_test(data_dir, include_category=args.include_category)
        bpp = train_df.groupby("pair_id").size()
        logger.info(
            "Bars/pair stats: median=%d, p75=%d, max=%d, total_pairs=%d",
            int(bpp.median()), int(bpp.quantile(0.75)), int(bpp.max()), len(bpp),
        )

        ran_any = False
        for cp in SCALING_CHECKPOINTS:
            if cp <= last:
                continue
            pairs_ready = int((bpp >= cp).sum())
            if pairs_ready < MIN_PAIRS_FOR_CHECKPOINT:
                logger.info(
                    "Checkpoint %d: only %d pairs ready (need %d) — stopping",
                    cp, pairs_ready, MIN_PAIRS_FOR_CHECKPOINT,
                )
                break
            logger.info(
                "Checkpoint %d: %d pairs ready — running",
                cp, pairs_ready,
            )
            run_checkpoint(cp, data_dir=data_dir, include_category=args.include_category)
            state["last_checkpoint_ran"] = cp
            _save_state(state)
            ran_any = True

        if not ran_any:
            logger.info("No new checkpoints to run (last=%d)", last)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
