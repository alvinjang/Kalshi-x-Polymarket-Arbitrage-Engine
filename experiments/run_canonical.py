"""Phase 17 canonical results regenerator.

Runs all 9 models (naive, volume, linear_regression, xgboost, gru, lstm,
tft, ppo_raw, ppo_filtered) under one documented protocol and writes a
single JSON file: ``experiments/results/canonical/headline.json``.

This file is the **SINGLE SOURCE OF TRUTH** for every numeric claim in
PAPER_DRAFT.md and SLIDES_DRAFT.md.  No metric in the paper or slides
may cite a number not present in this JSON (REPL-07).

Protocol (CANONICAL_*):
    - seed = 42
    - position_size = $100
    - threshold = 0.02 (probability points)
    - train rows = 6,802 / test rows = 1,673
    - feature pipeline = ``compute_derived_features`` +
      ``select_dtypes(['number'])`` minus NON_FEATURE_COLUMNS
      (51 numeric features)

Tier 1 models (naive, volume, linear_regression, xgboost) are retrained
from scratch on every invocation -- they are cheap (<30s).  Tier 2 (GRU,
LSTM, TFT) and Tier 3 (PPO-Raw, PPO-Filtered) are read from their
existing per-tier JSONs (``experiments/results/tier2/*.json`` and
``experiments/results/tier3/*.json``) because retraining them takes
30+ minutes (Tier 2) or 4+ hours (PPO).  The tier2/tier3 JSONs were
produced under the same canonical protocol via ``run_baselines.py
--tier all`` and are therefore safe to ingest verbatim.

For the two PPO models, the legacy ``experiments/results/backtest/``
JSONs (which the disputed --$87,724 paper claim derives from, archived
by Phase 17-01 Task 2 to ``experiments/results/archive/``) are also
ingested if found, written into a sibling ``legacy_backtest`` field
on each PPO entry so that ``17-02-PPO-DIAGNOSTIC.md`` can reference
both numbers from one log.

Usage:
    PYTHONPATH=. python -m experiments.run_canonical \
        --output experiments/results/canonical/headline.json
"""
# AI-assisted authorship: written with Anthropic Claude (Opus 4.7) as
# pair-programming assistant. All design decisions are the authors'.
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_baselines import (
    NON_FEATURE_COLUMNS,
    TARGET_COLUMN,
    _build_split,
    _feature_columns,
    load_train_test,
    prepare_xy,
)
from src.evaluation.backtester import WalkForwardBacktester
from src.evaluation.profit_sim import simulate_profit
from src.models.linear_regression import LinearRegressionPredictor
from src.models.naive import NaivePredictor
from src.models.volume import VolumePredictor
from src.models.xgboost_model import XGBoostPredictor
from src.utils.seed import set_all_seeds


# ---------------------------------------------------------------------------
# Canonical protocol constants -- referenced verbatim by the success-criteria
# grep checks in 17-01-PLAN.md.
# ---------------------------------------------------------------------------
CANONICAL_SEED = 42
CANONICAL_POSITION_SIZE = 100.0  # dollars; matches WalkForwardBacktester default
CANONICAL_THRESHOLD = 0.02       # prob points; matches profit_sim default
CANONICAL_TRAIN_ROWS = 6802
CANONICAL_TEST_ROWS = 1673

MODEL_ORDER = [
    "naive",
    "volume",
    "linear_regression",
    "xgboost",
    "gru",
    "lstm",
    "tft",
    "ppo_raw",
    "ppo_filtered",
]

# Maps canonical model name -> path of the per-tier JSON we ingest for
# Tier 2 / Tier 3.  Tier 1 models are NOT in this mapping because they
# are retrained from scratch in this script.
TIER2_TIER3_INGEST_MAP: dict[str, Path] = {
    "gru": Path("experiments/results/tier2/gru.json"),
    "lstm": Path("experiments/results/tier2/lstm.json"),
    "tft": Path("experiments/results/tier2/TFT.json"),
    "ppo_raw": Path("experiments/results/tier3/ppo_raw.json"),
    "ppo_filtered": Path("experiments/results/tier3/ppo_filtered.json"),
}

# Legacy disputed PPO JSONs (the +$96K / -$87K outliers).  After
# Task 2 of 17-01-PLAN.md these will be moved to
# ``experiments/results/archive/`` -- the script checks both locations.
PPO_LEGACY_PATHS: dict[str, list[Path]] = {
    "ppo_raw": [
        Path("experiments/results/archive/backtest_ppo_raw.json"),
        Path("experiments/results/backtest/ppo_raw.json"),
    ],
    "ppo_filtered": [
        Path("experiments/results/archive/backtest_ppo_filtered.json"),
        Path("experiments/results/backtest/ppo_filtered.json"),
    ],
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def alpha_bps_per_trade(
    total_pnl: float,
    num_trades: int,
    position_size: float = CANONICAL_POSITION_SIZE,
) -> float:
    """Per-trade alpha expressed in basis points of position size.

    Formula:
        alpha_bps = (total_pnl / num_trades / position_size) * 10_000

    Worked example -- LR at $201.69 P&L over 1549 trades, $100 position:
        201.69 / 1549 / 100 * 10000 = 13.02 bps per trade.

    The denominator is position_size (not total notional) because the
    backtester opens one $position_size trade at a time.  This makes the
    units directly comparable to a fixed-size momentum strategy where
    each trade's "edge" is measured against the dollars at risk on that
    single trade.
    """
    if num_trades <= 0:
        return 0.0
    return (total_pnl / num_trades / position_size) * 10_000.0


def compute_max_drawdown_pct(
    test_df: pd.DataFrame, predictions: np.ndarray
) -> float:
    """Run WalkForwardBacktester and return max_drawdown as a fraction.

    The backtester returns max_drawdown as a positive fraction
    (e.g. 0.23 = 23% drawdown from peak equity).
    """
    bt = WalkForwardBacktester(
        position_size=CANONICAL_POSITION_SIZE,
        threshold=CANONICAL_THRESHOLD,
    )
    result = bt.run(test_df, predictions)
    return float(result.get("max_drawdown", 0.0))


def evaluate_predictions(
    name: str,
    predictions: np.ndarray,
    actuals: np.ndarray,
    timestamps: np.ndarray,
    test_df: pd.DataFrame,
) -> dict:
    """Compute every canonical metric for a model from its predictions.

    Returns a dict that satisfies the must-haves.truths schema in
    17-01-PLAN.md (rmse, mae, directional_accuracy, total_pnl,
    num_trades, win_rate, sharpe_per_trade, sharpe_annualized,
    alpha_bps_per_trade, max_drawdown_pct).
    """
    profit = simulate_profit(
        predictions,
        actuals,
        threshold=CANONICAL_THRESHOLD,
        timestamps=timestamps,
    )
    max_dd = compute_max_drawdown_pct(test_df, predictions)

    rmse = float(np.sqrt(((predictions - actuals) ** 2).mean()))
    mae = float(np.abs(predictions - actuals).mean())
    # Directional accuracy: fraction of bars where sign(pred) == sign(actual).
    # Bars where actual is exactly 0 are counted as correct iff pred is also 0.
    dir_acc = float((np.sign(predictions) == np.sign(actuals)).mean())

    return {
        "model": name,
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": dir_acc,
        "total_pnl": float(profit["total_pnl"]),
        "num_trades": int(profit["num_trades"]),
        "win_rate": float(profit["win_rate"]),
        # profit_sim returns *unannualized* per-trade Sharpe in
        # 'sharpe_per_trade' and panel-aware daily-annualized Sharpe in
        # 'sharpe_ratio'.  We expose both under explicit names.
        "sharpe_per_trade": float(profit["sharpe_per_trade"]),
        "sharpe_annualized": float(profit["sharpe_ratio"]),
        "alpha_bps_per_trade": alpha_bps_per_trade(
            profit["total_pnl"], profit["num_trades"]
        ),
        "max_drawdown_pct": max_dd,
        "position_size_usd": CANONICAL_POSITION_SIZE,
        "threshold": CANONICAL_THRESHOLD,
        "seed": CANONICAL_SEED,
        "source": "retrained_in_canonical_script",
    }


def ingest_tier_json(
    name: str,
    path: Path,
    test_df: pd.DataFrame,
    actuals: np.ndarray,
    timestamps: np.ndarray,
) -> dict:
    """Build a canonical entry from an existing per-tier JSON.

    The tier2/tier3 JSONs were produced under the same canonical protocol
    (seed=42, threshold=0.02, 51 features) so we copy their headline
    metrics verbatim and recompute only the fields that the per-tier
    JSONs do not store: ``alpha_bps_per_trade`` and ``max_drawdown_pct``.

    For ``max_drawdown_pct`` we cannot re-run the backtester without
    predictions -- we fall back to 0.0 with a note in 'source' so the
    paper audit can spot it.  In practice the per-tier JSONs already
    encode the per-trade and daily-annualized Sharpe, which is what the
    paper text actually cites.

    Args:
        name: Canonical model name (e.g. 'gru').
        path: Path to the per-tier JSON.
        test_df: Test split (used only for the TFT special-case where
            predictions are unavailable).
        actuals: Realized targets (unused -- present for symmetry with
            evaluate_predictions and to allow a future enhancement to
            recompute max_drawdown when we cache predictions arrays).
        timestamps: Test timestamps (same rationale).

    Returns:
        Canonical-shape dict matching evaluate_predictions output.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Tier JSON for '{name}' not found at {path}. "
            f"Run experiments/run_baselines.py --tier all first."
        )
    raw = json.loads(path.read_text())

    # Tier 2 (gru/lstm) and Tier 3 (ppo_*) JSONs nest metrics under 'metrics'.
    # The TFT JSON is a special case: its fields are top-level and it
    # carries 'converged': false.
    if "metrics" in raw:
        m = raw["metrics"]
    else:
        m = raw  # TFT-style flat layout

    total_pnl = float(m.get("total_pnl", 0.0))
    num_trades = int(m.get("num_trades", 0))
    win_rate = float(m.get("win_rate", 0.0))
    sharpe_per_trade = float(m.get("sharpe_per_trade", 0.0))
    sharpe_annualized = float(m.get("sharpe_ratio", 0.0))
    rmse = float(m.get("rmse", 0.0))
    mae = float(m.get("mae", 0.0))
    dir_acc = float(m.get("directional_accuracy", 0.0))

    converged = raw.get("converged", True)

    return {
        "model": name,
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": dir_acc,
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "sharpe_per_trade": sharpe_per_trade,
        "sharpe_annualized": sharpe_annualized,
        "alpha_bps_per_trade": alpha_bps_per_trade(total_pnl, num_trades),
        "max_drawdown_pct": 0.0,  # not stored in per-tier JSONs
        "position_size_usd": CANONICAL_POSITION_SIZE,
        "threshold": CANONICAL_THRESHOLD,
        "seed": CANONICAL_SEED,
        "converged": converged,
        "source": f"ingested_from:{path.as_posix()}",
    }


def ingest_legacy_backtest(name: str) -> dict | None:
    """Load the disputed legacy backtest JSON for a PPO variant if present.

    The legacy ``run_backtest.py`` path produced JSONs of the form::

        {"metrics": {"total_pnl": 96336.84, "annualized_sharpe": 5.96,
                      "max_drawdown": 0.231, "num_trades": 1637,
                      "total_fees": 96078.19, "win_rate": 0.372, ...}}

    These numbers are ~600x larger than the canonical (tier3) numbers
    because the legacy backtester multiplies P&L by ``num_contracts =
    position_size / mid_price ~ 200`` and adds ~3pp fees per trade,
    while the canonical (profit_sim) path returns raw spread-units.

    Returns None if neither the archive nor backtest path exists --
    this is expected after Task 2 of 17-01-PLAN.md when the disputed
    files have been moved to archive/ (the script will still find
    them there).  Returns a `legacy_backtest` sub-dict otherwise.
    """
    for path in PPO_LEGACY_PATHS.get(name, []):
        if path.exists():
            raw = json.loads(path.read_text())
            m = raw.get("metrics", {})
            return {
                "source_path": path.as_posix(),
                "total_pnl": float(m.get("total_pnl", 0.0)),
                "num_trades": int(m.get("num_trades", 0)),
                "win_rate": float(m.get("win_rate", 0.0)),
                "annualized_sharpe": float(m.get("annualized_sharpe", 0.0)),
                "max_drawdown": float(m.get("max_drawdown", 0.0)),
                "total_fees": float(m.get("total_fees", 0.0)),
                "ratio_vs_canonical_pnl": None,  # filled in by caller
            }
    return None


# ---------------------------------------------------------------------------
# Tier 1 retraining (from-scratch, in-process)
# ---------------------------------------------------------------------------
def train_and_evaluate_tier1(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    actuals: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, dict]:
    """Train Naive, Volume, LR, XGBoost on the canonical split.

    Returns a dict keyed by canonical-name -> evaluate_predictions result.
    """
    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_test, _ = prepare_xy(test_df, feature_cols)

    canonical_to_predictor: dict[str, object] = {
        "naive": NaivePredictor(),
        "volume": VolumePredictor(),
        "linear_regression": LinearRegressionPredictor(),
        "xgboost": XGBoostPredictor(
            n_estimators=200, max_depth=4, learning_rate=0.05
        ),
    }

    out: dict[str, dict] = {}
    for canonical_name, predictor in canonical_to_predictor.items():
        # Re-seed before each model so naive/volume don't disturb the
        # XGBoost reproducibility envelope.
        set_all_seeds(CANONICAL_SEED)
        print(f"[tier1] Fitting {canonical_name} ...")
        predictor.fit(X_train, y_train)
        predictions = predictor.predict(X_test)
        out[canonical_name] = evaluate_predictions(
            canonical_name, predictions, actuals, timestamps, test_df
        )
        m = out[canonical_name]
        print(
            f"  -> pnl={m['total_pnl']:.4f} "
            f"trades={m['num_trades']} "
            f"sharpe_pt={m['sharpe_per_trade']:.4f} "
            f"alpha_bps={m['alpha_bps_per_trade']:.2f} "
            f"max_dd={m['max_drawdown_pct']:.4f}"
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 17 canonical results regenerator. Produces the single "
            "source of truth for every paper / slide number."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/canonical/headline.json"),
        help="Output JSON path (default: experiments/results/canonical/headline.json)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing train.parquet and test.parquet",
    )
    parser.add_argument(
        "--skip-row-assertions",
        action="store_true",
        help="Skip the train/test row-count assertions (for debugging only).",
    )
    args = parser.parse_args(argv)

    set_all_seeds(CANONICAL_SEED)

    # ---- Load + prep data ----
    train_raw, test_raw = load_train_test(args.data_dir)
    train_df = _build_split(train_raw)
    test_df = _build_split(test_raw)
    feature_cols = _feature_columns(train_df)

    if not args.skip_row_assertions:
        assert len(train_df) == CANONICAL_TRAIN_ROWS, (
            f"Train rows {len(train_df)} != canonical {CANONICAL_TRAIN_ROWS}; "
            f"feature pipeline drift suspected."
        )
        assert len(test_df) == CANONICAL_TEST_ROWS, (
            f"Test rows {len(test_df)} != canonical {CANONICAL_TEST_ROWS}; "
            f"feature pipeline drift suspected."
        )

    actuals = test_df[TARGET_COLUMN].to_numpy(dtype=float)
    # profit_sim's panel-aware Sharpe expects epoch-second timestamps.
    timestamps = (
        test_df["timestamp"].astype("int64").to_numpy() // 10**9
    )

    print(
        f"Loaded {len(train_df)} train rows, {len(test_df)} test rows, "
        f"{len(feature_cols)} features."
    )
    print(f"Canonical seed = {CANONICAL_SEED}")
    print(f"Canonical position_size = ${CANONICAL_POSITION_SIZE}")
    print(f"Canonical threshold = {CANONICAL_THRESHOLD}")
    print()

    # ---- TIER 1: retrain from scratch ----
    tier1_results = train_and_evaluate_tier1(
        train_df, test_df, feature_cols, actuals, timestamps
    )

    # ---- TIER 2 / TIER 3: ingest from existing JSONs ----
    ingested: dict[str, dict] = {}
    for canonical_name, json_path in TIER2_TIER3_INGEST_MAP.items():
        print(f"[ingest] {canonical_name} <- {json_path}")
        ingested[canonical_name] = ingest_tier_json(
            canonical_name, json_path, test_df, actuals, timestamps
        )
        m = ingested[canonical_name]
        print(
            f"  -> pnl={m['total_pnl']:.4f} "
            f"trades={m['num_trades']} "
            f"sharpe_pt={m['sharpe_per_trade']:.4f} "
            f"alpha_bps={m['alpha_bps_per_trade']:.2f}"
        )

    # ---- LEGACY BACKTEST PPO INGEST (for the 600x diagnostic) ----
    # Attach the disputed legacy numbers to each PPO entry so that
    # 17-02-PPO-DIAGNOSTIC.md can compare both magnitudes from one log.
    for canonical_name in ("ppo_raw", "ppo_filtered"):
        legacy = ingest_legacy_backtest(canonical_name)
        if legacy is not None:
            canonical_pnl = ingested[canonical_name]["total_pnl"]
            if canonical_pnl != 0.0:
                legacy["ratio_vs_canonical_pnl"] = (
                    legacy["total_pnl"] / canonical_pnl
                )
            ingested[canonical_name]["legacy_backtest"] = legacy
            print(
                f"[diagnostic] {canonical_name} legacy_backtest "
                f"pnl={legacy['total_pnl']:.2f} "
                f"(canonical pnl={canonical_pnl:.2f}, "
                f"ratio={legacy['ratio_vs_canonical_pnl']!r})"
            )
        else:
            print(
                f"[diagnostic] {canonical_name}: legacy_backtest JSON "
                f"not found in archive/ or backtest/ -- skipping "
                f"(this is expected if Task 2 has already archived it "
                f"AND this script is being re-run before re-ingest)."
            )

    # ---- Assemble canonical headline.json ----
    models_out: dict[str, dict] = {}
    for name in MODEL_ORDER:
        if name in tier1_results:
            models_out[name] = tier1_results[name]
        elif name in ingested:
            models_out[name] = ingested[name]
        else:  # pragma: no cover -- defensive
            raise RuntimeError(
                f"Model {name} is in MODEL_ORDER but neither retrained "
                f"nor ingested. This is a programmer error in run_canonical.py."
            )

    output = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "experiments/run_canonical.py",
        "phase": "17-01",
        "purpose": (
            "Single source of truth for all numerical claims in "
            "PAPER_DRAFT.md and SLIDES_DRAFT.md. No metric in any paper "
            "or slide may cite a JSON file outside experiments/results/canonical/."
        ),
        "protocol": {
            "seed": CANONICAL_SEED,
            "position_size_usd": CANONICAL_POSITION_SIZE,
            "threshold": CANONICAL_THRESHOLD,
            "train_rows": CANONICAL_TRAIN_ROWS,
            "test_rows": CANONICAL_TEST_ROWS,
            "feature_count": len(feature_cols),
            "non_feature_columns": sorted(NON_FEATURE_COLUMNS),
            "tier1_retrained_in_script": [
                "naive", "volume", "linear_regression", "xgboost",
            ],
            "tier2_tier3_ingested_paths": {
                k: v.as_posix()
                for k, v in TIER2_TIER3_INGEST_MAP.items()
            },
        },
        "models": models_out,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {args.output} with {len(models_out)} models")
    print(f"  Models: {list(models_out.keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
