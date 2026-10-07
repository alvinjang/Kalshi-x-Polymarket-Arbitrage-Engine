"""Live vs backtest reconciliation for closed prediction market positions.

This module provides pure-Python analysis functions that:

1. Load closed positions from the live SQLite position store.
2. Filter positions to a reconciliation window (April 11 onward, excluding
   any positions closed via force_close_schema_fix).
3. Run a shadow simulation: for each closed position, look up the entry bar
   in bars.parquet, run the deployed models, and compute what the backtest
   would have predicted and what P&L it would have realized.
4. Produce summary comparison tables, category breakdowns, and exit-reason
   attribution.
5. Apply an acceptance gate: raise ValueError if fewer than 80% of positions
   could be matched to a bar in bars.parquet.

Fee model note: Shadow-simulation P&L uses profit_sim.simulate_profit
(threshold-only model -- no explicit cost deduction).  Table 2 P&L in the
paper uses the verify_headline deduction model (2pp subtracted from each
winning trade).  The two are NOT directly comparable in absolute terms;
this module focuses on directional accuracy and tracking error.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.profit_sim import simulate_profit
from src.features.category import derive_category_from_ticker
from src.live.position_manager import PositionManager


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_closed_positions(db_path: str | Path) -> list[dict]:
    """Return all closed positions from the live position DB.

    Uses PositionManager.get_closed_positions() as the sole DB-access path
    (avoids raw sqlite3 in analysis code, preventing schema drift).

    Args:
        db_path: Path to positions.db on disk.

    Returns:
        List of dicts, one per closed position, newest first.
    """
    pm = PositionManager(db_path=str(db_path))
    return pm.get_closed_positions()


# ---------------------------------------------------------------------------
# 2. Window filter
# ---------------------------------------------------------------------------

def filter_window(
    positions: list[dict],
    start: str = "2026-04-11",
    end: str | None = None,
) -> list[dict]:
    """Filter positions to the reconciliation window.

    Exclusion rules:
    - exit_reason == "force_close_schema_fix"  (stale pair_id schema)
    - entry_time < start  (ISO 8601 comparison is safe; both are UTC strings)
    - entry_time > end  (if end provided)

    Args:
        positions: Raw list from load_closed_positions().
        start: ISO 8601 date string (e.g. "2026-04-11"). Positions with
            entry_time before this are excluded.
        end: Optional ISO 8601 date string. Positions with entry_time after
            this are excluded.

    Returns:
        Filtered list of position dicts.
    """
    result = []
    for pos in positions:
        if pos.get("exit_reason") == "force_close_schema_fix":
            continue
        entry_time = pos.get("entry_time", "")
        if entry_time < start:
            continue
        if end is not None and entry_time > end:
            continue
        result.append(pos)
    return result


# ---------------------------------------------------------------------------
# 3. Shadow simulation
# ---------------------------------------------------------------------------

def run_shadow_simulation(
    positions: list[dict],
    bars_path: str | Path,
    models_dir: str | Path,
) -> list[dict]:
    """Replay each closed position through the deployed models.

    For each position:
    1. Find the entry bar in bars.parquet at or just before entry_time.
    2. If no bar found, record matched=False and skip.
    3. Align bar columns to deployed feature_columns.json.
    4. Predict with LR and XGBoost models.
    5. Compute simulated P&L via simulate_profit (RECON-04 canonical fee fn).
    6. Return result dict with live_pnl, sim_pnl, predictions, metadata.

    Args:
        positions: Filtered list of closed position dicts.
        bars_path: Path to data/live/bars.parquet.
        models_dir: Path to models/deployed/ directory containing
            linear_regression.pkl, xgboost.pkl, feature_columns.json.

    Returns:
        List of result dicts (one per position). Each has matched=True or False.
    """
    from src.models.base import BasePredictor

    models_dir = Path(models_dir)
    bars_path = Path(bars_path)

    # Load bars once
    bars = pd.read_parquet(bars_path)

    # Load deployed models
    lr_model = BasePredictor.load(str(models_dir / "linear_regression.pkl"))
    xgb_model = BasePredictor.load(str(models_dir / "xgboost.pkl"))

    with open(models_dir / "feature_columns.json") as f:
        feature_columns: list[str] = json.load(f)

    results: list[dict] = []

    for pos in positions:
        pair_id = pos["pair_id"]
        entry_time_iso = pos.get("entry_time", "")
        category = derive_category_from_ticker(pos.get("kalshi_ticker", ""))

        # Convert ISO string to Unix timestamp for bar lookup
        try:
            entry_ts = int(pd.Timestamp(entry_time_iso).timestamp())
        except Exception:
            results.append(_unmatched_result(pos, category))
            continue

        # Find entry bar
        pair_bars = bars[
            (bars["pair_id"] == pair_id) & (bars["timestamp"] <= entry_ts)
        ]
        if pair_bars.empty:
            results.append(_unmatched_result(pos, category))
            continue

        entry_bar = pair_bars.nlargest(1, "timestamp").iloc[0]

        # Align to feature columns and impute NaN (rolling/diff ops on single
        # row will produce NaN; models require finite input)
        row_df = pd.DataFrame([entry_bar])
        available_cols = [c for c in feature_columns if c in row_df.columns]
        X = row_df[available_cols].fillna(0.0)

        if X.empty or X.shape[1] == 0:
            results.append(_unmatched_result(pos, category))
            continue

        # Predict
        try:
            lr_pred = float(lr_model.predict(X)[0])
            xgb_pred = float(xgb_model.predict(X)[0])
        except Exception:
            results.append(_unmatched_result(pos, category))
            continue

        avg_pred = (lr_pred + xgb_pred) / 2.0
        actual_change = float(pos["exit_spread"]) - float(pos["entry_spread"])

        # Simulated P&L using canonical threshold-only fee model (RECON-04)
        sim_result = simulate_profit(
            np.array([avg_pred]),
            np.array([actual_change]),
            threshold=0.02,
        )

        results.append({
            "pair_id": pair_id,
            "kalshi_ticker": pos.get("kalshi_ticker", ""),
            "direction": pos.get("direction", ""),
            "exit_reason": pos.get("exit_reason", ""),
            "category": category,
            "live_pnl": float(pos.get("realized_pnl", 0.0)),
            "sim_pnl": float(sim_result["total_pnl"]),
            "lr_pred": lr_pred,
            "xgb_pred": xgb_pred,
            "avg_pred": avg_pred,
            "actual_change": actual_change,
            "matched": True,
        })

    return results


def _unmatched_result(pos: dict, category: str) -> dict:
    """Return a result dict marking this position as unmatched."""
    return {
        "pair_id": pos["pair_id"],
        "kalshi_ticker": pos.get("kalshi_ticker", ""),
        "direction": pos.get("direction", ""),
        "exit_reason": pos.get("exit_reason", ""),
        "category": category,
        "live_pnl": float(pos.get("realized_pnl", 0.0)),
        "sim_pnl": 0.0,
        "lr_pred": None,
        "xgb_pred": None,
        "avg_pred": None,
        "actual_change": None,
        "matched": False,
    }


# ---------------------------------------------------------------------------
# 4. Summary table
# ---------------------------------------------------------------------------

def build_summary(matched_results: list[dict]) -> dict[str, Any]:
    """Build the top-level reconciliation summary.

    Args:
        matched_results: List of result dicts where matched=True.
            Caller should pass only matched results (filter on matched key).

    Returns:
        Dict with keys: live_total_pnl, sim_total_pnl, tracking_error,
        matched_count, unmatched_count, gap_metric.

        NOTE: unmatched_count and gap_metric default to 0 here.
        Caller should set summary["unmatched_count"] and
        summary["gap_metric"] after calling build_summary, based on
        total positions - matched positions.
    """
    live_total = sum(r["live_pnl"] for r in matched_results)
    sim_total = sum(r["sim_pnl"] for r in matched_results)
    matched_count = len(matched_results)

    return {
        "live_total_pnl": float(live_total),
        "sim_total_pnl": float(sim_total),
        "tracking_error": float(live_total - sim_total),
        "matched_count": matched_count,
        "unmatched_count": 0,   # caller must set based on all_results
        "gap_metric": 0.0,      # caller must set: unmatched / total
    }


# ---------------------------------------------------------------------------
# 5. Category breakdown
# ---------------------------------------------------------------------------

def category_breakdown(matched_results: list[dict]) -> dict[str, dict]:
    """Group matched results by asset-class category.

    Category is derived from derive_category_from_ticker(pos["kalshi_ticker"]).
    This is the CORRECT approach for live positions (RECON-06).

    Args:
        matched_results: List of matched result dicts.

    Returns:
        Dict mapping category name -> {live_pnl, sim_pnl, count, tracking_error}.
    """
    groups: dict[str, dict] = {}

    for r in matched_results:
        cat = r.get("category", "other")
        if cat not in groups:
            groups[cat] = {"live_pnl": 0.0, "sim_pnl": 0.0, "count": 0}
        groups[cat]["live_pnl"] += r["live_pnl"]
        groups[cat]["sim_pnl"] += r["sim_pnl"]
        groups[cat]["count"] += 1

    # Add tracking_error per category
    for cat in groups:
        groups[cat]["live_pnl"] = round(groups[cat]["live_pnl"], 6)
        groups[cat]["sim_pnl"] = round(groups[cat]["sim_pnl"], 6)
        groups[cat]["tracking_error"] = round(
            groups[cat]["live_pnl"] - groups[cat]["sim_pnl"], 6
        )

    return groups


# ---------------------------------------------------------------------------
# 6. Exit reason attribution
# ---------------------------------------------------------------------------

def exit_reason_attribution(matched_results: list[dict]) -> dict[str, dict]:
    """Group matched results by exit reason.

    Expected exit reasons: TIME_STOP, RESOLUTION_EXIT, MOMENTUM,
    STOP_LOSS, TAKE_PROFIT (and optionally MANUAL).

    Args:
        matched_results: List of matched result dicts.

    Returns:
        Dict mapping exit_reason -> {live_count, sim_count, live_pnl, sim_pnl}.
        sim_count == live_count since each matched result has exactly one sim.
    """
    groups: dict[str, dict] = {}

    for r in matched_results:
        reason = r.get("exit_reason", "UNKNOWN")
        if reason not in groups:
            groups[reason] = {
                "live_count": 0,
                "sim_count": 0,
                "live_pnl": 0.0,
                "sim_pnl": 0.0,
            }
        groups[reason]["live_count"] += 1
        groups[reason]["sim_count"] += 1
        groups[reason]["live_pnl"] += r["live_pnl"]
        groups[reason]["sim_pnl"] += r["sim_pnl"]

    # Round for clean JSON output
    for reason in groups:
        groups[reason]["live_pnl"] = round(groups[reason]["live_pnl"], 6)
        groups[reason]["sim_pnl"] = round(groups[reason]["sim_pnl"], 6)

    return groups


# ---------------------------------------------------------------------------
# 7. Acceptance gate
# ---------------------------------------------------------------------------

def acceptance_gate(matched: int, total: int) -> bool:
    """Verify that the shadow simulation achieved >= 80% bar coverage.

    Args:
        matched: Number of positions where a bar was found in bars.parquet.
        total: Total number of positions in the filtered window.

    Returns:
        True if matched/total >= 0.80.

    Raises:
        ValueError: If matched/total < 0.80, with a diagnostic message
            containing the actual percentage and the gap count.
    """
    if total == 0:
        return True  # Empty window is trivially "matched"

    ratio = matched / total
    if ratio >= 0.80:
        return True

    pct_str = f"{ratio:.1%}"
    raise ValueError(
        f"Reconciliation gap too large: {pct_str} matched "
        f"({total - matched} unmatched out of {total} total). "
        "Diagnose missing bars in bars.parquet."
    )
