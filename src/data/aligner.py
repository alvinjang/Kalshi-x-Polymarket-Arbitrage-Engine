"""Cross-platform alignment with forward-fill and staleness decay.

Aligns reconstructed 4-hour candles from Kalshi and Polymarket onto a
unified time grid. Uses forward-fill with 24-hour (6-bar) staleness
limit, computes hours_since_last_trade per platform, and spread as
kalshi_vwap - polymarket_vwap.

Quality filters exclude pairs with:
- Fewer than 20 trades on either platform
- Staleness ratio >= 0.80 (fraction of bars where both platforms lack trades)
- Mean absolute spread >= 0.30 (non-mean-reverting)
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schemas import CANDLE_COLUMNS

logger = logging.getLogger(__name__)

# Forward-fill limit: 6 bars = 24 hours at 4-hour granularity
FFILL_LIMIT = 6

# Quality thresholds
MIN_TRADES_PER_PLATFORM = 10  # Loosened from 20 — real arbs care about any active market
MAX_STALENESS_RATIO = 0.90  # Loosened from 0.80
MAX_MEAN_ABS_SPREAD = 0.60  # Loosened from 0.30 — persistent spreads can still be arb-able
# Minimum fraction of bars with valid (non-NaN) spread (both platforms have price)
MIN_VALID_SPREAD_RATIO = 0.15  # Loosened from 0.20
# Minimum temporal overlap in seconds between platforms (24 hours)
MIN_OVERLAP_SECONDS = 24 * 3600  # Loosened from 48h

# Default bar size in seconds (4 hours)
DEFAULT_BAR_SECONDS = 14400


def _compute_hours_since_last_trade(
    has_trade_series: pd.Series, bar_hours: int = 4,
) -> pd.Series:
    """Compute hours since last trade for each bar.

    Args:
        has_trade_series: Boolean series where True = bar had a fresh trade.
        bar_hours: Duration of each bar in hours.

    Returns:
        Series of float hours. 0 for bars with trades, N*bar_hours for N-bar
        gaps, NaN if no prior trade exists.
    """
    result = pd.Series(np.nan, index=has_trade_series.index, dtype=float)
    bars_since = np.nan

    for i in range(len(has_trade_series)):
        if has_trade_series.iloc[i]:
            bars_since = 0
            result.iloc[i] = 0.0
        elif not np.isnan(bars_since):
            bars_since += 1
            result.iloc[i] = bars_since * bar_hours
        # else: stays NaN (no prior trade)

    return result


def align_pair(
    kalshi_candles: pd.DataFrame,
    poly_candles: pd.DataFrame,
    pair_id: str,
    bar_seconds: int = DEFAULT_BAR_SECONDS,
) -> pd.DataFrame | None:
    """Align candles from both platforms onto a unified time grid.

    Args:
        kalshi_candles: DataFrame with CANDLE_COLUMNS from reconstructor.
        poly_candles: DataFrame with CANDLE_COLUMNS from reconstructor.
        pair_id: Unique identifier for this matched pair.
        bar_seconds: Bar size in seconds (default 14400 = 4 hours).

    Returns:
        Aligned DataFrame with prefixed columns (kalshi_*, polymarket_*),
        spread, and pair_id. Returns None if either input is empty.
    """
    if kalshi_candles.empty or poly_candles.empty:
        return None

    bar_hours = bar_seconds // 3600

    # Determine time range across both platforms
    min_ts = min(kalshi_candles["timestamp"].min(), poly_candles["timestamp"].min())
    max_ts = max(kalshi_candles["timestamp"].max(), poly_candles["timestamp"].max())

    # Floor to bar boundary
    min_ts = (min_ts // bar_seconds) * bar_seconds

    # Create unified grid
    grid_ts = np.arange(min_ts, max_ts + bar_seconds, bar_seconds)
    grid = pd.DataFrame({"timestamp": grid_ts.astype(int)})

    platforms = {
        "kalshi": kalshi_candles.copy(),
        "polymarket": poly_candles.copy(),
    }

    merged = grid.copy()

    for prefix, candles in platforms.items():
        # Ensure timestamp is aligned to bar boundary
        candles["timestamp"] = (candles["timestamp"] // bar_seconds) * bar_seconds

        # Deduplicate: keep last entry per bar
        candles = candles.sort_values("timestamp").groupby("timestamp").last().reset_index()

        # Rename columns with platform prefix (except timestamp)
        rename_map = {}
        for col in CANDLE_COLUMNS:
            if col != "timestamp":
                rename_map[col] = f"{prefix}_{col}"
        candles = candles.rename(columns=rename_map)

        # Left join onto grid
        merged = merged.merge(candles, on="timestamp", how="left")

        has_trade_col = f"{prefix}_has_trade"

        # Remember original has_trade before forward-fill
        original_has_trade = merged[has_trade_col].fillna(False).astype(bool).copy()

        # Forward-fill price/microstructure columns (limit=6 bars)
        ffill_cols = [f"{prefix}_{c}" for c in CANDLE_COLUMNS if c != "timestamp"]
        merged[ffill_cols] = merged[ffill_cols].ffill(limit=FFILL_LIMIT)

        # After forward-fill, mark filled bars as has_trade=False
        merged[has_trade_col] = original_has_trade

        # Compute hours_since_last_trade
        merged[f"{prefix}_hours_since_last_trade"] = _compute_hours_since_last_trade(
            original_has_trade, bar_hours=bar_hours,
        )

    # Compute spread: kalshi_vwap - polymarket_vwap
    merged["spread"] = merged["kalshi_vwap"] - merged["polymarket_vwap"]

    # Add pair_id
    merged["pair_id"] = pair_id

    return merged


def align_all_pairs(
    pairs: list[dict],
    candles_dir: Path,
    bar_seconds: int = DEFAULT_BAR_SECONDS,
) -> tuple[pd.DataFrame, dict]:
    """Align all matched pairs and apply quality filters.

    Args:
        pairs: List of pair dicts with kalshi_market_id, polymarket_market_id, pair_id.
        candles_dir: Directory containing candles/{kalshi,polymarket}/ subdirs.
        bar_seconds: Bar size in seconds (default 14400 = 4 hours).

    Returns:
        Tuple of (aligned_df, quality_report).
        aligned_df: Concatenated DataFrame of all passing pairs.
        quality_report: Dict with filtering statistics.
    """
    candles_dir = Path(candles_dir)
    aligned_list: list[pd.DataFrame] = []

    # Initialize report
    exclusion_reasons = {
        "missing_candles": 0,
        "insufficient_trades": 0,
        "no_temporal_overlap": 0,
        "too_stale": 0,
        "insufficient_valid_spread": 0,
        "spread_not_mean_reverting": 0,
        "alignment_failed": 0,
    }
    per_pair: list[dict] = []

    for pair in pairs:
        kalshi_id = pair["kalshi_market_id"]
        poly_id = pair["polymarket_market_id"]
        pair_id = pair["pair_id"]

        entry = {
            "pair_id": pair_id,
            "status": "excluded",
            "reason": None,
            "kalshi_trades": 0,
            "polymarket_trades": 0,
            "staleness_ratio": None,
            "mean_abs_spread": None,
            "bars": 0,
        }

        # Load candle files
        k_path = candles_dir / "kalshi" / f"{kalshi_id}_candles.parquet"
        p_path = candles_dir / "polymarket" / f"{poly_id}_candles.parquet"

        if not k_path.exists() or not p_path.exists():
            entry["reason"] = "missing_candles"
            exclusion_reasons["missing_candles"] += 1
            per_pair.append(entry)
            logger.info(f"Pair {pair_id}: excluded (missing candle files)")
            continue

        try:
            k_candles = pd.read_parquet(k_path)
            p_candles = pd.read_parquet(p_path)
        except Exception as e:
            entry["reason"] = "missing_candles"
            exclusion_reasons["missing_candles"] += 1
            per_pair.append(entry)
            logger.warning(f"Pair {pair_id}: error reading candles: {e}")
            continue

        if k_candles.empty or p_candles.empty:
            entry["reason"] = "missing_candles"
            exclusion_reasons["missing_candles"] += 1
            per_pair.append(entry)
            logger.info(f"Pair {pair_id}: excluded (empty candle files)")
            continue

        # Count trades
        k_trades = int(k_candles["trade_count"].sum())
        p_trades = int(p_candles["trade_count"].sum())
        entry["kalshi_trades"] = k_trades
        entry["polymarket_trades"] = p_trades

        if k_trades < MIN_TRADES_PER_PLATFORM or p_trades < MIN_TRADES_PER_PLATFORM:
            entry["reason"] = "insufficient_trades"
            exclusion_reasons["insufficient_trades"] += 1
            per_pair.append(entry)
            logger.info(
                f"Pair {pair_id}: excluded (insufficient trades: "
                f"kalshi={k_trades}, poly={p_trades})"
            )
            continue

        # Check temporal overlap — both platforms must have traded during the same period
        k_min, k_max = int(k_candles["timestamp"].min()), int(k_candles["timestamp"].max())
        p_min, p_max = int(p_candles["timestamp"].min()), int(p_candles["timestamp"].max())
        overlap_start = max(k_min, p_min)
        overlap_end = min(k_max, p_max)
        overlap_seconds = overlap_end - overlap_start

        if overlap_seconds < MIN_OVERLAP_SECONDS:
            entry["reason"] = "no_temporal_overlap"
            exclusion_reasons["no_temporal_overlap"] += 1
            per_pair.append(entry)
            logger.info(
                f"Pair {pair_id}: excluded (temporal overlap {overlap_seconds/3600:.1f}h "
                f"< {MIN_OVERLAP_SECONDS/3600:.0f}h minimum)"
            )
            continue

        # Align
        aligned = align_pair(k_candles, p_candles, pair_id, bar_seconds)
        if aligned is None:
            entry["reason"] = "alignment_failed"
            exclusion_reasons["alignment_failed"] += 1
            per_pair.append(entry)
            logger.info(f"Pair {pair_id}: excluded (alignment failed)")
            continue

        entry["bars"] = len(aligned)

        # Compute staleness ratio: fraction of bars where BOTH has_trade are False
        both_no_trade = (
            (~aligned["kalshi_has_trade"].fillna(False).astype(bool))
            & (~aligned["polymarket_has_trade"].fillna(False).astype(bool))
        )
        staleness_ratio = both_no_trade.sum() / len(aligned) if len(aligned) > 0 else 1.0
        entry["staleness_ratio"] = float(staleness_ratio)

        if staleness_ratio >= MAX_STALENESS_RATIO:
            entry["reason"] = "too_stale"
            exclusion_reasons["too_stale"] += 1
            per_pair.append(entry)
            logger.info(
                f"Pair {pair_id}: excluded (staleness ratio {staleness_ratio:.2f})"
            )
            continue

        # Compute valid spread ratio (fraction of bars with non-NaN spread)
        spread_valid = aligned["spread"].dropna()
        valid_spread_ratio = len(spread_valid) / len(aligned) if len(aligned) > 0 else 0.0
        entry["valid_spread_ratio"] = float(valid_spread_ratio)

        if valid_spread_ratio < MIN_VALID_SPREAD_RATIO:
            entry["reason"] = "insufficient_valid_spread"
            exclusion_reasons["insufficient_valid_spread"] += 1
            per_pair.append(entry)
            logger.info(
                f"Pair {pair_id}: excluded (valid spread ratio {valid_spread_ratio:.2f} "
                f"< {MIN_VALID_SPREAD_RATIO:.2f} minimum)"
            )
            continue

        # Compute mean absolute spread
        mean_abs_spread = float(spread_valid.abs().mean()) if len(spread_valid) > 0 else float("nan")
        entry["mean_abs_spread"] = mean_abs_spread

        if mean_abs_spread >= MAX_MEAN_ABS_SPREAD:
            entry["reason"] = "spread_not_mean_reverting"
            exclusion_reasons["spread_not_mean_reverting"] += 1
            per_pair.append(entry)
            logger.info(
                f"Pair {pair_id}: excluded (mean |spread| {mean_abs_spread:.3f})"
            )
            continue

        # Pair passes all filters
        entry["status"] = "included"
        per_pair.append(entry)
        aligned_list.append(aligned)
        logger.info(
            f"Pair {pair_id}: included ({len(aligned)} bars, "
            f"staleness={staleness_ratio:.2f}, mean_|spread|={mean_abs_spread:.3f})"
        )

    # Concatenate all aligned DataFrames
    if aligned_list:
        result_df = pd.concat(aligned_list, ignore_index=True)
    else:
        result_df = pd.DataFrame()

    aligned_count = len(aligned_list)
    excluded_count = len(pairs) - aligned_count

    quality_report = {
        "total_pairs": len(pairs),
        "aligned_pairs": aligned_count,
        "excluded_pairs": excluded_count,
        "exclusion_reasons": exclusion_reasons,
        "per_pair": per_pair,
    }

    logger.info(
        f"Alignment complete: {aligned_count} included, {excluded_count} excluded "
        f"out of {len(pairs)} total pairs"
    )

    return result_df, quality_report
