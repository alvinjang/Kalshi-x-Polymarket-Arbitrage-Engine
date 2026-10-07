"""Trading profit simulation for spread-prediction strategies.

Given predicted spread changes and realized spread changes, this module
simulates a simple threshold-based trading strategy and reports P&L,
trade count, win rate, Sharpe ratio, and a cumulative P&L series for
equity-curve plotting.

The same simulator is used for regression baselines, time-series models,
and RL policies so results are directly comparable.

IMPORTANT: Sharpe ratio is computed as a TIME-SERIES metric over all bars
(including non-trading bars as zero return), not per-trade. This prevents
the inflation that comes from computing Sharpe only over trades.

Annualization uses 4-hour bars: 6 bars/day * 365 days/year = 2190 bars/year.
Prediction markets trade 24/7, so no trading-day adjustment.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import math

import numpy as np


# Annualization factor for 4-hour bars, 24/7 markets.
# 6 bars per day * 365 days per year = 2190 bars per year.
BARS_PER_YEAR = 6 * 365  # = 2190


def simulate_profit(
    predictions: np.ndarray,
    actuals: np.ndarray,
    threshold: float = 0.02,
    timestamps: np.ndarray | None = None,
) -> dict:
    """Simulate threshold-based directional trading.

    Strategy:
        For each timestep i:
          - If ``|predictions[i]| > threshold``, enter a trade.
          - Position direction = ``sign(predictions[i])``.
          - Profit = ``actuals[i] * sign(predictions[i])``.
        Otherwise, no trade (zero return for that bar).

    Args:
        predictions: Predicted spread changes, shape (n,).
        actuals: Realized spread changes, shape (n,).
        threshold: Minimum absolute predicted spread required to trade.
        timestamps: Per-row timestamps, shape (n,). When provided,
            bar_returns are aggregated by timestamp into a portfolio
            time series before computing Sharpe. This correctly handles
            panel data (multiple pairs at the same timestamp) instead
            of treating cross-pair observations as sequential bars.

    Returns:
        Dict with keys:
          - ``total_pnl`` (float): sum of per-trade P&L.
          - ``num_trades`` (int): number of trades taken.
          - ``win_rate`` (float): fraction of trades with positive P&L.
            0.0 if no trades.
          - ``sharpe_ratio`` (float): annualized time-series Sharpe over
            per-bar returns (including non-trading bars as 0). Uses
            sqrt(2190) for 4-hour bars, 24/7. 0.0 if std is 0.
          - ``sharpe_per_trade`` (float): UNANNUALIZED Sharpe of per-trade
            returns. Raw mean/std ratio. Useful for comparing trade quality.
          - ``pnl_series`` (list[float]): cumulative P&L across all bars
            (for equity-curve plotting).
    """
    predictions = np.asarray(predictions, dtype=float)
    actuals = np.asarray(actuals, dtype=float)

    if predictions.shape != actuals.shape:
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs actuals {actuals.shape}"
        )

    n = len(predictions)
    trade_mask = np.abs(predictions) > threshold
    num_trades = int(trade_mask.sum())

    if num_trades == 0:
        return {
            "total_pnl": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "sharpe_per_trade": 0.0,
            "pnl_series": [],
        }

    # Per-bar position: +1, -1, or 0 (no trade)
    positions = np.zeros(n)
    positions[trade_mask] = np.sign(predictions[trade_mask])

    # Per-bar returns (0 when not trading)
    bar_returns = positions * actuals

    # Per-trade returns (only for bars where we traded)
    trade_returns = bar_returns[trade_mask]

    total_pnl = float(trade_returns.sum())
    win_rate = float((trade_returns > 0.0).mean())

    # Cumulative P&L series across all bars (including zeros)
    pnl_series = np.cumsum(bar_returns).tolist()

    # Time-series Sharpe: panel-aware when timestamps are provided.
    #
    # For panel data (multiple pairs at the same timestamp), aggregate
    # bar_returns into DAILY portfolio returns (sum across all pairs and
    # intra-day bars), then compute annualized Sharpe on the daily series.
    # This matches standard practice in pairs-trading / statistical
    # arbitrage literature.
    #
    # Why daily (not per-4h-bar)?  The 4h bar Sharpe annualized by
    # sqrt(2190) assumes 2190 independent observations/year.  With 144
    # correlated pairs the effective sample size is much smaller.  Daily
    # aggregation is the standard granularity for reporting and comparison.
    TRADING_DAYS_PER_YEAR = 365  # prediction markets trade 24/7
    if timestamps is not None:
        timestamps = np.asarray(timestamps)
        # Convert epoch seconds → calendar day (integer day ordinal)
        days = (timestamps // 86400).astype(np.int64)
        unique_days = np.unique(days)
        if len(unique_days) < 2:
            sharpe_ratio = 0.0
        else:
            # Daily portfolio return = sum of bar_returns across all
            # pairs and intra-day bars for that calendar day.
            daily_returns = np.array(
                [bar_returns[days == d].sum() for d in unique_days]
            )
            day_mean = float(daily_returns.mean())
            day_std = float(daily_returns.std(ddof=1))
            if day_std == 0.0:
                sharpe_ratio = 0.0
            else:
                sharpe_ratio = (
                    day_mean / day_std * math.sqrt(TRADING_DAYS_PER_YEAR)
                )
    else:
        # Fallback for single-asset or when timestamps unavailable
        bar_std = float(bar_returns.std())
        if bar_std == 0.0 or n < 2:
            sharpe_ratio = 0.0
        else:
            bar_mean = float(bar_returns.mean())
            sharpe_ratio = bar_mean / bar_std * math.sqrt(BARS_PER_YEAR)

    # Per-trade Sharpe (unannualized, for reference)
    if num_trades < 2:
        sharpe_per_trade = 0.0
    else:
        trade_std = float(trade_returns.std())
        if trade_std == 0.0:
            sharpe_per_trade = 0.0
        else:
            trade_mean = float(trade_returns.mean())
            sharpe_per_trade = trade_mean / trade_std

    return {
        "total_pnl": total_pnl,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "sharpe_per_trade": sharpe_per_trade,
        "pnl_series": pnl_series,
    }
