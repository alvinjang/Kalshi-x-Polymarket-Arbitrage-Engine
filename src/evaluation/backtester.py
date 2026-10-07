"""Walk-forward portfolio backtester with realistic transaction costs.

Replaces the inflated Sharpe ratios from ``profit_sim.py`` (which uses raw
spread units without capital normalization) with honest, paper-ready metrics.

The backtester steps through test bars chronologically, applies model
predictions with a threshold-based entry rule, computes dollar P&L after
entry/exit fees, and aggregates to daily percentage returns for proper
annualized Sharpe computation.

Key parameters (from CONTEXT.md):
    - Initial capital: $10,000
    - Position size: $100 per trade
    - Entry cost: 3pp of contract price
    - Exit cost: 2pp of contract price
    - Annualization: sqrt(365) on daily returns (24/7 markets)
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class WalkForwardBacktester:
    """Walk-forward portfolio simulator with transaction costs.

    Processes test data bar-by-bar in timestamp order, entering trades
    when ``|prediction| > threshold``.  Each position is held for one bar
    (4 hours) and closed at bar end.  P&L is capital-normalized to produce
    realistic Sharpe ratios.

    Args:
        initial_capital: Starting portfolio value in dollars.
        position_size: Dollar amount invested per trade.
        entry_cost_pp: Entry cost in percentage points of contract price.
        exit_cost_pp: Exit cost in percentage points of contract price.
        threshold: Minimum absolute prediction required to enter a trade.
    """

    initial_capital: float = 10_000.0
    position_size: float = 100.0
    entry_cost_pp: float = 0.03
    exit_cost_pp: float = 0.02
    threshold: float = 0.02

    def run(self, df: pd.DataFrame, predictions: np.ndarray) -> dict:
        """Run the walk-forward backtest.

        Args:
            df: DataFrame with columns ``timestamp``, ``pair_id``, ``spread``,
                ``kalshi_close``, ``polymarket_close``.  Optionally contains
                ``spread_change_target``; if missing, it is computed inline.
            predictions: 1-D array of predicted spread changes, aligned
                row-for-row with *df*.

        Returns:
            Dict with keys: ``num_trades``, ``total_pnl``, ``total_fees``,
            ``annualized_sharpe``, ``max_drawdown``, ``calmar_ratio``,
            ``win_rate``, ``avg_trade_duration_hours``, ``equity_curve``,
            ``daily_returns``, ``trade_log``.
        """
        predictions = np.asarray(predictions, dtype=float)
        df = df.copy().reset_index(drop=True)

        # Compute target if not present
        if "spread_change_target" not in df.columns:
            df["spread_change_target"] = (
                df.groupby("pair_id")["spread"].shift(-1) - df["spread"]
            )
            # Drop NaN targets and corresponding predictions
            valid_mask = df["spread_change_target"].notna()
            df = df.loc[valid_mask].reset_index(drop=True)
            predictions = predictions[valid_mask.values]

        # Sort by timestamp (primary), pair_id (secondary)
        sort_order = df[["timestamp", "pair_id"]].apply(tuple, axis=1).argsort()
        df = df.iloc[sort_order].reset_index(drop=True)
        predictions = predictions[sort_order]

        # Walk through each row
        trade_log: list[dict] = []

        for i in range(len(df)):
            pred = predictions[i]
            if abs(pred) <= self.threshold:
                continue

            row = df.iloc[i]
            ts = row["timestamp"]
            pair_id = row["pair_id"]
            actual_change = row["spread_change_target"]

            # Mid-price for contract sizing
            kalshi_close = row.get("kalshi_close", 0.50)
            polymarket_close = row.get("polymarket_close", 0.50)
            mid_price = (kalshi_close + polymarket_close) / 2.0

            if mid_price <= 0:
                continue

            direction = 1.0 if pred > 0 else -1.0
            num_contracts = self.position_size / mid_price

            gross_pnl = num_contracts * actual_change * direction
            entry_cost = num_contracts * self.entry_cost_pp
            exit_cost = num_contracts * self.exit_cost_pp
            net_pnl = gross_pnl - entry_cost - exit_cost

            trade_log.append({
                "timestamp": int(ts),
                "pair_id": pair_id,
                "direction": direction,
                "num_contracts": num_contracts,
                "gross_pnl": gross_pnl,
                "entry_cost": entry_cost,
                "exit_cost": exit_cost,
                "net_pnl": net_pnl,
            })

        num_trades = len(trade_log)

        if num_trades == 0:
            return self._empty_result()

        # Aggregate to daily returns
        total_pnl = sum(t["net_pnl"] for t in trade_log)
        total_fees = sum(t["entry_cost"] + t["exit_cost"] for t in trade_log)
        win_count = sum(1 for t in trade_log if t["net_pnl"] > 0)
        win_rate = win_count / num_trades

        # Group by calendar day (timestamp // 86400)
        daily_dollar: dict[int, float] = {}
        for t in trade_log:
            day = t["timestamp"] // 86400
            daily_dollar[day] = daily_dollar.get(day, 0.0) + t["net_pnl"]

        # Build daily returns as percentage of portfolio value at start of day
        sorted_days = sorted(daily_dollar.keys())
        portfolio_value = self.initial_capital
        daily_returns: list[float] = []
        equity_curve: list[tuple[int, float]] = [(sorted_days[0] - 1, portfolio_value)]

        for day in sorted_days:
            dollar_return = daily_dollar[day]
            pct_return = dollar_return / portfolio_value
            daily_returns.append(pct_return)
            portfolio_value += dollar_return
            equity_curve.append((day, portfolio_value))

        # Sharpe ratio: mean(daily_pct) / std(daily_pct, ddof=1) * sqrt(365)
        annualized_sharpe = self._compute_sharpe(daily_returns)

        # Max drawdown from equity curve
        max_drawdown = self._compute_max_drawdown(equity_curve)

        # Calmar ratio: annualized return / max drawdown
        calmar_ratio = self._compute_calmar(
            equity_curve, max_drawdown, len(sorted_days)
        )

        return {
            "num_trades": num_trades,
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "annualized_sharpe": annualized_sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "win_rate": win_rate,
            "avg_trade_duration_hours": 4.0,
            "equity_curve": equity_curve,
            "daily_returns": daily_returns,
            "trade_log": trade_log,
        }

    def _empty_result(self) -> dict:
        """Return zero-valued result dict when no trades are taken."""
        return {
            "num_trades": 0,
            "total_pnl": 0.0,
            "total_fees": 0.0,
            "annualized_sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "avg_trade_duration_hours": 4.0,
            "equity_curve": [],
            "daily_returns": [],
            "trade_log": [],
        }

    @staticmethod
    def _compute_sharpe(daily_returns: list[float]) -> float:
        """Annualized Sharpe from daily percentage returns."""
        if len(daily_returns) < 2:
            return 0.0
        arr = np.array(daily_returns)
        mean_ret = float(arr.mean())
        std_ret = float(arr.std(ddof=1))
        if std_ret == 0.0:
            return 0.0
        return mean_ret / std_ret * math.sqrt(365)

    @staticmethod
    def _compute_max_drawdown(
        equity_curve: list[tuple[int, float]],
    ) -> float:
        """Peak-to-trough percentage decline of the equity curve."""
        if len(equity_curve) < 2:
            return 0.0
        values = [v for _, v in equity_curve]
        peak = values[0]
        max_dd = 0.0
        for v in values[1:]:
            if v > peak:
                peak = v
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def _compute_calmar(
        equity_curve: list[tuple[int, float]],
        max_drawdown: float,
        n_days: int,
    ) -> float:
        """Calmar ratio = annualized return / max drawdown."""
        if max_drawdown == 0.0 or n_days == 0:
            return 0.0
        start_value = equity_curve[0][1]
        end_value = equity_curve[-1][1]
        total_return = (end_value - start_value) / start_value
        # Guard: if portfolio lost everything (total_return <= -1),
        # the base (1+total_return) is <= 0 and fractional exponent
        # produces NaN.  Return negative infinity-like value instead.
        base = 1.0 + total_return
        if base <= 0:
            return -999.0
        # Annualize: (1 + total_return)^(365/n_days) - 1
        annualized_return = base ** (365.0 / n_days) - 1.0
        return annualized_return / max_drawdown


def compute_break_even_fee(
    df: pd.DataFrame,
    predictions: np.ndarray,
    initial_capital: float = 10_000,
    position_size: float = 100,
    threshold: float = 0.02,
) -> float:
    """Binary search for the round-trip fee level where Sharpe drops to ~0.

    Searches from 0pp to 20pp total round-trip cost. Returns the break-even
    fee in percentage points (e.g., 0.05 means 5pp total).

    Args:
        df: Test DataFrame (same schema as ``WalkForwardBacktester.run``).
        predictions: Predicted spread changes aligned with *df*.
        initial_capital: Starting capital.
        position_size: Dollar amount per trade.
        threshold: Minimum prediction magnitude to trade.

    Returns:
        Break-even round-trip fee in percentage points.  Returns 0.0 if
        Sharpe is already <= 0 at zero fees.
    """
    # First check if Sharpe is positive at zero cost
    bt_zero = WalkForwardBacktester(
        initial_capital=initial_capital,
        position_size=position_size,
        entry_cost_pp=0.0,
        exit_cost_pp=0.0,
        threshold=threshold,
    )
    result_zero = bt_zero.run(df, predictions)
    if result_zero["annualized_sharpe"] <= 0:
        return 0.0

    low = 0.0
    high = 0.20  # 20pp max
    for _ in range(50):  # 50 iterations gives ~1e-15 precision
        mid = (low + high) / 2.0
        # Split round-trip into 60% entry / 40% exit (matching 3:2 ratio)
        entry_pp = mid * 0.6
        exit_pp = mid * 0.4
        bt = WalkForwardBacktester(
            initial_capital=initial_capital,
            position_size=position_size,
            entry_cost_pp=entry_pp,
            exit_cost_pp=exit_pp,
            threshold=threshold,
        )
        result = bt.run(df, predictions)
        if result["annualized_sharpe"] > 0:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0
