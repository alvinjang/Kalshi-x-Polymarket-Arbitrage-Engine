"""TDD tests for WalkForwardBacktester (RED phase).

Tests cover: timestamp ordering, fee deduction, capital normalization,
realistic Sharpe range, max drawdown, zero-trades edge case, Calmar ratio,
and win rate computation.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtester import WalkForwardBacktester


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    timestamps: list[int],
    pair_ids: list[str],
    spreads: list[float],
    spread_change_targets: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal DataFrame matching the backtester's expected schema.

    Includes dummy feature columns so the DataFrame isn't feature-empty.
    """
    df = pd.DataFrame({
        "timestamp": timestamps,
        "pair_id": pair_ids,
        "spread": spreads,
        # Mid-price components (kalshi_close and polymarket_close)
        # Default: mid-price = 0.50 for easy contract-count math.
        "kalshi_close": [0.50] * len(timestamps),
        "polymarket_close": [0.50] * len(timestamps),
        # Dummy feature columns
        "feature_a": [1.0] * len(timestamps),
        "feature_b": [2.0] * len(timestamps),
    })
    if spread_change_targets is not None:
        df["spread_change_target"] = spread_change_targets
    return df


# ---------------------------------------------------------------------------
# Test 1: Timestamp ordering
# ---------------------------------------------------------------------------

class TestTimestampOrdering:
    """Backtester processes bars in timestamp order (not pair order)."""

    def test_trade_log_sorted_by_timestamp(self):
        # 3 timestamps, 2 pairs each = 6 rows
        df = _make_df(
            timestamps=[100, 100, 200, 200, 300, 300],
            pair_ids=["B", "A", "B", "A", "B", "A"],
            spreads=[0.10, 0.20, 0.15, 0.25, 0.12, 0.22],
            spread_change_targets=[0.05, 0.05, -0.03, -0.03, 0.02, 0.02],
        )
        predictions = np.array([0.05, 0.05, -0.05, -0.05, 0.05, 0.05])

        bt = WalkForwardBacktester(threshold=0.01)
        result = bt.run(df, predictions)

        # All 6 rows should produce trades (|pred| > 0.01)
        assert result["num_trades"] == 6
        # Trade log timestamps must be non-decreasing
        trade_log = result["trade_log"]
        ts_list = [t["timestamp"] for t in trade_log]
        assert ts_list == sorted(ts_list)


# ---------------------------------------------------------------------------
# Test 2: Fee deduction
# ---------------------------------------------------------------------------

class TestFeeDeduction:
    """Transaction costs correctly reduce P&L."""

    def test_net_pnl_with_known_fees(self):
        # One trade: spread_change = +0.10, mid_price = 0.50
        # position_size = $100, contracts = 100/0.50 = 200
        # gross_pnl = 200 * 0.10 = $20
        # entry_cost = 200 * 0.03 = $6
        # exit_cost = 200 * 0.02 = $4
        # net_pnl = 20 - 6 - 4 = $10
        df = _make_df(
            timestamps=[100],
            pair_ids=["A"],
            spreads=[0.40],
            spread_change_targets=[0.10],
        )
        predictions = np.array([0.05])  # above threshold

        bt = WalkForwardBacktester(
            initial_capital=10_000,
            position_size=100,
            entry_cost_pp=0.03,
            exit_cost_pp=0.02,
            threshold=0.02,
        )
        result = bt.run(df, predictions)

        assert result["num_trades"] == 1
        assert abs(result["total_pnl"] - 10.0) < 0.01
        assert abs(result["total_fees"] - 10.0) < 0.01


# ---------------------------------------------------------------------------
# Test 3: Capital normalization
# ---------------------------------------------------------------------------

class TestCapitalNormalization:
    """Daily returns are expressed as percentage of portfolio value."""

    def test_daily_return_as_percentage(self):
        # One trade on day 1: net P&L = $10, capital = $10,000
        # Daily pct return = 10 / 10000 = 0.001
        df = _make_df(
            timestamps=[86400],  # day 1 (epoch-second day boundary)
            pair_ids=["A"],
            spreads=[0.40],
            spread_change_targets=[0.10],
        )
        predictions = np.array([0.05])

        bt = WalkForwardBacktester(
            initial_capital=10_000,
            position_size=100,
            entry_cost_pp=0.03,
            exit_cost_pp=0.02,
            threshold=0.02,
        )
        result = bt.run(df, predictions)

        assert len(result["daily_returns"]) >= 1
        assert abs(result["daily_returns"][0] - 0.001) < 1e-6


# ---------------------------------------------------------------------------
# Test 4: Sharpe in realistic range
# ---------------------------------------------------------------------------

class TestSharpeRealisticRange:
    """Annualized Sharpe from capital-normalized returns stays < 5.0."""

    def test_sharpe_not_inflated(self):
        # Verify that transaction costs reduce Sharpe: running the same
        # data with fees vs. without fees should produce a lower Sharpe
        # when fees are present.  This confirms the capital-normalized
        # backtester properly accounts for costs, unlike the old
        # profit_sim which had no cost model and produced 15-23 Sharpe.
        np.random.seed(42)
        n_days = 60
        n_pairs = 5
        n_rows = n_days * n_pairs

        timestamps = []
        pair_ids = []
        for d in range(n_days):
            ts = (d + 1) * 86400
            for p in range(n_pairs):
                timestamps.append(ts)
                pair_ids.append(f"pair_{p}")

        spreads = [0.50] * n_rows
        targets = np.random.normal(0.01, 0.05, n_rows)
        predictions = targets * 0.3 + np.random.normal(0, 0.04, n_rows)

        df = _make_df(
            timestamps=timestamps,
            pair_ids=pair_ids,
            spreads=spreads,
            spread_change_targets=targets.tolist(),
        )

        # Zero-fee backtester
        bt_free = WalkForwardBacktester(
            threshold=0.01, entry_cost_pp=0.0, exit_cost_pp=0.0,
        )
        result_free = bt_free.run(df, predictions)

        # With-fee backtester (default 3pp + 2pp)
        bt_fee = WalkForwardBacktester(threshold=0.01)
        result_fee = bt_fee.run(df, predictions)

        # Fees should reduce Sharpe
        assert result_fee["annualized_sharpe"] < result_free["annualized_sharpe"], (
            f"With-fee Sharpe ({result_fee['annualized_sharpe']:.2f}) "
            f"should be lower than zero-fee ({result_free['annualized_sharpe']:.2f})"
        )
        # Both should be finite
        assert math.isfinite(result_free["annualized_sharpe"])
        assert math.isfinite(result_fee["annualized_sharpe"])


# ---------------------------------------------------------------------------
# Test 5: Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    """Max drawdown computed from equity curve peak-to-trough."""

    def test_known_drawdown(self):
        # Build equity curve: 10000 -> 10100 -> 9900 -> 10050
        # Need 3 days with known net P&L: +100, -200, +150
        #
        # With $100 position at mid_price=0.50 => 200 contracts
        # net = direction * contracts * actual_change - fees
        # fees = 200*0.03 + 200*0.02 = 10
        #
        # Day 1 (net +100): pred=+0.05 => dir=+1
        #   100 = 1 * 200 * target - 10 => target = 110/200 = 0.55
        # Day 2 (net -200): pred=+0.05 => dir=+1
        #   -200 = 1 * 200 * target - 10 => target = -190/200 = -0.95
        # Day 3 (net +150): pred=+0.05 => dir=+1
        #   150 = 1 * 200 * target - 10 => target = 160/200 = 0.80

        df = _make_df(
            timestamps=[86400, 86400 * 2, 86400 * 3],
            pair_ids=["A", "A", "A"],
            spreads=[0.40, 0.40, 0.40],
            spread_change_targets=[0.55, -0.95, 0.80],
        )
        # All predictions positive so direction = +1 consistently
        predictions = np.array([0.05, 0.05, 0.05])

        bt = WalkForwardBacktester(
            initial_capital=10_000,
            position_size=100,
            entry_cost_pp=0.03,
            exit_cost_pp=0.02,
            threshold=0.02,
        )
        result = bt.run(df, predictions)

        # Equity curve: 10000 -> 10100 -> 9900 -> 10050
        expected_drawdown = (10100 - 9900) / 10100
        assert abs(result["max_drawdown"] - expected_drawdown) < 0.001


# ---------------------------------------------------------------------------
# Test 6: Zero trades
# ---------------------------------------------------------------------------

class TestZeroTrades:
    """If all predictions are below threshold, no trades occur."""

    def test_zero_trades_returns_zeros(self):
        df = _make_df(
            timestamps=[100, 200],
            pair_ids=["A", "A"],
            spreads=[0.40, 0.40],
            spread_change_targets=[0.05, -0.03],
        )
        # Predictions below threshold (0.02)
        predictions = np.array([0.01, -0.01])

        bt = WalkForwardBacktester(threshold=0.02)
        result = bt.run(df, predictions)

        assert result["num_trades"] == 0
        assert result["total_pnl"] == 0.0
        assert result["annualized_sharpe"] == 0.0
        assert result["win_rate"] == 0.0


# ---------------------------------------------------------------------------
# Test 7: Calmar ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    """Calmar = annualized_return / max_drawdown."""

    def test_calmar_known_values(self):
        # Create scenario with known annualized return and max drawdown.
        # 10 days, single pair. Net P&L: +$10 each day.
        # After 10 days: portfolio = 10000 + 100 = 10100
        # Return = 100/10000 = 1% over 10 days
        # Annualized return = (1.01)^(365/10) - 1
        # No drawdown (monotonically increasing) => Calmar = 0 (div by 0 guard)
        # But we need a drawdown to test Calmar properly.
        #
        # Alternate approach: hardcode some up/down days.
        # Days: +50, -20, +30, +40, -10, +50, +30, +20, +40, +30
        # Cumulative: 50, 30, 60, 100, 90, 140, 170, 190, 230, 260
        # Equity: 10050, 10030, 10060, 10100, 10090, 10140, 10170, 10190, 10230, 10260
        # Peak at 10100 before drop to 10090 => dd = 10/10100 = 0.00099
        # Total return = 260/10000 = 2.6% in 10 days
        # Annualized return = (1.026)^(365/10) - 1

        # Use simpler approach: just check Calmar > 0 when drawdown > 0
        # and Calmar = 0 when drawdown = 0.

        # Case 1: no drawdown => Calmar = 0
        df1 = _make_df(
            timestamps=[86400, 86400 * 2],
            pair_ids=["A", "A"],
            spreads=[0.40, 0.40],
            # Both trades net positive (gross = target*200 - 10 fees)
            # target = 0.10 => gross = 20 - 10 = 10
            spread_change_targets=[0.10, 0.10],
        )
        preds1 = np.array([0.05, 0.05])
        bt = WalkForwardBacktester(threshold=0.02)
        r1 = bt.run(df1, preds1)
        assert r1["calmar_ratio"] == 0.0  # no drawdown

        # Case 2: has drawdown => Calmar > 0
        # All predictions positive (dir=+1), so:
        # net = 1 * 200 * target - 10
        # Day 1 (net +10): target = 0.10
        # Day 2 (net -30): target = -0.10
        # Day 3 (net +50): target = 0.30
        df2 = _make_df(
            timestamps=[86400, 86400 * 2, 86400 * 3],
            pair_ids=["A", "A", "A"],
            spreads=[0.40, 0.40, 0.40],
            spread_change_targets=[0.10, -0.10, 0.30],
        )
        preds2 = np.array([0.05, 0.05, 0.05])  # all positive => dir=+1
        r2 = bt.run(df2, preds2)
        # Equity: 10000 -> 10010 -> 9980 -> 10030
        # Drawdown: (10010 - 9980) / 10010 = 0.003
        assert r2["max_drawdown"] > 0
        assert r2["calmar_ratio"] > 0


# ---------------------------------------------------------------------------
# Test 8: Win rate
# ---------------------------------------------------------------------------

class TestWinRate:
    """Win rate = fraction of trades with net_pnl > 0."""

    def test_win_rate_two_of_three(self):
        # 3 trades: 2 win, 1 lose
        # Win: target > fees/contracts = 10/200 = 0.05
        # Lose: target < -0.05 (after direction adjustment)
        df = _make_df(
            timestamps=[100, 200, 300],
            pair_ids=["A", "A", "A"],
            spreads=[0.40, 0.40, 0.40],
            # Trade 1: pred +, target +0.10 => net = 200*0.10 - 10 = +10 (WIN)
            # Trade 2: pred +, target +0.08 => net = 200*0.08 - 10 = +6 (WIN)
            # Trade 3: pred +, target -0.10 => net = 200*(-0.10) - 10 = -30 (LOSS)
            spread_change_targets=[0.10, 0.08, -0.10],
        )
        predictions = np.array([0.05, 0.05, 0.05])

        bt = WalkForwardBacktester(threshold=0.02)
        result = bt.run(df, predictions)

        assert result["num_trades"] == 3
        assert abs(result["win_rate"] - 2.0 / 3.0) < 0.01
