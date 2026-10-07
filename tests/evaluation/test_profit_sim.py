"""Tests for trading profit simulation."""
import math

import numpy as np
import pytest

from src.evaluation.profit_sim import simulate_profit


class TestSimulateProfit:
    def test_returns_dict_with_expected_keys(self, sample_y_true, sample_y_pred):
        result = simulate_profit(sample_y_pred, sample_y_true, threshold=0.01)
        assert set(result.keys()) == {
            "total_pnl",
            "num_trades",
            "win_rate",
            "sharpe_ratio",
            "sharpe_per_trade",
            "pnl_series",
        }

    def test_no_trades_when_all_predictions_below_threshold(self):
        predictions = np.array([0.001, -0.001, 0.002, -0.0005])
        actuals = np.array([0.1, -0.1, 0.1, -0.1])
        result = simulate_profit(predictions, actuals, threshold=0.02)
        assert result["num_trades"] == 0
        assert result["total_pnl"] == 0.0
        assert result["win_rate"] == 0.0
        assert result["sharpe_ratio"] == 0.0
        assert result["pnl_series"] == []

    def test_single_profitable_trade_positive_side(self):
        # predicted +0.05 (above 0.02 threshold) -> bet positive direction
        # actual spread change +0.03 -> profit = 0.03 * sign(+0.05) = +0.03
        predictions = np.array([0.05])
        actuals = np.array([0.03])
        result = simulate_profit(predictions, actuals, threshold=0.02)
        assert result["num_trades"] == 1
        assert result["total_pnl"] == pytest.approx(0.03)
        assert result["win_rate"] == pytest.approx(1.0)
        assert result["pnl_series"] == pytest.approx([0.03])

    def test_single_profitable_trade_negative_side(self):
        # predicted -0.05 -> bet negative direction (sign = -1)
        # actual -0.04 -> profit = -0.04 * -1 = +0.04
        predictions = np.array([-0.05])
        actuals = np.array([-0.04])
        result = simulate_profit(predictions, actuals, threshold=0.02)
        assert result["num_trades"] == 1
        assert result["total_pnl"] == pytest.approx(0.04)
        assert result["win_rate"] == pytest.approx(1.0)

    def test_single_losing_trade(self):
        # predicted +0.05 -> sign = +1
        # actual -0.02 -> profit = -0.02 * 1 = -0.02 (loss)
        predictions = np.array([0.05])
        actuals = np.array([-0.02])
        result = simulate_profit(predictions, actuals, threshold=0.02)
        assert result["num_trades"] == 1
        assert result["total_pnl"] == pytest.approx(-0.02)
        assert result["win_rate"] == pytest.approx(0.0)

    def test_mixed_profitable_and_losing_trades(self):
        # predicted: [+0.05, -0.05, +0.03, +0.05]  (all above threshold)
        # actuals:   [+0.04, +0.02, -0.01, +0.05]
        # signs:     [+1, -1, +1, +1]
        # pnls:      [+0.04, -0.02, -0.01, +0.05]
        # wins: 2 / 4 = 0.5
        predictions = np.array([0.05, -0.05, 0.03, 0.05])
        actuals = np.array([0.04, 0.02, -0.01, 0.05])
        result = simulate_profit(predictions, actuals, threshold=0.02)
        assert result["num_trades"] == 4
        assert result["total_pnl"] == pytest.approx(0.04 - 0.02 - 0.01 + 0.05)
        assert result["win_rate"] == pytest.approx(0.5)
        # cumulative series
        assert result["pnl_series"] == pytest.approx([0.04, 0.02, 0.01, 0.06])

    def test_trades_filtered_by_threshold(self):
        # Only first and last are above threshold
        predictions = np.array([0.05, 0.01, -0.005, 0.04])
        actuals = np.array([0.03, 0.02, -0.02, 0.01])
        result = simulate_profit(predictions, actuals, threshold=0.02)
        assert result["num_trades"] == 2
        # pnls: 0.03 * +1 = +0.03, 0.01 * +1 = +0.01
        assert result["total_pnl"] == pytest.approx(0.04)
        assert result["win_rate"] == pytest.approx(1.0)

    def test_sharpe_ratio_zero_for_single_trade(self):
        # With <2 trades std undefined, sharpe should be 0
        predictions = np.array([0.05])
        actuals = np.array([0.03])
        result = simulate_profit(predictions, actuals, threshold=0.02)
        assert result["sharpe_ratio"] == 0.0

    def test_sharpe_ratio_matches_formula(self):
        # Two identical trades -> std == 0 -> sharpe == 0
        predictions = np.array([0.05, 0.05])
        actuals = np.array([0.03, 0.03])
        result = simulate_profit(predictions, actuals, threshold=0.02)
        assert result["sharpe_ratio"] == 0.0

    def test_sharpe_ratio_positive_with_winning_varied_trades(self):
        # Bar returns: [0.02, 0.04] (both bars trade)
        # Time-series Sharpe: mean/std * sqrt(2190) (4-hour bars, 24/7 markets)
        predictions = np.array([0.05, 0.05])
        actuals = np.array([0.02, 0.04])
        result = simulate_profit(predictions, actuals, threshold=0.02)
        # Computed value
        returns = np.array([0.02, 0.04])
        expected = returns.mean() / returns.std() * math.sqrt(2190)
        assert result["sharpe_ratio"] == pytest.approx(expected)
        # sharpe_per_trade is unannualized
        expected_per_trade = returns.mean() / returns.std()
        assert result["sharpe_per_trade"] == pytest.approx(expected_per_trade)

    def test_pnl_series_is_cumulative(self):
        predictions = np.array([0.05, -0.05, 0.05])
        actuals = np.array([0.01, 0.01, 0.02])
        # pnls per bar: 0.01*+1=0.01, 0.01*-1=-0.01, 0.02*+1=0.02
        # cumulative across all bars: [0.01, 0.00, 0.02]
        result = simulate_profit(predictions, actuals, threshold=0.02)
        assert result["pnl_series"] == pytest.approx([0.01, 0.00, 0.02])

    def test_default_threshold(self):
        # Default threshold=0.02
        predictions = np.array([0.01, 0.03])
        actuals = np.array([0.01, 0.02])
        result = simulate_profit(predictions, actuals)
        assert result["num_trades"] == 1
        assert result["total_pnl"] == pytest.approx(0.02)

    def test_length_mismatch_raises(self):
        predictions = np.array([0.05, 0.03])
        actuals = np.array([0.01])
        with pytest.raises(ValueError):
            simulate_profit(predictions, actuals)
