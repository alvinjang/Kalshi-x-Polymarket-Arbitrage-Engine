"""Tests for cross-platform aligner with forward-fill and staleness decay.

Tests cover:
- align_pair: grid creation, forward-fill, staleness decay, spread computation
- align_all_pairs: quality filters (min trades, staleness, spread), quality report
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.aligner import align_pair, align_all_pairs
from src.data.schemas import CANDLE_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candles(timestamps: list[int], vwaps: list[float], trade_counts=None) -> pd.DataFrame:
    """Create a minimal CANDLE_COLUMNS DataFrame for testing.

    Only bars with trades are emitted (matching reconstructor output).
    """
    n = len(timestamps)
    if trade_counts is None:
        trade_counts = [5] * n

    data = {
        "timestamp": timestamps,
        "vwap": vwaps,
        "open": vwaps,
        "high": [v + 0.02 for v in vwaps],
        "low": [v - 0.02 for v in vwaps],
        "close": vwaps,
        "volume": [100.0] * n,
        "trade_count": trade_counts,
        "dollar_volume": [v * 100 for v in vwaps],
        "buy_volume": [60.0] * n,
        "sell_volume": [40.0] * n,
        "realized_spread": [0.01] * n,
        "max_trade_size": [20.0] * n,
        "has_trade": [True] * n,
    }
    return pd.DataFrame(data, columns=CANDLE_COLUMNS)


BAR = 14400  # 4 hours in seconds
T0 = 1704067200  # 2024-01-01 00:00:00 UTC (aligned to 4h boundary)


# ---------------------------------------------------------------------------
# align_pair tests
# ---------------------------------------------------------------------------

class TestAlignPair:
    """Tests for align_pair function."""

    def test_overlapping_candles_correct_grid(self):
        """Overlapping candles produce correct unified grid and merge."""
        kalshi = _make_candles([T0, T0 + BAR, T0 + 2 * BAR], [0.60, 0.62, 0.65])
        poly = _make_candles([T0, T0 + BAR, T0 + 2 * BAR], [0.55, 0.57, 0.60])

        result = align_pair(kalshi, poly, "test-pair")

        assert result is not None
        assert len(result) == 3
        assert "pair_id" in result.columns
        assert result["pair_id"].iloc[0] == "test-pair"
        assert "kalshi_vwap" in result.columns
        assert "polymarket_vwap" in result.columns
        assert "spread" in result.columns

    def test_forward_fill_gap_within_limit(self):
        """Gap of 3 bars (12 hours) is filled via forward-fill (limit=6)."""
        # Kalshi has data at T0, then a 3-bar gap, then T0+4*BAR
        kalshi = _make_candles([T0, T0 + 4 * BAR], [0.60, 0.65])
        # Polymarket has data at every bar
        poly = _make_candles(
            [T0 + i * BAR for i in range(5)],
            [0.55, 0.56, 0.57, 0.58, 0.59],
        )

        result = align_pair(kalshi, poly, "gap-test")

        assert result is not None
        # All 5 bars should have data (kalshi forward-filled for bars 1-3)
        kalshi_vwaps = result["kalshi_vwap"].tolist()
        assert kalshi_vwaps[0] == 0.60  # original
        assert kalshi_vwaps[1] == 0.60  # forward-filled
        assert kalshi_vwaps[2] == 0.60  # forward-filled
        assert kalshi_vwaps[3] == 0.60  # forward-filled
        assert kalshi_vwaps[4] == 0.65  # original

    def test_forward_fill_gap_exceeds_limit(self):
        """Gap of 7 bars: first 6 filled, 7th becomes NaN."""
        # Kalshi only at T0 and T0+8*BAR
        kalshi = _make_candles([T0, T0 + 8 * BAR], [0.60, 0.70])
        # Polymarket at every bar
        poly = _make_candles(
            [T0 + i * BAR for i in range(9)],
            [0.55 + i * 0.01 for i in range(9)],
        )

        result = align_pair(kalshi, poly, "stale-test")
        assert result is not None

        kalshi_vwaps = result["kalshi_vwap"].tolist()
        # Bar 0: original 0.60
        assert kalshi_vwaps[0] == 0.60
        # Bars 1-6: forward-filled (limit=6)
        for i in range(1, 7):
            assert kalshi_vwaps[i] == 0.60, f"Bar {i} should be forward-filled"
        # Bar 7: beyond limit, should be NaN
        assert np.isnan(kalshi_vwaps[7]), "Bar 7 should be NaN (beyond 6-bar limit)"
        # Bar 8: original
        assert kalshi_vwaps[8] == 0.70

    def test_hours_since_last_trade(self):
        """hours_since_last_trade is 0 for fresh bars, 4*N for N-bar gaps."""
        # Kalshi at T0 and T0+3*BAR (gap of 2 bars)
        kalshi = _make_candles([T0, T0 + 3 * BAR], [0.60, 0.65])
        poly = _make_candles(
            [T0 + i * BAR for i in range(4)],
            [0.55, 0.56, 0.57, 0.58],
        )

        result = align_pair(kalshi, poly, "staleness-hours")
        assert result is not None

        hours_col = result["kalshi_hours_since_last_trade"].tolist()
        assert hours_col[0] == 0    # fresh bar
        assert hours_col[1] == 4    # 1 bar since last trade
        assert hours_col[2] == 8    # 2 bars since last trade
        assert hours_col[3] == 0    # fresh bar again

    def test_spread_computation(self):
        """Spread = kalshi_vwap - polymarket_vwap."""
        kalshi = _make_candles([T0], [0.60])
        poly = _make_candles([T0], [0.55])

        result = align_pair(kalshi, poly, "spread-test")
        assert result is not None

        spread = result["spread"].iloc[0]
        assert abs(spread - 0.05) < 1e-10

    def test_spread_nan_when_either_vwap_nan(self):
        """Spread is NaN when either platform's VWAP is NaN after staleness decay."""
        # Kalshi only at T0, poly at T0 and T0+8*BAR
        # At bar 7, kalshi_vwap will be NaN (past 6-bar ffill limit)
        kalshi = _make_candles([T0], [0.60])
        poly = _make_candles(
            [T0 + i * BAR for i in range(9)],
            [0.55 + i * 0.01 for i in range(9)],
        )

        result = align_pair(kalshi, poly, "nan-spread")
        assert result is not None

        # Bar 7 (index 7): kalshi_vwap is NaN, so spread should be NaN
        assert np.isnan(result["spread"].iloc[7])

    def test_empty_kalshi_candles_returns_none(self):
        """align_pair returns None when Kalshi candles are empty."""
        kalshi = pd.DataFrame(columns=CANDLE_COLUMNS)
        poly = _make_candles([T0], [0.55])

        result = align_pair(kalshi, poly, "empty-k")
        assert result is None

    def test_empty_polymarket_candles_returns_none(self):
        """align_pair returns None when Polymarket candles are empty."""
        kalshi = _make_candles([T0], [0.60])
        poly = pd.DataFrame(columns=CANDLE_COLUMNS)

        result = align_pair(kalshi, poly, "empty-p")
        assert result is None

    def test_pair_id_column_present(self):
        """Output DataFrame contains pair_id column."""
        kalshi = _make_candles([T0, T0 + BAR], [0.60, 0.62])
        poly = _make_candles([T0, T0 + BAR], [0.55, 0.57])

        result = align_pair(kalshi, poly, "pid-test")
        assert result is not None
        assert "pair_id" in result.columns
        assert (result["pair_id"] == "pid-test").all()

    def test_output_has_prefixed_columns(self):
        """Output columns are prefixed with platform name."""
        kalshi = _make_candles([T0], [0.60])
        poly = _make_candles([T0], [0.55])

        result = align_pair(kalshi, poly, "cols-test")
        assert result is not None

        # Check key prefixed columns exist
        for prefix in ["kalshi_", "polymarket_"]:
            assert f"{prefix}vwap" in result.columns
            assert f"{prefix}open" in result.columns
            assert f"{prefix}high" in result.columns
            assert f"{prefix}low" in result.columns
            assert f"{prefix}close" in result.columns
            assert f"{prefix}volume" in result.columns
            assert f"{prefix}has_trade" in result.columns
            assert f"{prefix}hours_since_last_trade" in result.columns


# ---------------------------------------------------------------------------
# align_all_pairs tests
# ---------------------------------------------------------------------------

class TestAlignAllPairs:
    """Tests for align_all_pairs function with quality filtering."""

    def _setup_candle_files(self, tmp_path, pair_id, kalshi_id, poly_id,
                            kalshi_candles, poly_candles):
        """Save candle DataFrames to the expected directory structure."""
        candles_dir = tmp_path / "candles"
        k_dir = candles_dir / "kalshi"
        p_dir = candles_dir / "polymarket"
        k_dir.mkdir(parents=True, exist_ok=True)
        p_dir.mkdir(parents=True, exist_ok=True)

        kalshi_candles.to_parquet(k_dir / f"{kalshi_id}_candles.parquet", index=False)
        poly_candles.to_parquet(p_dir / f"{poly_id}_candles.parquet", index=False)

        return candles_dir

    def test_excludes_pair_with_insufficient_trades(self, tmp_path):
        """Pairs with < 20 trades on either platform are excluded."""
        kalshi_id = "KX-TEST"
        poly_id = "0xtest"
        pair_id = "test-low-trades"

        # Kalshi: only 10 total trades (below 20 threshold)
        kalshi = _make_candles(
            [T0 + i * BAR for i in range(5)],
            [0.60] * 5,
            trade_counts=[2, 2, 2, 2, 2],  # total = 10
        )
        # Polymarket: 50 trades (above threshold)
        poly = _make_candles(
            [T0 + i * BAR for i in range(5)],
            [0.55] * 5,
            trade_counts=[10, 10, 10, 10, 10],  # total = 50
        )

        candles_dir = self._setup_candle_files(
            tmp_path, pair_id, kalshi_id, poly_id, kalshi, poly,
        )

        pairs = [{"kalshi_market_id": kalshi_id, "polymarket_market_id": poly_id, "pair_id": pair_id}]
        aligned_df, report = align_all_pairs(pairs, candles_dir)

        assert len(aligned_df) == 0
        assert report["excluded_pairs"] == 1
        assert report["exclusion_reasons"]["insufficient_trades"] == 1

    def test_excludes_pair_with_high_staleness(self, tmp_path):
        """Pairs with staleness_ratio >= 0.80 are excluded."""
        kalshi_id = "KX-STALE"
        poly_id = "0xstale"
        pair_id = "test-stale"

        # 20 bars total. Kalshi only has trade at bar 0 and bar 19.
        # After forward-fill (limit=6), bars 7-18 will have no data.
        # Staleness = fraction of bars where BOTH has_trade are False
        # Most bars will have poly data but not kalshi -> the combined
        # staleness check will catch this.
        kalshi = _make_candles([T0, T0 + 19 * BAR], [0.60, 0.65], trade_counts=[30, 30])
        poly = _make_candles([T0, T0 + 19 * BAR], [0.55, 0.58], trade_counts=[30, 30])

        candles_dir = self._setup_candle_files(
            tmp_path, pair_id, kalshi_id, poly_id, kalshi, poly,
        )

        pairs = [{"kalshi_market_id": kalshi_id, "polymarket_market_id": poly_id, "pair_id": pair_id}]
        aligned_df, report = align_all_pairs(pairs, candles_dir)

        assert report["excluded_pairs"] == 1
        assert report["exclusion_reasons"]["too_stale"] == 1

    def test_excludes_pair_with_extreme_spread(self, tmp_path):
        """Pairs with mean |spread| >= 0.30 are excluded."""
        kalshi_id = "KX-SPREAD"
        poly_id = "0xspread"
        pair_id = "test-spread"

        # Large spread: kalshi ~0.80, poly ~0.40 -> |spread| = 0.40
        kalshi = _make_candles(
            [T0 + i * BAR for i in range(10)],
            [0.80] * 10,
            trade_counts=[5] * 10,
        )
        poly = _make_candles(
            [T0 + i * BAR for i in range(10)],
            [0.40] * 10,
            trade_counts=[5] * 10,
        )

        candles_dir = self._setup_candle_files(
            tmp_path, pair_id, kalshi_id, poly_id, kalshi, poly,
        )

        pairs = [{"kalshi_market_id": kalshi_id, "polymarket_market_id": poly_id, "pair_id": pair_id}]
        aligned_df, report = align_all_pairs(pairs, candles_dir)

        assert report["excluded_pairs"] == 1
        assert report["exclusion_reasons"]["spread_not_mean_reverting"] == 1

    def test_quality_report_structure(self, tmp_path):
        """Quality report has all required keys and correct counts."""
        kalshi_id = "KX-GOOD"
        poly_id = "0xgood"
        pair_id = "test-good"

        kalshi = _make_candles(
            [T0 + i * BAR for i in range(10)],
            [0.60 + i * 0.005 for i in range(10)],
            trade_counts=[5] * 10,
        )
        poly = _make_candles(
            [T0 + i * BAR for i in range(10)],
            [0.57 + i * 0.005 for i in range(10)],
            trade_counts=[5] * 10,
        )

        candles_dir = self._setup_candle_files(
            tmp_path, pair_id, kalshi_id, poly_id, kalshi, poly,
        )

        pairs = [{"kalshi_market_id": kalshi_id, "polymarket_market_id": poly_id, "pair_id": pair_id}]
        aligned_df, report = align_all_pairs(pairs, candles_dir)

        # Check report structure
        assert "total_pairs" in report
        assert "aligned_pairs" in report
        assert "excluded_pairs" in report
        assert "exclusion_reasons" in report
        assert "per_pair" in report
        assert report["total_pairs"] == 1
        assert report["aligned_pairs"] + report["excluded_pairs"] == report["total_pairs"]

        # Check exclusion_reasons keys
        reasons = report["exclusion_reasons"]
        assert "missing_candles" in reasons
        assert "insufficient_trades" in reasons
        assert "too_stale" in reasons
        assert "spread_not_mean_reverting" in reasons
        assert "alignment_failed" in reasons

    def test_included_pair_in_output(self, tmp_path):
        """A good pair passes all filters and appears in output."""
        kalshi_id = "KX-PASS"
        poly_id = "0xpass"
        pair_id = "test-pass"

        kalshi = _make_candles(
            [T0 + i * BAR for i in range(10)],
            [0.60 + i * 0.005 for i in range(10)],
            trade_counts=[5] * 10,  # total 50 trades
        )
        poly = _make_candles(
            [T0 + i * BAR for i in range(10)],
            [0.57 + i * 0.005 for i in range(10)],
            trade_counts=[5] * 10,  # total 50 trades
        )

        candles_dir = self._setup_candle_files(
            tmp_path, pair_id, kalshi_id, poly_id, kalshi, poly,
        )

        pairs = [{"kalshi_market_id": kalshi_id, "polymarket_market_id": poly_id, "pair_id": pair_id}]
        aligned_df, report = align_all_pairs(pairs, candles_dir)

        assert len(aligned_df) == 10
        assert report["aligned_pairs"] == 1
        assert report["excluded_pairs"] == 0
        assert "pair_id" in aligned_df.columns
        assert (aligned_df["pair_id"] == pair_id).all()

    def test_missing_candle_files_excluded(self, tmp_path):
        """Pairs with missing candle files are excluded with reason."""
        candles_dir = tmp_path / "candles"
        candles_dir.mkdir(parents=True, exist_ok=True)
        (candles_dir / "kalshi").mkdir(exist_ok=True)
        (candles_dir / "polymarket").mkdir(exist_ok=True)

        pairs = [{"kalshi_market_id": "KX-MISSING", "polymarket_market_id": "0xmissing", "pair_id": "missing"}]
        aligned_df, report = align_all_pairs(pairs, candles_dir)

        assert len(aligned_df) == 0
        assert report["excluded_pairs"] == 1
        assert report["exclusion_reasons"]["missing_candles"] == 1

    def test_per_pair_report_entry(self, tmp_path):
        """per_pair report entries have required fields."""
        kalshi_id = "KX-RPT"
        poly_id = "0xrpt"
        pair_id = "test-rpt"

        kalshi = _make_candles(
            [T0 + i * BAR for i in range(5)],
            [0.60] * 5,
            trade_counts=[10] * 5,  # total 50
        )
        poly = _make_candles(
            [T0 + i * BAR for i in range(5)],
            [0.57] * 5,
            trade_counts=[10] * 5,  # total 50
        )

        candles_dir = self._setup_candle_files(
            tmp_path, pair_id, kalshi_id, poly_id, kalshi, poly,
        )

        pairs = [{"kalshi_market_id": kalshi_id, "polymarket_market_id": poly_id, "pair_id": pair_id}]
        _, report = align_all_pairs(pairs, candles_dir)

        assert len(report["per_pair"]) == 1
        entry = report["per_pair"][0]
        assert entry["pair_id"] == pair_id
        assert "status" in entry
        assert "kalshi_trades" in entry
        assert "polymarket_trades" in entry
        assert "staleness_ratio" in entry
        assert "mean_abs_spread" in entry
        assert "bars" in entry
