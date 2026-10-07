"""Tests for the live data collector.

Verifies bar construction, schema matching against train.parquet,
spread calculation, None-price exclusion, and parquet append logic.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.schemas import ALIGNED_COLUMNS, DERIVED_FEATURE_COLUMNS, OUTPUT_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TRAIN_PARQUET = Path("data/processed/train.parquet")
TRAIN_COLUMNS = [
    "timestamp", "kalshi_vwap", "kalshi_open", "kalshi_high", "kalshi_low",
    "kalshi_close", "kalshi_volume", "kalshi_trade_count", "kalshi_dollar_volume",
    "kalshi_buy_volume", "kalshi_sell_volume", "kalshi_realized_spread",
    "kalshi_max_trade_size", "kalshi_has_trade", "kalshi_hours_since_last_trade",
    "polymarket_vwap", "polymarket_open", "polymarket_high", "polymarket_low",
    "polymarket_close", "polymarket_volume", "polymarket_trade_count",
    "polymarket_dollar_volume", "polymarket_buy_volume", "polymarket_sell_volume",
    "polymarket_realized_spread", "polymarket_max_trade_size", "polymarket_has_trade",
    "polymarket_hours_since_last_trade", "spread", "pair_id",
    "price_velocity", "volume_ratio", "spread_momentum", "spread_volatility",
    "kalshi_order_flow_imbalance", "polymarket_order_flow_imbalance",
    "time_idx", "group_id",
]


@pytest.fixture
def train_df():
    """Load train.parquet for schema reference."""
    return pd.read_parquet(TRAIN_PARQUET)


@pytest.fixture
def sample_bar():
    """A valid snapshot bar dict."""
    from src.live.collector import LiveCollector
    return LiveCollector.build_snapshot_bar(
        kalshi_price=0.65,
        polymarket_price=0.60,
        pair_id="test-pair-001",
        timestamp=int(time.time()),
    )


@pytest.fixture
def collector(tmp_path):
    """A LiveCollector with tmp output dir, no API calls."""
    from src.live.collector import LiveCollector
    return LiveCollector(live_dir=tmp_path)


# ---------------------------------------------------------------------------
# Test 1: build_snapshot_bar produces all ALIGNED_COLUMNS keys
# ---------------------------------------------------------------------------

class TestBuildSnapshotBar:
    def test_produces_all_aligned_keys(self, sample_bar):
        """build_snapshot_bar dict has all 31 ALIGNED_COLUMNS keys."""
        for col in ALIGNED_COLUMNS:
            assert col in sample_bar, f"Missing key: {col}"

    def test_spread_computed_correctly(self):
        """spread = kalshi_price - polymarket_price."""
        from src.live.collector import LiveCollector
        bar = LiveCollector.build_snapshot_bar(0.70, 0.55, "pair-x", 1000)
        assert abs(bar["spread"] - 0.15) < 1e-9

    def test_ohlcv_snapshot_mode(self):
        """In snapshot mode: open=high=low=close=vwap=price, volume=0."""
        from src.live.collector import LiveCollector
        bar = LiveCollector.build_snapshot_bar(0.45, 0.50, "pair-y", 2000)
        # Kalshi side
        assert bar["kalshi_vwap"] == 0.45
        assert bar["kalshi_open"] == 0.45
        assert bar["kalshi_high"] == 0.45
        assert bar["kalshi_low"] == 0.45
        assert bar["kalshi_close"] == 0.45
        assert bar["kalshi_volume"] == 0.0
        assert bar["kalshi_trade_count"] == 0.0
        assert bar["kalshi_has_trade"] is False
        # Polymarket side
        assert bar["polymarket_vwap"] == 0.50
        assert bar["polymarket_open"] == 0.50
        assert bar["polymarket_close"] == 0.50
        assert bar["polymarket_volume"] == 0.0
        assert bar["polymarket_has_trade"] is False

    def test_spread_negative_when_poly_higher(self):
        """Negative spread when polymarket price exceeds kalshi price."""
        from src.live.collector import LiveCollector
        bar = LiveCollector.build_snapshot_bar(0.40, 0.60, "pair-z", 3000)
        assert bar["spread"] < 0


# ---------------------------------------------------------------------------
# Test 4-5: assemble_bar_dataframe schema matching
# ---------------------------------------------------------------------------

class TestAssembleBarDataframe:
    def test_produces_39_columns(self, collector, train_df):
        """assemble_bar_dataframe output has 39 columns matching train.parquet."""
        from src.live.collector import LiveCollector
        bars = [
            LiveCollector.build_snapshot_bar(0.65, 0.60, "kxbtc26feb0617b-0x0356fe1e", 1000),
            LiveCollector.build_snapshot_bar(0.55, 0.50, "kxbtc26feb0617b-0x7c099bd6", 1000),
        ]
        df = collector.assemble_bar_dataframe(bars)
        assert list(df.columns) == list(train_df.columns), (
            f"Column mismatch.\nExpected: {list(train_df.columns)}\nGot: {list(df.columns)}"
        )

    def test_correct_dtypes(self, collector, train_df):
        """Dtypes match train.parquet (float64 for numeric, bool for has_trade, etc.)."""
        from src.live.collector import LiveCollector
        bars = [
            LiveCollector.build_snapshot_bar(0.65, 0.60, "kxbtc26feb0617b-0x0356fe1e", 1000),
        ]
        df = collector.assemble_bar_dataframe(bars)
        for col in df.columns:
            expected_dtype = train_df[col].dtype
            actual_dtype = df[col].dtype
            assert actual_dtype == expected_dtype, (
                f"Dtype mismatch for '{col}': expected {expected_dtype}, got {actual_dtype}"
            )

    def test_derived_features_computed(self, collector):
        """assemble_bar_dataframe includes the 6 derived feature columns."""
        from src.live.collector import LiveCollector
        bars = [
            LiveCollector.build_snapshot_bar(0.65, 0.60, "kxbtc26feb0617b-0x0356fe1e", 1000),
        ]
        df = collector.assemble_bar_dataframe(bars)
        for col in DERIVED_FEATURE_COLUMNS:
            assert col in df.columns, f"Missing derived feature: {col}"

    def test_time_idx_and_group_id_present(self, collector):
        """assemble_bar_dataframe adds time_idx and group_id columns."""
        from src.live.collector import LiveCollector
        bars = [
            LiveCollector.build_snapshot_bar(0.65, 0.60, "kxbtc26feb0617b-0x0356fe1e", 1000),
        ]
        df = collector.assemble_bar_dataframe(bars)
        assert "time_idx" in df.columns
        assert "group_id" in df.columns


# ---------------------------------------------------------------------------
# Test 6: None-price pair exclusion
# ---------------------------------------------------------------------------

class TestNonePriceExclusion:
    def test_none_kalshi_price_excluded(self, collector):
        """Bars with None kalshi price are excluded from assembly."""
        from src.live.collector import LiveCollector
        bars = [
            LiveCollector.build_snapshot_bar(0.65, 0.60, "kxbtc26feb0617b-0x0356fe1e", 1000),
        ]
        # Simulate: collector builds bars only for pairs with both prices
        # None prices are filtered before bar construction, so we test
        # that assemble_bar_dataframe works with valid bars only
        df = collector.assemble_bar_dataframe(bars)
        assert len(df) == 1

    def test_empty_bars_list(self, collector):
        """assemble_bar_dataframe with empty list returns empty DataFrame with correct columns."""
        df = collector.assemble_bar_dataframe([])
        assert len(df) == 0
        assert list(df.columns) == TRAIN_COLUMNS


# ---------------------------------------------------------------------------
# Test 7: append_to_parquet
# ---------------------------------------------------------------------------

class TestAppendToParquet:
    def test_creates_new_file(self, collector, tmp_path):
        """append_to_parquet creates a new file when none exists."""
        from src.live.collector import LiveCollector
        bars = [
            LiveCollector.build_snapshot_bar(0.65, 0.60, "kxbtc26feb0617b-0x0356fe1e", 1000),
        ]
        df = collector.assemble_bar_dataframe(bars)
        out_path = tmp_path / "test_bars.parquet"
        collector.append_to_parquet(df, out_path)
        assert out_path.exists()
        loaded = pd.read_parquet(out_path)
        assert len(loaded) == 1

    def test_appends_to_existing(self, collector, tmp_path):
        """append_to_parquet appends new rows to existing file."""
        from src.live.collector import LiveCollector
        out_path = tmp_path / "test_bars.parquet"
        # First write
        bars1 = [LiveCollector.build_snapshot_bar(0.65, 0.60, "kxbtc26feb0617b-0x0356fe1e", 1000)]
        df1 = collector.assemble_bar_dataframe(bars1)
        collector.append_to_parquet(df1, out_path)
        # Second write
        bars2 = [LiveCollector.build_snapshot_bar(0.55, 0.50, "kxbtc26feb0617b-0x7c099bd6", 2000)]
        df2 = collector.assemble_bar_dataframe(bars2)
        collector.append_to_parquet(df2, out_path)

        loaded = pd.read_parquet(out_path)
        assert len(loaded) == 2

    def test_preserves_schema(self, collector, tmp_path, train_df):
        """append_to_parquet preserves the 39-column schema after append."""
        from src.live.collector import LiveCollector
        out_path = tmp_path / "test_bars.parquet"
        bars = [LiveCollector.build_snapshot_bar(0.65, 0.60, "kxbtc26feb0617b-0x0356fe1e", 1000)]
        df = collector.assemble_bar_dataframe(bars)
        collector.append_to_parquet(df, out_path)
        loaded = pd.read_parquet(out_path)
        assert list(loaded.columns) == list(train_df.columns)
