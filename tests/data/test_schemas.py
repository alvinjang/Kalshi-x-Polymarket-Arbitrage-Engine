"""Tests for schema definitions and validation."""
import pytest
import pandas as pd
from src.data.schemas import (
    MarketMetadata,
    CANDLESTICK_COLUMNS,
    validate_candlestick_df,
)


class TestMarketMetadata:
    def test_create_metadata(self):
        """MarketMetadata dataclass has expected fields."""
        meta = MarketMetadata(
            market_id="KXBTC-25DEC31-T50000",
            question="Will Bitcoin exceed $50,000?",
            category="crypto",
            platform="kalshi",
            resolution_date="2025-12-31T23:59:59Z",
            result="yes",
            outcomes=["Yes", "No"],
        )
        assert meta.market_id == "KXBTC-25DEC31-T50000"
        assert meta.platform == "kalshi"
        assert meta.result == "yes"

    def test_metadata_default_outcomes(self):
        """MarketMetadata defaults outcomes to ['Yes', 'No']."""
        meta = MarketMetadata(
            market_id="test",
            question="test?",
            category="crypto",
            platform="kalshi",
            resolution_date="2025-12-31",
            result=None,
        )
        assert meta.outcomes == ["Yes", "No"]


class TestCandlestickColumns:
    def test_required_columns_present(self):
        """CANDLESTICK_COLUMNS contains the 6 required columns."""
        assert "timestamp" in CANDLESTICK_COLUMNS
        assert "open" in CANDLESTICK_COLUMNS
        assert "high" in CANDLESTICK_COLUMNS
        assert "low" in CANDLESTICK_COLUMNS
        assert "close" in CANDLESTICK_COLUMNS
        assert "volume" in CANDLESTICK_COLUMNS
        assert len(CANDLESTICK_COLUMNS) == 6


class TestValidateCandlestickDf:
    def test_valid_df(self):
        """Validator returns no errors for valid DataFrame."""
        df = pd.DataFrame({
            "timestamp": [1704067200, 1704067260],
            "open": [0.55, 0.56],
            "high": [0.60, 0.61],
            "low": [0.50, 0.51],
            "close": [0.58, 0.59],
            "volume": [150.0, 100.0],
        })
        errors = validate_candlestick_df(df)
        assert errors == []

    def test_missing_column(self):
        """Validator reports missing required column."""
        df = pd.DataFrame({
            "timestamp": [1],
            "open": [0.5],
            # missing high, low, close, volume
        })
        errors = validate_candlestick_df(df)
        assert any("high" in e for e in errors)
        assert any("low" in e for e in errors)
        assert any("close" in e for e in errors)
        assert any("volume" in e for e in errors)

    def test_empty_df(self):
        """Validator rejects empty DataFrame."""
        df = pd.DataFrame(columns=CANDLESTICK_COLUMNS)
        errors = validate_candlestick_df(df)
        assert any("empty" in e.lower() for e in errors)

    def test_null_timestamp(self):
        """Validator rejects NaN in timestamp column."""
        df = pd.DataFrame({
            "timestamp": [1704067200, None],
            "open": [0.55, 0.56],
            "high": [0.60, 0.61],
            "low": [0.50, 0.51],
            "close": [0.58, 0.59],
            "volume": [150.0, 100.0],
        })
        errors = validate_candlestick_df(df)
        assert any("timestamp" in e and "NaN" in e for e in errors)
