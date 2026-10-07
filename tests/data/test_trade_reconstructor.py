"""Tests for the trade reconstructor module (Phase 2.1 Plan 01, Task 2)."""
import pandas as pd
import pytest

from src.data.schemas import CANDLE_COLUMNS


class TestReconstructCandles:
    """Test 4-hour OHLCV+VWAP candle reconstruction from raw trades."""

    def _make_trades_df(self, trades: list[dict]) -> pd.DataFrame:
        """Helper to create a trades DataFrame from list of dicts."""
        return pd.DataFrame(trades, columns=["timestamp", "price", "volume", "side"])

    def test_single_trade_all_ohlc_equal(self):
        """Single trade in a bar: open == high == low == close == price, vwap == price."""
        from src.data.trade_reconstructor import reconstruct_candles

        trades_df = self._make_trades_df([
            {"timestamp": 1704067200, "price": 0.50, "volume": 100.0, "side": "buy"},
        ])

        result = reconstruct_candles(trades_df)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["open"] == 0.50
        assert row["high"] == 0.50
        assert row["low"] == 0.50
        assert row["close"] == 0.50
        assert row["vwap"] == 0.50
        assert row["volume"] == 100.0
        assert row["trade_count"] == 1

    def test_vwap_weighted_correctly(self):
        """VWAP = sum(price * volume) / sum(volume) for multiple trades in one bar."""
        from src.data.trade_reconstructor import reconstruct_candles

        # All trades in same 4-hour bar (bar starts at 1704067200)
        trades_df = self._make_trades_df([
            {"timestamp": 1704067200, "price": 0.50, "volume": 100.0, "side": "buy"},
            {"timestamp": 1704068000, "price": 0.60, "volume": 200.0, "side": "sell"},
            {"timestamp": 1704069000, "price": 0.55, "volume": 50.0, "side": "buy"},
        ])

        result = reconstruct_candles(trades_df)

        assert len(result) == 1
        # VWAP = (0.50*100 + 0.60*200 + 0.55*50) / (100+200+50)
        # = (50 + 120 + 27.5) / 350 = 197.5 / 350 = 0.564285...
        expected_vwap = (0.50 * 100 + 0.60 * 200 + 0.55 * 50) / (100 + 200 + 50)
        assert abs(result.iloc[0]["vwap"] - expected_vwap) < 0.001

    def test_multiple_bars_emitted_separately(self, normalized_trades):
        """Trades spanning two 4-hour bars produce two separate candle rows."""
        from src.data.trade_reconstructor import reconstruct_candles

        trades_df = self._make_trades_df(normalized_trades)
        result = reconstruct_candles(trades_df)

        assert len(result) == 2
        # Bars are sorted by timestamp
        assert result.iloc[0]["timestamp"] < result.iloc[1]["timestamp"]

    def test_buy_sell_volume_split(self):
        """buy_volume and sell_volume correctly split by side."""
        from src.data.trade_reconstructor import reconstruct_candles

        trades_df = self._make_trades_df([
            {"timestamp": 1704067200, "price": 0.50, "volume": 100.0, "side": "buy"},
            {"timestamp": 1704067500, "price": 0.55, "volume": 80.0, "side": "buy"},
            {"timestamp": 1704068000, "price": 0.60, "volume": 50.0, "side": "sell"},
        ])

        result = reconstruct_candles(trades_df)

        assert len(result) == 1
        assert result.iloc[0]["buy_volume"] == 180.0
        assert result.iloc[0]["sell_volume"] == 50.0

    def test_realized_spread_both_sides(self):
        """realized_spread = mean(sell_prices) - mean(buy_prices) when both sides present."""
        from src.data.trade_reconstructor import reconstruct_candles

        trades_df = self._make_trades_df([
            {"timestamp": 1704067200, "price": 0.50, "volume": 100.0, "side": "buy"},
            {"timestamp": 1704067500, "price": 0.52, "volume": 80.0, "side": "buy"},
            {"timestamp": 1704068000, "price": 0.60, "volume": 50.0, "side": "sell"},
            {"timestamp": 1704068500, "price": 0.58, "volume": 30.0, "side": "sell"},
        ])

        result = reconstruct_candles(trades_df)

        # mean(sell_prices) = (0.60 + 0.58) / 2 = 0.59
        # mean(buy_prices) = (0.50 + 0.52) / 2 = 0.51
        # realized_spread = 0.59 - 0.51 = 0.08
        expected = (0.60 + 0.58) / 2 - (0.50 + 0.52) / 2
        assert abs(result.iloc[0]["realized_spread"] - expected) < 0.001

    def test_realized_spread_one_side_only(self):
        """realized_spread = 0.0 when only one side (buys only) is present."""
        from src.data.trade_reconstructor import reconstruct_candles

        trades_df = self._make_trades_df([
            {"timestamp": 1704067200, "price": 0.50, "volume": 100.0, "side": "buy"},
            {"timestamp": 1704067500, "price": 0.55, "volume": 80.0, "side": "buy"},
        ])

        result = reconstruct_candles(trades_df)

        assert result.iloc[0]["realized_spread"] == 0.0

    def test_price_clipping(self):
        """Prices outside [0.01, 0.99] are clipped before aggregation."""
        from src.data.trade_reconstructor import reconstruct_candles

        trades_df = self._make_trades_df([
            {"timestamp": 1704067200, "price": 0.005, "volume": 100.0, "side": "buy"},
            {"timestamp": 1704067500, "price": 1.50, "volume": 100.0, "side": "sell"},
        ])

        result = reconstruct_candles(trades_df)

        # 0.005 -> 0.01, 1.50 -> 0.99
        assert result.iloc[0]["low"] == 0.01
        assert result.iloc[0]["high"] == 0.99
        # VWAP with clipped prices
        expected_vwap = (0.01 * 100 + 0.99 * 100) / 200
        assert abs(result.iloc[0]["vwap"] - expected_vwap) < 0.001

    def test_empty_input_returns_empty_with_columns(self):
        """Empty DataFrame input returns empty DataFrame with CANDLE_COLUMNS."""
        from src.data.trade_reconstructor import reconstruct_candles

        empty_df = pd.DataFrame(columns=["timestamp", "price", "volume", "side"])
        result = reconstruct_candles(empty_df)

        assert len(result) == 0
        assert list(result.columns) == CANDLE_COLUMNS

    def test_dollar_volume(self):
        """dollar_volume = sum(price * volume) per bar."""
        from src.data.trade_reconstructor import reconstruct_candles

        trades_df = self._make_trades_df([
            {"timestamp": 1704067200, "price": 0.50, "volume": 100.0, "side": "buy"},
            {"timestamp": 1704068000, "price": 0.60, "volume": 200.0, "side": "sell"},
        ])

        result = reconstruct_candles(trades_df)

        expected = 0.50 * 100 + 0.60 * 200
        assert abs(result.iloc[0]["dollar_volume"] - expected) < 0.001

    def test_max_trade_size(self):
        """max_trade_size = largest single trade volume in bar."""
        from src.data.trade_reconstructor import reconstruct_candles

        trades_df = self._make_trades_df([
            {"timestamp": 1704067200, "price": 0.50, "volume": 100.0, "side": "buy"},
            {"timestamp": 1704068000, "price": 0.60, "volume": 300.0, "side": "sell"},
            {"timestamp": 1704069000, "price": 0.55, "volume": 50.0, "side": "buy"},
        ])

        result = reconstruct_candles(trades_df)

        assert result.iloc[0]["max_trade_size"] == 300.0

    def test_output_columns_match_candle_columns(self, normalized_trades):
        """Output DataFrame columns match CANDLE_COLUMNS exactly (order and names)."""
        from src.data.trade_reconstructor import reconstruct_candles

        trades_df = self._make_trades_df(normalized_trades)
        result = reconstruct_candles(trades_df)

        assert list(result.columns) == CANDLE_COLUMNS

    def test_has_trade_always_true(self, normalized_trades):
        """has_trade is True for all emitted bars."""
        from src.data.trade_reconstructor import reconstruct_candles

        trades_df = self._make_trades_df(normalized_trades)
        result = reconstruct_candles(trades_df)

        assert result["has_trade"].all()

    def test_bar_timestamp_is_floor_aligned(self):
        """Bar timestamp is floor-aligned to bar boundary."""
        from src.data.trade_reconstructor import reconstruct_candles

        # Trade at 1704081000 (in 4-hour bar starting at 1704067200)
        # 1704081000 // 14400 = 118338 -> 118338 * 14400 = 1704067200
        trades_df = self._make_trades_df([
            {"timestamp": 1704081000, "price": 0.55, "volume": 100.0, "side": "buy"},
        ])

        result = reconstruct_candles(trades_df)

        expected_bar_ts = (1704081000 // 14400) * 14400
        assert result.iloc[0]["timestamp"] == expected_bar_ts

    def test_ohlc_ordering(self):
        """Open is first trade, close is last trade, high/low are extremes."""
        from src.data.trade_reconstructor import reconstruct_candles

        trades_df = self._make_trades_df([
            {"timestamp": 1704067200, "price": 0.50, "volume": 100.0, "side": "buy"},
            {"timestamp": 1704068000, "price": 0.70, "volume": 50.0, "side": "sell"},
            {"timestamp": 1704069000, "price": 0.40, "volume": 80.0, "side": "buy"},
            {"timestamp": 1704070000, "price": 0.55, "volume": 60.0, "side": "sell"},
        ])

        result = reconstruct_candles(trades_df)

        row = result.iloc[0]
        assert row["open"] == 0.50   # first trade
        assert row["close"] == 0.55  # last trade
        assert row["high"] == 0.70   # max price
        assert row["low"] == 0.40    # min price
