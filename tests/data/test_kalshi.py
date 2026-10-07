"""Unit tests for KalshiAdapter: historical cutoff routing, market discovery, candlestick parsing."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from src.data.kalshi import KalshiAdapter
from src.data.schemas import MarketMetadata, CANDLESTICK_COLUMNS


# ---------------------------------------------------------------------------
# Historical cutoff tests
# ---------------------------------------------------------------------------
class TestKalshiHistoricalCutoff:
    """Tests for _get_historical_cutoff and endpoint routing."""

    def test_calls_historical_cutoff_endpoint(self):
        """_get_historical_cutoff calls GET /historical/cutoff and returns market_settled_ts."""
        mock_client = Mock()
        mock_client.get.return_value = {
            "market_settled_ts": "2026-01-01T00:00:00Z",
            "orders_updated_ts": "2026-01-01T00:00:00Z",
            "trades_created_ts": "2026-01-01T00:00:00Z",
        }
        adapter = KalshiAdapter(client=mock_client)
        result = adapter._get_historical_cutoff()
        mock_client.get.assert_called_once_with("historical/cutoff")
        assert result == "2026-01-01T00:00:00Z"

    def test_caches_cutoff_result(self):
        """Second call to _get_historical_cutoff does not make another HTTP request."""
        mock_client = Mock()
        mock_client.get.return_value = {
            "market_settled_ts": "2026-01-01T00:00:00Z",
            "orders_updated_ts": "2026-01-01T00:00:00Z",
            "trades_created_ts": "2026-01-01T00:00:00Z",
        }
        adapter = KalshiAdapter(client=mock_client)
        adapter._get_historical_cutoff()
        adapter._get_historical_cutoff()
        assert mock_client.get.call_count == 1  # Only one HTTP call

    def test_routes_pre_cutoff_market_to_historical_endpoint(self):
        """Market with close_time before cutoff uses /historical/markets/{ticker}/candlesticks."""
        mock_client = Mock()

        def side_effect(path, **kwargs):
            if path == "historical/cutoff":
                return {"market_settled_ts": "2026-01-01T00:00:00Z"}
            # Any candlestick endpoint
            return {"candlesticks": []}

        mock_client.get.side_effect = side_effect
        adapter = KalshiAdapter(client=mock_client)
        adapter._get_historical_cutoff()  # Prime cache

        adapter.get_candlesticks(
            "TICKER-123", close_time="2025-06-15T00:00:00Z", start_ts=0, end_ts=100
        )
        # Verify historical endpoint was called
        calls = [c for c in mock_client.get.call_args_list if "historical/markets" in str(c)]
        assert len(calls) > 0, "Expected call to historical/markets/ endpoint"

    def test_routes_post_cutoff_market_to_live_endpoint(self):
        """Market with close_time after cutoff uses /series/{series}/markets/{ticker}/candlesticks."""
        mock_client = Mock()

        def side_effect(path, **kwargs):
            if path == "historical/cutoff":
                return {"market_settled_ts": "2026-01-01T00:00:00Z"}
            return {"candlesticks": []}

        mock_client.get.side_effect = side_effect
        adapter = KalshiAdapter(client=mock_client)
        adapter._get_historical_cutoff()
        adapter._market_series = {"TICKER-456": "SERIES-A"}

        adapter.get_candlesticks(
            "TICKER-456", close_time="2026-03-15T00:00:00Z", start_ts=0, end_ts=100
        )
        calls = [c for c in mock_client.get.call_args_list if "series/" in str(c)]
        assert len(calls) > 0, "Expected call to series/ (live) endpoint"

    def test_ingest_all_calls_cutoff_first(self):
        """ingest_all calls _get_historical_cutoff before processing markets."""
        mock_client = Mock()
        call_order = []

        def track_calls(path, **kwargs):
            call_order.append(path)
            if path == "historical/cutoff":
                return {"market_settled_ts": "2026-01-01T00:00:00Z"}
            if path == "series":
                return {"series": []}
            return {}

        mock_client.get.side_effect = track_calls
        adapter = KalshiAdapter(client=mock_client)
        adapter.ingest_all(["Economics"], Path("/tmp/test"))
        assert call_order[0] == "historical/cutoff"


# ---------------------------------------------------------------------------
# Market discovery tests
# ---------------------------------------------------------------------------
class TestKalshiListMarkets:
    """Tests for list_markets via /series -> /events chain."""

    def test_discovers_markets_via_series_events(
        self, mock_kalshi_series_response, mock_kalshi_events_response
    ):
        """list_markets chains /series -> /events and returns market metadata."""
        mock_client = Mock()

        def side_effect(path, **kwargs):
            params = kwargs.get("params", {})
            if path == "series":
                return mock_kalshi_series_response
            if path == "events":
                return mock_kalshi_events_response
            return {}

        mock_client.get.side_effect = side_effect
        adapter = KalshiAdapter(client=mock_client)
        markets = adapter.list_markets(["Crypto"])

        assert len(markets) >= 1
        assert all(isinstance(m, MarketMetadata) for m in markets)

    def test_returns_market_metadata_with_kalshi_platform(
        self, mock_kalshi_series_response, mock_kalshi_events_response
    ):
        """Each MarketMetadata has platform='kalshi'."""
        mock_client = Mock()

        def side_effect(path, **kwargs):
            if path == "series":
                return mock_kalshi_series_response
            if path == "events":
                return mock_kalshi_events_response
            return {}

        mock_client.get.side_effect = side_effect
        adapter = KalshiAdapter(client=mock_client)
        markets = adapter.list_markets(["Crypto"])

        for m in markets:
            assert m.platform == "kalshi"
            assert m.market_id != ""

    def test_handles_cursor_pagination(self):
        """list_markets follows cursor pagination until empty cursor."""
        mock_client = Mock()
        page1 = {
            "events": [
                {
                    "event_ticker": "EVT1",
                    "markets": [
                        {
                            "ticker": "MKT-1",
                            "title": "Market 1",
                            "status": "settled",
                            "result": "yes",
                            "close_time": "2025-12-31T23:59:59Z",
                        }
                    ],
                }
            ],
            "cursor": "next_page_token",
        }
        page2 = {
            "events": [
                {
                    "event_ticker": "EVT2",
                    "markets": [
                        {
                            "ticker": "MKT-2",
                            "title": "Market 2",
                            "status": "settled",
                            "result": "no",
                            "close_time": "2025-12-31T23:59:59Z",
                        }
                    ],
                }
            ],
            "cursor": "",
        }
        call_count = [0]

        def side_effect(path, **kwargs):
            if path == "series":
                return {"series": [{"ticker": "SER1", "title": "S1", "category": "Crypto"}]}
            if path == "events":
                call_count[0] += 1
                return page1 if call_count[0] == 1 else page2
            return {}

        mock_client.get.side_effect = side_effect
        adapter = KalshiAdapter(client=mock_client)
        markets = adapter.list_markets(["Crypto"])

        assert len(markets) == 2
        assert markets[0].market_id == "MKT-1"
        assert markets[1].market_id == "MKT-2"


# ---------------------------------------------------------------------------
# Candlestick parsing tests
# ---------------------------------------------------------------------------
class TestKalshiGetCandlesticks:
    """Tests for get_candlesticks: parsing, null OHLC, chunking."""

    def test_parses_valid_ohlc(self, mock_kalshi_candlestick_response):
        """Valid OHLC fields are parsed into correct DataFrame columns."""
        mock_client = Mock()

        def side_effect(path, **kwargs):
            if path == "historical/cutoff":
                return {"market_settled_ts": "2026-01-01T00:00:00Z"}
            return mock_kalshi_candlestick_response

        mock_client.get.side_effect = side_effect
        adapter = KalshiAdapter(client=mock_client)
        adapter._get_historical_cutoff()

        df = adapter.get_candlesticks(
            "TEST-MKT", close_time="2025-06-01T00:00:00Z", start_ts=0, end_ts=100
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1
        # First candle has valid trade data
        first_row = df.iloc[0]
        assert first_row["close"] == 0.58
        assert first_row["open"] == 0.55
        assert first_row["high"] == 0.60
        assert first_row["low"] == 0.50

    def test_null_ohlc_uses_bid_ask_midpoint(self, mock_kalshi_null_ohlc_response):
        """When trade close is null, close = (yes_bid_close + yes_ask_close) / 2."""
        mock_client = Mock()

        def side_effect(path, **kwargs):
            if path == "historical/cutoff":
                return {"market_settled_ts": "2026-01-01T00:00:00Z"}
            return mock_kalshi_null_ohlc_response

        mock_client.get.side_effect = side_effect
        adapter = KalshiAdapter(client=mock_client)
        adapter._get_historical_cutoff()

        df = adapter.get_candlesticks(
            "LOW-LIQ", close_time="2025-06-01T00:00:00Z", start_ts=0, end_ts=100
        )

        assert len(df) == 5
        # First candle: bid=0.50, ask=0.55, midpoint=0.525
        assert df.iloc[0]["close"] == pytest.approx(0.525, abs=0.001)
        # All have has_trades=False
        assert all(not row["has_trades"] for _, row in df.iterrows())

    def test_fully_null_candlestick(self):
        """When both trade and bid/ask are absent, close is None."""
        mock_client = Mock()
        response = {
            "candlesticks": [
                {
                    "end_period_ts": 1704067200,
                    "period_interval": 1,
                    "price": {"open": None, "high": None, "low": None, "close": None},
                    "volume": "0",
                    "open_interest": "0",
                }
            ]
        }

        def side_effect(path, **kwargs):
            if path == "historical/cutoff":
                return {"market_settled_ts": "2026-01-01T00:00:00Z"}
            return response

        mock_client.get.side_effect = side_effect
        adapter = KalshiAdapter(client=mock_client)
        adapter._get_historical_cutoff()

        df = adapter.get_candlesticks(
            "EMPTY-MKT", close_time="2025-06-01T00:00:00Z", start_ts=0, end_ts=100
        )

        assert len(df) == 1
        assert pd.isna(df.iloc[0]["close"])
        assert df.iloc[0]["has_trades"] == False  # noqa: E712 (numpy bool)

    def test_has_trades_flag(self, mock_kalshi_candlestick_response):
        """has_trades is True when price.close is not None, False otherwise."""
        mock_client = Mock()

        def side_effect(path, **kwargs):
            if path == "historical/cutoff":
                return {"market_settled_ts": "2026-01-01T00:00:00Z"}
            return mock_kalshi_candlestick_response

        mock_client.get.side_effect = side_effect
        adapter = KalshiAdapter(client=mock_client)
        adapter._get_historical_cutoff()

        df = adapter.get_candlesticks(
            "TEST-MKT", close_time="2025-06-01T00:00:00Z", start_ts=0, end_ts=100
        )

        # First candle has trade data (close=0.58), second has null trade price
        assert df.iloc[0]["has_trades"] == True  # noqa: E712 (numpy bool)
        assert df.iloc[1]["has_trades"] == False  # noqa: E712 (numpy bool)

    def test_time_window_chunking(self):
        """Market spanning >4500 minutes triggers multiple API requests."""
        mock_client = Mock()
        chunk_call_count = [0]

        def side_effect(path, **kwargs):
            if path == "historical/cutoff":
                return {"market_settled_ts": "2026-01-01T00:00:00Z"}
            # Count candlestick fetches
            if "candlesticks" in path:
                chunk_call_count[0] += 1
            return {"candlesticks": []}

        mock_client.get.side_effect = side_effect
        adapter = KalshiAdapter(client=mock_client)
        adapter._get_historical_cutoff()

        # Span that requires 2+ chunks: >4500 minutes = >270000 seconds
        adapter.get_candlesticks(
            "BIG-MKT",
            close_time="2025-06-01T00:00:00Z",
            start_ts=0,
            end_ts=600000,  # 600000 seconds > 270000, so at least 3 chunks
        )

        assert chunk_call_count[0] >= 3, (
            f"Expected at least 3 chunk requests for 600000s span, got {chunk_call_count[0]}"
        )

    def test_returns_dataframe_with_required_columns(self, mock_kalshi_candlestick_response):
        """Returned DataFrame has all CANDLESTICK_COLUMNS plus optional columns."""
        mock_client = Mock()

        def side_effect(path, **kwargs):
            if path == "historical/cutoff":
                return {"market_settled_ts": "2026-01-01T00:00:00Z"}
            return mock_kalshi_candlestick_response

        mock_client.get.side_effect = side_effect
        adapter = KalshiAdapter(client=mock_client)
        adapter._get_historical_cutoff()

        df = adapter.get_candlesticks(
            "TEST-MKT", close_time="2025-06-01T00:00:00Z", start_ts=0, end_ts=100
        )

        for col in CANDLESTICK_COLUMNS:
            assert col in df.columns, f"Missing required column: {col}"
        # Also has optional columns
        assert "has_trades" in df.columns
        assert "yes_bid_close" in df.columns
        assert "yes_ask_close" in df.columns
        assert "open_interest" in df.columns


# ---------------------------------------------------------------------------
# Caching tests
# ---------------------------------------------------------------------------
class TestKalshiCaching:
    """Tests for get_or_fetch_candlesticks caching behavior."""

    def test_cache_hit_skips_fetch(self, tmp_data_dir):
        """When parquet file exists, no API call is made."""
        mock_client = Mock()
        adapter = KalshiAdapter(client=mock_client)

        # Create a cached parquet file
        cache_dir = tmp_data_dir / "raw" / "kalshi"
        df_cached = pd.DataFrame(
            {
                "timestamp": [1704067200],
                "open": [0.55],
                "high": [0.60],
                "low": [0.50],
                "close": [0.58],
                "volume": [150.0],
            }
        )
        df_cached.to_parquet(cache_dir / "CACHED-MKT.parquet", index=False)

        result = adapter.get_or_fetch_candlesticks("CACHED-MKT", cache_dir)

        assert result is not None
        assert len(result) == 1
        # get_candlesticks should NOT have been called
        # (The mock_client.get was never called for candlestick data)
        candlestick_calls = [
            c for c in mock_client.get.call_args_list if "candlestick" in str(c)
        ]
        assert len(candlestick_calls) == 0

    def test_cache_miss_writes_parquet(self, tmp_data_dir):
        """On cache miss, fetches data, validates, writes parquet, and returns DataFrame."""
        mock_client = Mock()

        def side_effect(path, **kwargs):
            if path == "historical/cutoff":
                return {"market_settled_ts": "2026-01-01T00:00:00Z"}
            return {
                "candlesticks": [
                    {
                        "end_period_ts": 1704067200,
                        "period_interval": 1,
                        "price": {"open": "0.55", "high": "0.60", "low": "0.50", "close": "0.58"},
                        "yes_bid": {"close": "0.56"},
                        "yes_ask": {"close": "0.60"},
                        "volume": "150",
                        "open_interest": "1000",
                    }
                ]
            }

        mock_client.get.side_effect = side_effect
        adapter = KalshiAdapter(client=mock_client)
        adapter._get_historical_cutoff()

        cache_dir = tmp_data_dir / "raw" / "kalshi"
        result = adapter.get_or_fetch_candlesticks(
            "NEW-MKT", cache_dir, close_time="2025-06-01T00:00:00Z"
        )

        assert result is not None
        assert len(result) == 1
        # Parquet file should now exist
        assert (cache_dir / "NEW-MKT.parquet").exists()
