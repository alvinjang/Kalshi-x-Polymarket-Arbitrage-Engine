"""Tests for the trade fetcher module (Phase 2.1 Plan 01, Task 1)."""
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, call


class TestFetchKalshiTrades:
    """Test Kalshi trade fetching with cursor pagination."""

    def test_pagination_terminates_on_empty_cursor(self, kalshi_raw_trades_api_response):
        """Cursor pagination fetches all pages and stops when cursor is empty."""
        from src.data.trade_fetcher import fetch_kalshi_trades

        page1, page2 = kalshi_raw_trades_api_response
        client = MagicMock()
        client.get.side_effect = [page1, page2]

        trades = fetch_kalshi_trades("KXU3-26FEB-T4.3", client)

        assert len(trades) == 3
        assert client.get.call_count == 2
        # First call has no cursor, second has cursor
        first_call_params = client.get.call_args_list[0]
        assert first_call_params == call(
            "markets/trades", params={"ticker": "KXU3-26FEB-T4.3", "limit": 200}
        )
        second_call_params = client.get.call_args_list[1]
        assert second_call_params == call(
            "markets/trades",
            params={"ticker": "KXU3-26FEB-T4.3", "limit": 200, "cursor": "page2_cursor"},
        )

    def test_normalization_fields(self, kalshi_raw_trades_api_response):
        """Kalshi trades are normalized: ISO8601->unix int, price->float, side->lowercase."""
        from src.data.trade_fetcher import fetch_kalshi_trades

        page1, page2 = kalshi_raw_trades_api_response
        client = MagicMock()
        client.get.side_effect = [page1, page2]

        trades = fetch_kalshi_trades("KXU3-26FEB-T4.3", client)

        first = trades[0]
        assert isinstance(first["timestamp"], int)
        assert isinstance(first["price"], float)
        assert isinstance(first["volume"], float)
        assert first["side"] in ("yes", "no", "unknown")
        assert set(first.keys()) == {"timestamp", "price", "volume", "side"}

    def test_sorted_by_timestamp(self, kalshi_raw_trades_api_response):
        """Results are sorted by timestamp ascending."""
        from src.data.trade_fetcher import fetch_kalshi_trades

        page1, page2 = kalshi_raw_trades_api_response
        client = MagicMock()
        client.get.side_effect = [page1, page2]

        trades = fetch_kalshi_trades("KXU3-26FEB-T4.3", client)

        timestamps = [t["timestamp"] for t in trades]
        assert timestamps == sorted(timestamps)

    def test_empty_response_returns_empty_list(self):
        """Empty trades from API returns empty list without crash."""
        from src.data.trade_fetcher import fetch_kalshi_trades

        client = MagicMock()
        client.get.return_value = {"trades": [], "cursor": ""}

        trades = fetch_kalshi_trades("NONEXISTENT", client)

        assert trades == []
        assert client.get.call_count == 1


class TestFetchPolymarketTrades:
    """Test Polymarket trade fetching with offset pagination."""

    def test_pagination_terminates_on_short_page(self, polymarket_raw_trades_api_response):
        """Offset pagination stops when fewer than 500 trades are returned."""
        from src.data.trade_fetcher import fetch_polymarket_trades

        page1, page2 = polymarket_raw_trades_api_response
        client = MagicMock()
        # page1 has 2 trades (< 500), so pagination stops after first page
        client.get.return_value = page1

        trades = fetch_polymarket_trades("0xabc123", client)

        assert len(trades) == 2
        assert client.get.call_count == 1

    def test_normalization_fields(self, polymarket_raw_trades_api_response):
        """Polymarket trades are normalized: string->typed, side->lowercase."""
        from src.data.trade_fetcher import fetch_polymarket_trades

        page1, _ = polymarket_raw_trades_api_response
        client = MagicMock()
        client.get.return_value = page1

        trades = fetch_polymarket_trades("0xabc123", client)

        first = trades[0]
        assert isinstance(first["timestamp"], int)
        assert first["timestamp"] == 1704067200
        assert isinstance(first["price"], float)
        assert first["price"] == 0.65
        assert isinstance(first["volume"], float)
        assert first["volume"] == 100.0
        assert first["side"] == "buy"  # normalized from "BUY"
        assert set(first.keys()) == {"timestamp", "price", "volume", "side"}

    def test_empty_response_returns_empty_list(self):
        """Empty trades from API returns empty list without crash."""
        from src.data.trade_fetcher import fetch_polymarket_trades

        client = MagicMock()
        client.get.return_value = []

        trades = fetch_polymarket_trades("0xnonexistent", client)

        assert trades == []

    def test_offset_pagination_max_limit(self):
        """Pagination terminates when offset exceeds MAX_TRADE_OFFSET (15000)."""
        from src.data.trade_fetcher import fetch_polymarket_trades

        client = MagicMock()
        # Always return exactly 500 trades to keep pagination going
        client.get.return_value = [
            {"timestamp": str(i), "price": "0.50", "size": "10", "side": "BUY"}
            for i in range(500)
        ]

        trades = fetch_polymarket_trades("0xabc123", client)

        # 15000 / 500 = 30 pages at offsets 0, 500, 1000, ..., 14500
        # Plus offset 15000 itself = 31 calls total
        assert client.get.call_count == 31
        assert len(trades) == 31 * 500


class TestFetchAndSaveTrades:
    """Test the batch fetch-and-save orchestration."""

    def test_saves_parquet_files(self, tmp_data_dir, sample_accepted_pairs):
        """Fetched trades are saved as per-market parquet files."""
        from src.data.trade_fetcher import fetch_and_save_trades

        kalshi_client = MagicMock()
        poly_client = MagicMock()

        # Mock responses for both platforms
        kalshi_client.get.return_value = {
            "trades": [
                {
                    "created_time": "2024-01-01T00:00:00Z",
                    "yes_price_dollars": "0.55",
                    "count_fp": "100",
                    "taker_side": "yes",
                }
            ],
            "cursor": "",
        }
        poly_client.get.return_value = [
            {"timestamp": "1704067200", "price": "0.65", "size": "100", "side": "BUY"},
        ]

        output_dir = tmp_data_dir / "raw"
        stats = fetch_and_save_trades(sample_accepted_pairs, kalshi_client, poly_client, output_dir)

        # Check files exist
        assert (output_dir / "kalshi" / "KXU3-26FEB-T4.3_trades.parquet").exists()
        assert (output_dir / "polymarket" / "0xabc123_trades.parquet").exists()
        assert stats["kalshi_fetched"] == 2
        assert stats["polymarket_fetched"] == 2

    def test_skips_cached_files(self, tmp_data_dir, sample_accepted_pairs):
        """Already-existing parquet files are skipped (caching)."""
        from src.data.trade_fetcher import fetch_and_save_trades

        kalshi_client = MagicMock()
        poly_client = MagicMock()

        output_dir = tmp_data_dir / "raw"

        # Pre-create a cached file
        cached_path = output_dir / "kalshi" / "KXU3-26FEB-T4.3_trades.parquet"
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp": [1], "price": [0.5], "volume": [10.0], "side": ["buy"]}).to_parquet(cached_path)

        # Mock remaining responses
        kalshi_client.get.return_value = {
            "trades": [
                {
                    "created_time": "2024-01-01T00:00:00Z",
                    "yes_price_dollars": "0.55",
                    "count_fp": "100",
                    "taker_side": "yes",
                }
            ],
            "cursor": "",
        }
        poly_client.get.return_value = [
            {"timestamp": "1704067200", "price": "0.65", "size": "100", "side": "BUY"},
        ]

        stats = fetch_and_save_trades(sample_accepted_pairs, kalshi_client, poly_client, output_dir)

        assert stats["kalshi_cached"] >= 1

    def test_handles_empty_trades(self, tmp_data_dir):
        """Markets with zero trades are counted but no file saved."""
        from src.data.trade_fetcher import fetch_and_save_trades

        kalshi_client = MagicMock()
        poly_client = MagicMock()

        kalshi_client.get.return_value = {"trades": [], "cursor": ""}
        poly_client.get.return_value = []

        pairs = [{"kalshi_market_id": "EMPTY1", "polymarket_market_id": "0xempty"}]
        output_dir = tmp_data_dir / "raw"

        stats = fetch_and_save_trades(pairs, kalshi_client, poly_client, output_dir)

        assert stats["kalshi_empty"] == 1
        assert stats["polymarket_empty"] == 1
        assert not (output_dir / "kalshi" / "EMPTY1_trades.parquet").exists()
