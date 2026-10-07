"""Tests for PolymarketAdapter: market discovery, price fetching, and caching."""
import json
import time
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from src.data.schemas import MarketMetadata, CANDLESTICK_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gamma_event(
    title="Will Bitcoin reach $100k by March 2026?",
    description="This market resolves Yes if BTC price exceeds $100,000.",
    condition_id="0xabc123",
    clob_token_ids=None,
    end_date="2026-03-31",
    outcomes=None,
):
    """Build a single Gamma event dict with one nested market."""
    if clob_token_ids is None:
        clob_token_ids = ["71321095738111", "71321095738222"]
    if outcomes is None:
        outcomes = ["Yes", "No"]
    return {
        "id": "12345",
        "title": title,
        "description": description,
        "closed": True,
        "markets": [
            {
                "conditionId": condition_id,
                "question": title,
                "slug": "will-bitcoin-reach-100k",
                "clobTokenIds": json.dumps(clob_token_ids),
                "endDate": end_date,
                "volumeNum": 500000,
                "outcomes": json.dumps(outcomes),
                "description": description,
                "resolutionSource": "Coinbase",
            }
        ],
    }


def _make_adapter(gamma_responses=None, clob_responses=None, data_responses=None):
    """Build a PolymarketAdapter with mocked clients.

    Each *_responses is a list of return values for successive .get() calls.
    """
    from src.data.polymarket import PolymarketAdapter

    gamma = MagicMock()
    clob = MagicMock()
    data = MagicMock()

    if gamma_responses is not None:
        gamma.get.side_effect = gamma_responses
    if clob_responses is not None:
        clob.get.side_effect = clob_responses
    if data_responses is not None:
        data.get.side_effect = data_responses

    adapter = PolymarketAdapter(
        gamma_client=gamma, clob_client=clob, data_client=data
    )
    return adapter, gamma, clob, data


# ===========================================================================
# list_markets
# ===========================================================================


class TestPolymarketListMarkets:
    """Tests for Gamma API market discovery with keyword filtering."""

    def test_keyword_filtering(self, mock_polymarket_gamma_events_response):
        """Events matching crypto keywords are discovered."""
        adapter, gamma, _, _ = _make_adapter(
            gamma_responses=[
                mock_polymarket_gamma_events_response,  # page 1 (< 100 -> stop)
            ]
        )
        markets = adapter.list_markets(["crypto"])
        assert len(markets) == 1
        assert markets[0].question == "Will Bitcoin reach $100k by March 2026?"

    def test_extracts_clob_token_ids(self, mock_polymarket_gamma_events_response):
        """clobTokenIds (stringified JSON) are parsed and stored in token map."""
        adapter, gamma, _, _ = _make_adapter(
            gamma_responses=[mock_polymarket_gamma_events_response]
        )
        markets = adapter.list_markets(["crypto"])
        token_ids = adapter._market_token_map.get("0xabc123", [])
        assert token_ids == ["71321095738111", "71321095738222"]

    def test_returns_polymarket_platform(self, mock_polymarket_gamma_events_response):
        """MarketMetadata has platform='polymarket' and market_id=conditionId."""
        adapter, *_ = _make_adapter(
            gamma_responses=[mock_polymarket_gamma_events_response]
        )
        markets = adapter.list_markets(["crypto"])
        m = markets[0]
        assert m.platform == "polymarket"
        assert m.market_id == "0xabc123"
        assert isinstance(m, MarketMetadata)

    def test_pagination_stops_on_short_response(self):
        """Pagination stops when fewer than 100 events are returned."""
        page1 = [_make_gamma_event(title=f"Bitcoin event {i}") for i in range(100)]
        page2 = [_make_gamma_event(title="Final bitcoin event", condition_id="0xfinal")]
        adapter, gamma, _, _ = _make_adapter(gamma_responses=[page1, page2])

        adapter.list_markets(["crypto"])
        # Should have made exactly 2 calls (page1 had 100 -> continue, page2 had 1 -> stop)
        assert gamma.get.call_count == 2

    def test_ignores_non_matching_events(self):
        """Events without matching keywords are skipped."""
        events = [
            _make_gamma_event(
                title="Will it rain tomorrow?",
                description="Weather prediction market.",
                condition_id="0xweather",
            )
        ]
        adapter, *_ = _make_adapter(gamma_responses=[events])
        markets = adapter.list_markets(["crypto"])
        assert len(markets) == 0

    def test_finance_keyword_match(self):
        """Finance keywords like 'inflation' match correctly."""
        events = [
            _make_gamma_event(
                title="Will CPI exceed 3% in March?",
                description="Consumer price index inflation report.",
                condition_id="0xcpi",
            )
        ]
        adapter, *_ = _make_adapter(gamma_responses=[events])
        markets = adapter.list_markets(["finance"])
        assert len(markets) == 1
        assert markets[0].category == "finance"

    def test_default_categories(self):
        """Passing None for categories defaults to all keyword categories."""
        events = [
            _make_gamma_event(title="Bitcoin price prediction", condition_id="0xbtc"),
        ]
        adapter, *_ = _make_adapter(gamma_responses=[events])
        markets = adapter.list_markets(None)
        assert len(markets) >= 1


# ===========================================================================
# get_candlesticks — CLOB prices-history
# ===========================================================================


class TestPolymarketGetCandlesticks:
    """Tests for CLOB /prices-history fetching with chunking and fallback."""

    def test_clob_prices_history_with_chunks(self, mock_polymarket_clob_prices_response):
        """CLOB /prices-history is called with startTs/endTs chunking."""
        adapter, _, clob, _ = _make_adapter(
            clob_responses=[mock_polymarket_clob_prices_response]
        )
        adapter._market_token_map["0xabc"] = ["token_yes", "token_no"]

        # Use a small time window that fits in one chunk
        df = adapter.get_candlesticks(
            "0xabc",
            start_ts=1704067200,
            end_ts=1704067200 + 86400,  # 1 day (< 14 day chunk)
        )

        # Verify CLOB was called with correct params
        call_args = clob.get.call_args
        assert call_args[0][0] == "prices-history"
        params = call_args[1].get("params") or call_args[0][1] if len(call_args[0]) > 1 else call_args[1]["params"]
        assert "startTs" in params or "startTs" in str(params)

    def test_tries_both_tokens(self):
        """If Yes token returns empty, tries No token."""
        empty_response = {"history": []}
        no_token_response = {
            "history": [{"t": 1704067200, "p": "0.35"}]
        }
        adapter, _, clob, _ = _make_adapter(
            clob_responses=[empty_response, no_token_response]
        )
        adapter._market_token_map["0xabc"] = ["token_yes", "token_no"]

        df = adapter.get_candlesticks(
            "0xabc", start_ts=1704067200, end_ts=1704067200 + 86400
        )
        assert len(df) == 1
        # Should have tried both tokens
        assert clob.get.call_count == 2

    def test_parses_clob_response_format(self, mock_polymarket_clob_prices_response):
        """CLOB response {history: [{t: int, p: str}]} is parsed into DataFrame."""
        adapter, _, clob, _ = _make_adapter(
            clob_responses=[mock_polymarket_clob_prices_response]
        )
        adapter._market_token_map["0xabc"] = ["token_yes", "token_no"]

        df = adapter.get_candlesticks(
            "0xabc", start_ts=1704067200, end_ts=1704067200 + 86400
        )
        assert not df.empty
        # Check required columns
        for col in CANDLESTICK_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"
        # Check types
        assert df["close"].dtype == float
        assert df["timestamp"].dtype in (int, "int64")

    def test_falls_back_to_trades(self, mock_polymarket_trades_response):
        """Falls back to Data API /trades when CLOB returns empty."""
        empty_clob = {"history": []}
        adapter, _, clob, data = _make_adapter(
            clob_responses=[empty_clob, empty_clob],  # Both tokens empty
            data_responses=[mock_polymarket_trades_response, []],  # trades then empty
        )
        adapter._market_token_map["0xabc"] = ["token_yes", "token_no"]

        df = adapter.get_candlesticks(
            "0xabc", start_ts=1704067200, end_ts=1704067200 + 86400
        )
        # Data API should have been called
        assert data.get.call_count >= 1
        assert not df.empty

    def test_trades_offset_limit(self):
        """Data API fallback respects 3000 max offset (stops pagination)."""
        # Build responses: 7 pages of 500 trades each (3500 would exceed 3000 offset)
        trade_page = [
            {"id": str(i), "price": "0.65", "size": "100", "timestamp": str(1704067200 + i * 60), "side": "BUY"}
            for i in range(500)
        ]
        # After 7 pages at offset 0, 500, 1000, 1500, 2000, 2500, 3000 -> stop
        adapter, _, clob, data = _make_adapter(
            clob_responses=[{"history": []}, {"history": []}],
            data_responses=[trade_page] * 10,  # More pages than should be consumed
        )
        adapter._market_token_map["0xabc"] = ["token_yes", "token_no"]

        df = adapter.get_candlesticks(
            "0xabc", start_ts=1704067200, end_ts=1704067200 + 86400
        )

        # Should stop at offset 3000 -> max 7 pages (0, 500, ..., 3000)
        assert data.get.call_count <= 7

    def test_empty_token_map_returns_empty_df(self):
        """If no clobTokenIds for a market, returns empty DataFrame."""
        adapter, *_ = _make_adapter()
        df = adapter.get_candlesticks("0xunknown")
        assert df.empty


# ===========================================================================
# _trades_to_ohlcv
# ===========================================================================


class TestTradesToOhlcv:
    """Tests for trade aggregation into OHLCV bars."""

    def test_aggregates_trades_to_hourly(self, mock_polymarket_trades_response):
        """Raw trades are aggregated into hourly OHLCV bars."""
        from src.data.polymarket import PolymarketAdapter

        adapter = PolymarketAdapter.__new__(PolymarketAdapter)
        df = adapter._trades_to_ohlcv(mock_polymarket_trades_response, freq_minutes=60)

        assert not df.empty
        # Trades span ~2 hours -> expect at least 1 bar (maybe 2-3 depending on rounding)
        assert len(df) >= 1

    def test_returns_required_columns(self, mock_polymarket_trades_response):
        """Output DataFrame has all CANDLESTICK_COLUMNS."""
        from src.data.polymarket import PolymarketAdapter

        adapter = PolymarketAdapter.__new__(PolymarketAdapter)
        df = adapter._trades_to_ohlcv(mock_polymarket_trades_response, freq_minutes=60)

        for col in CANDLESTICK_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_ohlcv_values_correct(self):
        """OHLCV values are correctly computed from trades within same hour."""
        from src.data.polymarket import PolymarketAdapter

        trades = [
            {"price": "0.50", "size": "100", "timestamp": "1704067200"},  # hour start
            {"price": "0.70", "size": "50", "timestamp": "1704067500"},   # same hour
            {"price": "0.40", "size": "75", "timestamp": "1704067800"},   # same hour
            {"price": "0.60", "size": "200", "timestamp": "1704068000"},  # same hour
        ]
        adapter = PolymarketAdapter.__new__(PolymarketAdapter)
        df = adapter._trades_to_ohlcv(trades, freq_minutes=60)

        row = df.iloc[0]
        assert row["open"] == pytest.approx(0.50)
        assert row["high"] == pytest.approx(0.70)
        assert row["low"] == pytest.approx(0.40)
        assert row["close"] == pytest.approx(0.60)
        assert row["volume"] == pytest.approx(425.0)


# ===========================================================================
# Caching (via base class get_or_fetch_candlesticks)
# ===========================================================================


class TestPolymarketCaching:
    """Tests for file-based caching."""

    def test_cache_hit_skips_fetch(self, tmp_data_dir):
        """Existing parquet file is returned without calling API."""
        from src.data.polymarket import PolymarketAdapter

        output_dir = tmp_data_dir / "raw" / "polymarket"
        # Pre-create a cache file
        cache_df = pd.DataFrame({
            "timestamp": [1704067200],
            "open": [0.55],
            "high": [0.60],
            "low": [0.50],
            "close": [0.58],
            "volume": [150.0],
        })
        cache_path = output_dir / "0xcached.parquet"
        cache_df.to_parquet(cache_path, index=False)

        adapter, _, clob, data = _make_adapter()
        result = adapter.get_or_fetch_candlesticks("0xcached", output_dir)

        assert result is not None
        assert len(result) == 1
        # API should NOT have been called
        clob.get.assert_not_called()
        data.get.assert_not_called()

    def test_cache_miss_writes_parquet(self, tmp_data_dir):
        """When no cache exists, fetches data and writes parquet file."""
        clob_response = {
            "history": [
                {"t": 1704067200, "p": "0.65"},
                {"t": 1704070800, "p": "0.68"},
            ]
        }
        adapter, _, clob, _ = _make_adapter(clob_responses=[clob_response])
        adapter._market_token_map["0xnew"] = ["token_yes", "token_no"]

        output_dir = tmp_data_dir / "raw" / "polymarket"
        result = adapter.get_or_fetch_candlesticks("0xnew", output_dir)

        assert result is not None
        assert len(result) == 2
        # Parquet file should now exist
        cache_path = output_dir / "0xnew.parquet"
        assert cache_path.exists()
