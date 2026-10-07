"""Shared test fixtures for data ingestion tests."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
import tempfile


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary directory mimicking data/raw/ structure."""
    kalshi_dir = tmp_path / "raw" / "kalshi"
    kalshi_dir.mkdir(parents=True)
    poly_dir = tmp_path / "raw" / "polymarket"
    poly_dir.mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_kalshi_candlestick_response():
    """Sample Kalshi historical candlestick API response."""
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
            },
            {
                "end_period_ts": 1704067260,
                "period_interval": 1,
                "price": {"open": None, "high": None, "low": None, "close": None},
                "yes_bid": {"close": "0.57"},
                "yes_ask": {"close": "0.61"},
                "volume": "0",
                "open_interest": "1000",
            },
        ]
    }


@pytest.fixture
def mock_kalshi_null_ohlc_response():
    """Kalshi response where ALL candlesticks have null trade prices (common for low-liquidity)."""
    return {
        "candlesticks": [
            {
                "end_period_ts": 1704067200 + i * 60,
                "period_interval": 1,
                "price": {"open": None, "high": None, "low": None, "close": None},
                "yes_bid": {"close": str(0.50 + i * 0.01)},
                "yes_ask": {"close": str(0.55 + i * 0.01)},
                "volume": "0",
                "open_interest": "500",
            }
            for i in range(5)
        ]
    }


@pytest.fixture
def mock_kalshi_series_response():
    """Sample Kalshi /series API response."""
    return {
        "series": [
            {
                "ticker": "KXBTC",
                "title": "Bitcoin Price",
                "category": "Crypto",
            },
            {
                "ticker": "KXCPI",
                "title": "CPI Report",
                "category": "Economics",
            },
        ]
    }


@pytest.fixture
def mock_kalshi_events_response():
    """Sample Kalshi /events API response with nested markets."""
    return {
        "events": [
            {
                "event_ticker": "KXBTC-25DEC31",
                "markets": [
                    {
                        "ticker": "KXBTC-25DEC31-T50000",
                        "event_ticker": "KXBTC-25DEC31",
                        "title": "Will Bitcoin exceed $50,000 by Dec 31?",
                        "status": "settled",
                        "result": "yes",
                        "close_time": "2025-12-31T23:59:59Z",
                    }
                ],
            }
        ],
        "cursor": "",
    }


@pytest.fixture
def mock_kalshi_cutoff_response():
    """Sample Kalshi /historical/cutoff response."""
    return {
        "market_settled_ts": "2026-01-01T00:00:00Z",
        "orders_updated_ts": "2026-01-01T00:00:00Z",
        "trades_created_ts": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def mock_polymarket_gamma_events_response():
    """Sample Polymarket Gamma API /events response."""
    return [
        {
            "id": "12345",
            "title": "Will Bitcoin reach $100k by March 2026?",
            "description": "This market resolves Yes if BTC price exceeds $100,000.",
            "closed": True,
            "markets": [
                {
                    "conditionId": "0xabc123",
                    "question": "Will Bitcoin reach $100k by March 2026?",
                    "slug": "will-bitcoin-reach-100k-march-2026",
                    "clobTokenIds": json.dumps(["71321095738111", "71321095738222"]),
                    "endDate": "2026-03-31",
                    "volumeNum": 500000,
                    "outcomes": '["Yes", "No"]',
                    "description": "Resolves based on Coinbase BTC/USD price.",
                    "resolutionSource": "Coinbase",
                }
            ],
        }
    ]


@pytest.fixture
def mock_polymarket_clob_prices_response():
    """Sample Polymarket CLOB /prices-history response."""
    return {
        "history": [
            {"t": 1704067200, "p": "0.65"},
            {"t": 1704070800, "p": "0.68"},
            {"t": 1704074400, "p": "0.62"},
            {"t": 1704078000, "p": "0.70"},
        ]
    }


@pytest.fixture
def mock_polymarket_trades_response():
    """Sample Polymarket Data API /trades response."""
    return [
        {"id": "1", "price": "0.65", "size": "100", "timestamp": "1704067200", "side": "BUY"},
        {"id": "2", "price": "0.66", "size": "50", "timestamp": "1704067500", "side": "SELL"},
        {"id": "3", "price": "0.68", "size": "200", "timestamp": "1704070900", "side": "BUY"},
        {"id": "4", "price": "0.62", "size": "75", "timestamp": "1704074500", "side": "SELL"},
    ]


# -------------------------------------------------------------------------
# Trade fetcher / reconstructor fixtures (Phase 2.1)
# -------------------------------------------------------------------------

@pytest.fixture
def kalshi_raw_trades_api_response():
    """Raw Kalshi /markets/trades API response (before normalization).

    Simulates two pages of cursor-paginated results.
    """
    page1 = {
        "trades": [
            {
                "created_time": "2024-01-01T00:00:00Z",
                "yes_price_dollars": "0.55",
                "count_fp": "100",
                "taker_side": "yes",
            },
            {
                "created_time": "2024-01-01T01:00:00Z",
                "yes_price_dollars": "0.60",
                "count_fp": "50",
                "taker_side": "no",
            },
        ],
        "cursor": "page2_cursor",
    }
    page2 = {
        "trades": [
            {
                "created_time": "2024-01-01T04:00:00Z",
                "yes_price_dollars": "0.58",
                "count_fp": "75",
                "taker_side": "yes",
            },
        ],
        "cursor": "",
    }
    return [page1, page2]


@pytest.fixture
def polymarket_raw_trades_api_response():
    """Raw Polymarket Data API /trades response (before normalization).

    Simulates two pages of offset-paginated results.
    """
    page1 = [
        {"timestamp": "1704067200", "price": "0.65", "size": "100", "side": "BUY"},
        {"timestamp": "1704067500", "price": "0.66", "size": "50", "side": "SELL"},
    ]
    page2 = [
        {"timestamp": "1704070900", "price": "0.68", "size": "200", "side": "BUY"},
    ]
    return [page1, page2]


@pytest.fixture
def normalized_trades():
    """Normalized trades (output of fetch_*_trades) for reconstructor tests.

    Spans two 4-hour bars:
    - Bar 1 (ts 1704067200..1704081599): 3 trades
    - Bar 2 (ts 1704081600..1704095999): 2 trades
    """
    return [
        {"timestamp": 1704067200, "price": 0.50, "volume": 100.0, "side": "buy"},
        {"timestamp": 1704070000, "price": 0.60, "volume": 200.0, "side": "sell"},
        {"timestamp": 1704075000, "price": 0.55, "volume": 50.0, "side": "buy"},
        {"timestamp": 1704082000, "price": 0.62, "volume": 150.0, "side": "buy"},
        {"timestamp": 1704090000, "price": 0.58, "volume": 80.0, "side": "sell"},
    ]


@pytest.fixture
def sample_accepted_pairs():
    """Minimal accepted pairs for fetch_and_save_trades testing."""
    return [
        {
            "kalshi_market_id": "KXU3-26FEB-T4.3",
            "polymarket_market_id": "0xabc123",
            "pair_id": "kxu3-0xabc123",
        },
        {
            "kalshi_market_id": "KXUE-JPN26FEB-2.1",
            "polymarket_market_id": "0xdef456",
            "pair_id": "kxue-0xdef456",
        },
    ]
