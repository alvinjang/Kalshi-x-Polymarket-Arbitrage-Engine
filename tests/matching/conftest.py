"""Shared fixtures for matching pipeline tests."""
import pytest


@pytest.fixture(scope="module")
def semantic_matcher():
    """Load SemanticMatcher once per test module (model loading is expensive)."""
    from src.matching.semantic_matcher import SemanticMatcher
    return SemanticMatcher()


@pytest.fixture()
def mock_kalshi_settlement_response():
    """Mock Kalshi API response with settlement rules."""
    return {
        "market": {
            "rules_primary": "Resolves based on CoinDesk BTC Index at midnight ET.",
            "rules_secondary": "If CoinDesk unavailable, uses Coinbase spot price.",
        }
    }


@pytest.fixture()
def mock_poly_settlement_response():
    """Mock Polymarket Gamma API response with settlement info."""
    return [
        {
            "description": "Resolves Yes if BTC/USD >= $80,000 at any point before Jan 1, 2026 UTC",
            "resolutionSource": "https://coinbase.com",
        }
    ]


@pytest.fixture()
def sample_scored_candidates():
    """Sample scored candidate dicts (output format of score_and_rank_candidates)."""
    return [
        {
            "kalshi_market_id": "KXBTC-25DEC31-T80000",
            "polymarket_market_id": "0xabc123",
            "kalshi_question": "Will Bitcoin exceed $80,000 by December 31, 2025?",
            "polymarket_question": "Bitcoin above 80k by end of 2025",
            "category": "Crypto",
            "kalshi_resolution_date": "2025-12-31T23:59:59Z",
            "polymarket_resolution_date": "2026-01-01T00:00:00Z",
            "keyword_score": 0.35,
            "semantic_score": 0.78,
            "confidence_score": 0.651,
        },
        {
            "kalshi_market_id": "KXETH-25DEC31-T5000",
            "polymarket_market_id": "0xdef456",
            "kalshi_question": "Will Ethereum exceed $5,000 by December 31, 2025?",
            "polymarket_question": "Will ETH hit $5,000 before 2026?",
            "category": "Crypto",
            "kalshi_resolution_date": "2025-12-31T23:59:59Z",
            "polymarket_resolution_date": "2025-12-31T23:59:59Z",
            "keyword_score": 0.40,
            "semantic_score": 0.85,
            "confidence_score": 0.715,
        },
        {
            "kalshi_market_id": "KXCPI-26FEB-T3",
            "polymarket_market_id": "0xghi789",
            "kalshi_question": "Will CPI be above 3% in February 2026?",
            "polymarket_question": "CPI above 3 percent in Feb 2026",
            "category": "Economics",
            "kalshi_resolution_date": "2026-03-15T00:00:00Z",
            "polymarket_resolution_date": "2026-03-15T00:00:00Z",
            "keyword_score": 0.30,
            "semantic_score": 0.90,
            "confidence_score": 0.720,
        },
    ]


@pytest.fixture()
def sample_kalshi_markets():
    """Sample Kalshi market metadata dicts for testing."""
    return [
        {
            "market_id": "KXBTC-25DEC31-T80000",
            "question": "Will Bitcoin exceed $80,000 by December 31, 2025?",
            "category": "Crypto",
            "platform": "kalshi",
            "resolution_date": "2025-12-31T23:59:59Z",
            "result": None,
            "outcomes": ["Yes", "No"],
        },
        {
            "market_id": "KXETH-25DEC31-T5000",
            "question": "Will Ethereum exceed $5,000 by December 31, 2025?",
            "category": "Crypto",
            "platform": "kalshi",
            "resolution_date": "2025-12-31T23:59:59Z",
            "result": None,
            "outcomes": ["Yes", "No"],
        },
        {
            "market_id": "KXCPI-26FEB-T3",
            "question": "Will CPI be above 3% in February 2026?",
            "category": "Economics",
            "platform": "kalshi",
            "resolution_date": "2026-03-15T00:00:00Z",
            "result": None,
            "outcomes": ["Yes", "No"],
        },
        {
            "market_id": "KXFED-26MAR-TCUT",
            "question": "Will the Federal Reserve cut rates in March 2026?",
            "category": "Economics",
            "platform": "kalshi",
            "resolution_date": "2026-03-20T00:00:00Z",
            "result": None,
            "outcomes": ["Yes", "No"],
        },
    ]


@pytest.fixture()
def sample_poly_markets():
    """Sample Polymarket market metadata dicts for testing."""
    return [
        {
            "market_id": "0xabc123",
            "question": "Bitcoin above 80k by end of 2025",
            "category": "crypto",
            "platform": "polymarket",
            "resolution_date": "2026-01-01T00:00:00Z",
            "result": None,
            "outcomes": ["Yes", "No"],
            "clob_token_ids": ["12345"],
        },
        {
            "market_id": "0xdef456",
            "question": "Will ETH hit $5,000 before 2026?",
            "category": "crypto",
            "platform": "polymarket",
            "resolution_date": "2025-12-31T23:59:59Z",
            "result": None,
            "outcomes": ["Yes", "No"],
            "clob_token_ids": ["67890"],
        },
        {
            "market_id": "0xghi789",
            "question": "CPI above 3 percent in Feb 2026",
            "category": "finance",
            "platform": "polymarket",
            "resolution_date": "2026-03-15T00:00:00Z",
            "result": None,
            "outcomes": ["Yes", "No"],
            "clob_token_ids": ["11111"],
        },
        {
            "market_id": "0xjkl012",
            "question": "Fed rate cut in March 2026?",
            "category": "finance",
            "platform": "polymarket",
            "resolution_date": "2026-03-20T00:00:00Z",
            "result": None,
            "outcomes": ["Yes", "No"],
            "clob_token_ids": ["22222"],
        },
    ]
