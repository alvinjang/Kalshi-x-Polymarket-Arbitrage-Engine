"""Tests for keyword-based matching pipeline (Stage 1)."""
import pytest

from src.matching.keyword_matcher import (
    CATEGORY_MAP,
    STOP_WORDS,
    SYNONYMS,
    extract_key_tokens,
    generate_candidates,
    jaccard_similarity,
    normalize_number,
)


# ---------------------------------------------------------------------------
# normalize_number
# ---------------------------------------------------------------------------

class TestNormalizeNumber:
    """Tests for numeric string normalization."""

    def test_k_suffix(self):
        assert normalize_number("80k") == "80000"

    def test_m_suffix(self):
        assert normalize_number("1.5m") == "1500000"

    def test_b_suffix(self):
        assert normalize_number("2b") == "2000000000"

    def test_plain_number(self):
        assert normalize_number("50000") == "50000"

    def test_non_numeric_passthrough(self):
        assert normalize_number("hello") == "hello"

    def test_decimal_without_suffix(self):
        assert normalize_number("3.5") == "3.5"

    def test_percent(self):
        # After stripping %, "3" is just a number
        result = normalize_number("3%")
        assert result == "3"


# ---------------------------------------------------------------------------
# extract_key_tokens
# ---------------------------------------------------------------------------

class TestExtractKeyTokens:
    """Tests for token extraction with synonyms and normalization."""

    def test_bitcoin_question(self):
        tokens = extract_key_tokens("Will Bitcoin exceed $80,000 by Dec 31?")
        assert "bitcoin" in tokens
        assert "80000" in tokens
        assert "dec" in tokens
        assert "31" in tokens

    def test_stop_words_removed(self):
        tokens = extract_key_tokens("Will the market be above this price?")
        assert "will" not in tokens
        assert "the" not in tokens
        assert "be" not in tokens
        assert "above" not in tokens
        assert "this" not in tokens
        assert "market" not in tokens
        assert "price" not in tokens

    def test_synonym_btc(self):
        tokens = extract_key_tokens("BTC price prediction")
        assert "bitcoin" in tokens
        assert "btc" not in tokens

    def test_synonym_eth(self):
        tokens = extract_key_tokens("ETH will reach $5,000")
        assert "ethereum" in tokens

    def test_synonym_fed(self):
        tokens = extract_key_tokens("Fed rate cut")
        assert "federal_reserve" in tokens

    def test_synonym_cpi(self):
        tokens = extract_key_tokens("CPI above 3 percent")
        assert "consumer_price_index" in tokens

    def test_number_normalization_in_tokens(self):
        tokens = extract_key_tokens("Bitcoin above 80k")
        assert "80000" in tokens

    def test_short_tokens_filtered(self):
        """Tokens shorter than 2 characters should be removed."""
        tokens = extract_key_tokens("a b c hello world")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "hello" in tokens
        assert "world" in tokens


# ---------------------------------------------------------------------------
# jaccard_similarity
# ---------------------------------------------------------------------------

class TestJaccardSimilarity:
    """Tests for Jaccard token overlap scoring."""

    def test_same_entity_high_score(self):
        """Cross-platform BTC 80k markets should have high overlap."""
        score = jaccard_similarity(
            "Will Bitcoin exceed $80,000?",
            "Bitcoin above 80k?",
        )
        assert score > 0.3

    def test_different_entity_low_score(self):
        """BTC vs ETH markets should have low overlap."""
        score = jaccard_similarity(
            "Will Bitcoin exceed $80,000?",
            "Will Ethereum reach $5,000?",
        )
        assert score < 0.15

    def test_empty_input_returns_zero(self):
        assert jaccard_similarity("", "Bitcoin") == 0.0

    def test_both_empty_returns_zero(self):
        assert jaccard_similarity("", "") == 0.0

    def test_identical_text_returns_one(self):
        text = "Will Bitcoin exceed $80,000 by December 31?"
        assert jaccard_similarity(text, text) == 1.0

    def test_moderate_overlap(self):
        """Same topic, different numbers should have moderate overlap."""
        score = jaccard_similarity(
            "Will CPI be above 3% in February 2026?",
            "CPI above 3 percent in Feb 2026",
        )
        assert 0.2 < score < 0.9


# ---------------------------------------------------------------------------
# CATEGORY_MAP
# ---------------------------------------------------------------------------

class TestCategoryMap:
    """Tests for category normalization mapping."""

    def test_economics_maps_to_finance(self):
        assert CATEGORY_MAP["Economics"] == "finance"

    def test_financials_maps_to_finance(self):
        assert CATEGORY_MAP["Financials"] == "finance"

    def test_crypto_maps_to_crypto(self):
        assert CATEGORY_MAP["Crypto"] == "crypto"

    def test_lowercase_finance_identity(self):
        assert CATEGORY_MAP["finance"] == "finance"

    def test_lowercase_crypto_identity(self):
        assert CATEGORY_MAP["crypto"] == "crypto"


# ---------------------------------------------------------------------------
# generate_candidates
# ---------------------------------------------------------------------------

class TestGenerateCandidates:
    """Tests for candidate pair generation with category filtering."""

    def test_category_compatible_pairs(self, sample_kalshi_markets, sample_poly_markets):
        """Crypto Kalshi markets should pair with crypto Polymarket markets."""
        candidates = generate_candidates(sample_kalshi_markets, sample_poly_markets)
        for km, pm, score in candidates:
            km_cat = CATEGORY_MAP.get(km["category"])
            pm_cat = CATEGORY_MAP.get(pm["category"])
            assert km_cat == pm_cat, (
                f"Category mismatch: {km['category']} ({km_cat}) vs {pm['category']} ({pm_cat})"
            )

    def test_cross_category_excluded(self, sample_kalshi_markets, sample_poly_markets):
        """BTC (Crypto) should NOT pair with CPI (finance)."""
        candidates = generate_candidates(
            sample_kalshi_markets, sample_poly_markets, min_keyword_score=0.0,
        )
        for km, pm, score in candidates:
            # BTC market should not pair with CPI market
            if km["market_id"] == "KXBTC-25DEC31-T80000":
                assert pm["category"] == "crypto"

    def test_min_keyword_score_filter(self, sample_kalshi_markets, sample_poly_markets):
        """Higher min_keyword_score should produce fewer candidates."""
        all_candidates = generate_candidates(
            sample_kalshi_markets, sample_poly_markets, min_keyword_score=0.0,
        )
        filtered = generate_candidates(
            sample_kalshi_markets, sample_poly_markets, min_keyword_score=0.2,
        )
        assert len(filtered) <= len(all_candidates)

    def test_output_format(self, sample_kalshi_markets, sample_poly_markets):
        """Each candidate should be (kalshi_dict, poly_dict, keyword_score)."""
        candidates = generate_candidates(sample_kalshi_markets, sample_poly_markets)
        for item in candidates:
            assert isinstance(item, tuple)
            assert len(item) == 3
            km, pm, score = item
            assert isinstance(km, dict)
            assert isinstance(pm, dict)
            assert isinstance(score, float)
            assert km["platform"] == "kalshi"
            assert pm["platform"] == "polymarket"

    def test_sorted_descending(self, sample_kalshi_markets, sample_poly_markets):
        """Candidates should be sorted by keyword_score descending."""
        candidates = generate_candidates(sample_kalshi_markets, sample_poly_markets)
        scores = [c[2] for c in candidates]
        assert scores == sorted(scores, reverse=True)
