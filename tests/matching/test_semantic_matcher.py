"""Tests for sentence-transformer semantic similarity scoring (Stage 2)."""
import numpy as np
import pytest

from src.matching.semantic_matcher import SemanticMatcher, score_candidates


class TestSemanticMatcher:
    """Tests for SemanticMatcher encoding and pairwise similarity."""

    def test_encode_batch_shape(self, semantic_matcher):
        """Encode 3 strings and check output shape is (3, 384)."""
        texts = ["Hello world", "Bitcoin price prediction", "CPI data release"]
        embeddings = semantic_matcher.encode_batch(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    def test_score_pairs_range(self, semantic_matcher):
        """All pairwise scores should be floats in [0, 1]."""
        texts_a = ["Will Bitcoin exceed $80,000?", "CPI above 3%"]
        texts_b = ["Bitcoin above 80k", "Consumer price index over 3 percent"]
        scores = semantic_matcher.score_pairs(texts_a, texts_b)
        assert len(scores) == 2
        for s in scores:
            assert isinstance(s, float)
            assert 0.0 <= s <= 1.0

    def test_similar_pair_high_score(self, semantic_matcher):
        """Semantically similar BTC markets should score > 0.5."""
        scores = semantic_matcher.score_pairs(
            ["Will Bitcoin exceed $80,000 by December 31, 2025?"],
            ["Bitcoin above 80k by end of 2025"],
        )
        assert scores[0] > 0.5

    def test_dissimilar_pair_low_score(self, semantic_matcher):
        """Semantically unrelated markets should score < 0.5."""
        scores = semantic_matcher.score_pairs(
            ["Will Bitcoin exceed $80,000?"],
            ["Will CPI be above 3% in February 2026?"],
        )
        assert scores[0] < 0.5

    def test_score_candidates(self, semantic_matcher, sample_kalshi_markets, sample_poly_markets):
        """score_candidates should add semantic_score to each candidate dict."""
        # Create 2 candidate tuples manually
        candidates = [
            (sample_kalshi_markets[0], sample_poly_markets[0], 0.5),  # BTC pair
            (sample_kalshi_markets[2], sample_poly_markets[2], 0.3),  # CPI pair
        ]
        results = score_candidates(semantic_matcher, candidates)
        assert len(results) == 2
        for r in results:
            assert "semantic_score" in r
            assert isinstance(r["semantic_score"], float)
            assert 0.0 <= r["semantic_score"] <= 1.0
            assert "kalshi_market_id" in r
            assert "polymarket_market_id" in r
            assert "keyword_score" in r
