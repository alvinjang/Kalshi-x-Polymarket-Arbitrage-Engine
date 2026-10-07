"""Tests for combined confidence scoring and ranking."""
import pytest

from src.matching.scorer import compute_confidence, score_and_rank_candidates


class TestComputeConfidence:
    """Tests for the weighted confidence formula."""

    def test_weighted_formula(self):
        """alpha=0.3: 0.3*0.5 + 0.7*0.8 = 0.71"""
        result = compute_confidence(0.5, 0.8, alpha=0.3)
        assert result == pytest.approx(0.71)

    def test_default_alpha(self):
        """Default alpha=0.3: 0.3*0.4 + 0.7*0.9 = 0.75"""
        result = compute_confidence(0.4, 0.9)
        assert result == pytest.approx(0.3 * 0.4 + 0.7 * 0.9)

    def test_boundary_zero(self):
        assert compute_confidence(0.0, 0.0) == 0.0

    def test_boundary_one(self):
        assert compute_confidence(1.0, 1.0) == 1.0

    def test_range(self):
        """Confidence should always be in [0, 1] for valid inputs."""
        test_cases = [
            (0.0, 1.0),
            (1.0, 0.0),
            (0.5, 0.5),
            (0.1, 0.9),
            (0.99, 0.01),
        ]
        for kw, sem in test_cases:
            result = compute_confidence(kw, sem)
            assert 0.0 <= result <= 1.0, f"Out of range for kw={kw}, sem={sem}: {result}"


class TestScoreAndRankCandidates:
    """Tests for the full scoring and ranking pipeline."""

    def test_produces_sorted_output(self, semantic_matcher, sample_kalshi_markets, sample_poly_markets):
        """Output should be sorted by confidence_score descending."""
        results = score_and_rank_candidates(
            sample_kalshi_markets,
            sample_poly_markets,
            semantic_matcher,
        )
        scores = [r["confidence_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_output_has_required_fields(self, semantic_matcher, sample_kalshi_markets, sample_poly_markets):
        """Each result dict must have all required scoring fields."""
        results = score_and_rank_candidates(
            sample_kalshi_markets,
            sample_poly_markets,
            semantic_matcher,
        )
        required_fields = {
            "kalshi_market_id",
            "polymarket_market_id",
            "kalshi_question",
            "polymarket_question",
            "category",
            "keyword_score",
            "semantic_score",
            "confidence_score",
        }
        for r in results:
            missing = required_fields - set(r.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_confidence_uses_both_scores(self, semantic_matcher, sample_kalshi_markets, sample_poly_markets):
        """confidence_score should be a weighted combination of keyword and semantic scores."""
        results = score_and_rank_candidates(
            sample_kalshi_markets,
            sample_poly_markets,
            semantic_matcher,
            alpha=0.3,
        )
        for r in results:
            expected = 0.3 * r["keyword_score"] + 0.7 * r["semantic_score"]
            assert r["confidence_score"] == pytest.approx(expected, abs=1e-6)

    def test_empty_input(self, semantic_matcher):
        """Empty market lists should return empty results."""
        results = score_and_rank_candidates([], [], semantic_matcher)
        assert results == []

    def test_no_category_overlap(self, semantic_matcher):
        """Markets with no category overlap should return empty results."""
        kalshi = [{"market_id": "K1", "question": "BTC?", "category": "Crypto", "platform": "kalshi"}]
        poly = [{"market_id": "P1", "question": "CPI?", "category": "finance", "platform": "polymarket"}]
        results = score_and_rank_candidates(kalshi, poly, semantic_matcher)
        assert results == []
