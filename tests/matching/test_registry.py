"""Tests for matched pairs registry I/O and deduplication."""
import json

import pytest

from src.matching.registry import save_registry, load_registry, deduplicate_pairs


class TestSaveAndLoadRegistry:
    """Tests for save_registry and load_registry."""

    def test_save_creates_json(self, tmp_path, sample_scored_candidates):
        """save_registry should create a valid JSON file."""
        output = tmp_path / "test.json"
        save_registry(sample_scored_candidates, output)
        assert output.exists()
        data = json.loads(output.read_text())
        assert isinstance(data, list)
        assert len(data) == 3

    def test_load_returns_data(self, tmp_path, sample_scored_candidates):
        """Round-trip save then load should preserve data."""
        output = tmp_path / "test.json"
        save_registry(sample_scored_candidates, output)
        loaded = load_registry(output)
        assert len(loaded) == len(sample_scored_candidates)
        assert loaded[0]["kalshi_market_id"] == sample_scored_candidates[0]["kalshi_market_id"]

    def test_load_missing_file_returns_empty(self, tmp_path):
        """load_registry on nonexistent path should return empty list."""
        result = load_registry(tmp_path / "nonexistent.json")
        assert result == []

    def test_save_creates_parent_dirs(self, tmp_path, sample_scored_candidates):
        """save_registry should create parent directories if missing."""
        output = tmp_path / "nested" / "dir" / "test.json"
        save_registry(sample_scored_candidates, output)
        assert output.exists()


class TestDeduplicatePairs:
    """Tests for deduplicate_pairs."""

    def test_keeps_highest_confidence(self):
        """When same kalshi_market_id appears twice, keep highest confidence."""
        pairs = [
            {"kalshi_market_id": "K1", "polymarket_market_id": "P1", "confidence_score": 0.5},
            {"kalshi_market_id": "K1", "polymarket_market_id": "P2", "confidence_score": 0.8},
            {"kalshi_market_id": "K2", "polymarket_market_id": "P3", "confidence_score": 0.6},
        ]
        result = deduplicate_pairs(pairs)
        assert len(result) == 2
        k1_pair = [p for p in result if p["kalshi_market_id"] == "K1"][0]
        assert k1_pair["confidence_score"] == 0.8
        assert k1_pair["polymarket_market_id"] == "P2"

    def test_no_duplicates_unchanged(self, sample_scored_candidates):
        """Input with no duplicates should return same list."""
        result = deduplicate_pairs(sample_scored_candidates)
        assert len(result) == len(sample_scored_candidates)

    def test_dedup_by_polymarket_id(self):
        """When same polymarket_market_id appears twice, keep highest confidence."""
        pairs = [
            {"kalshi_market_id": "K1", "polymarket_market_id": "P1", "confidence_score": 0.9},
            {"kalshi_market_id": "K2", "polymarket_market_id": "P1", "confidence_score": 0.4},
        ]
        result = deduplicate_pairs(pairs)
        assert len(result) == 1
        assert result[0]["kalshi_market_id"] == "K1"
