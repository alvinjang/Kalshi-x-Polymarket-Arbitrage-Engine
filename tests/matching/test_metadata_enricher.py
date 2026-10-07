"""Tests for settlement criteria enrichment from platform APIs."""
import copy
from unittest.mock import patch, MagicMock

import pytest
import requests

from src.matching.metadata_enricher import enrich_settlement_criteria


class TestEnrichSettlementCriteria:
    """Tests for enrich_settlement_criteria function."""

    def test_enriches_kalshi_fields(
        self, sample_scored_candidates, mock_kalshi_settlement_response, mock_poly_settlement_response
    ):
        """Kalshi settlement should contain rules_primary text."""
        candidates = [copy.deepcopy(sample_scored_candidates[0])]

        mock_kalshi_resp = MagicMock()
        mock_kalshi_resp.json.return_value = mock_kalshi_settlement_response
        mock_kalshi_resp.raise_for_status.return_value = None

        mock_poly_resp = MagicMock()
        mock_poly_resp.json.return_value = mock_poly_settlement_response
        mock_poly_resp.raise_for_status.return_value = None

        with patch("src.matching.metadata_enricher.ResilientClient") as MockClient:
            kalshi_instance = MagicMock()
            gamma_instance = MagicMock()
            kalshi_instance.session.get.return_value = mock_kalshi_resp
            kalshi_instance.base_url = "https://api.elections.kalshi.com/trade-api/v2"
            kalshi_instance.timeout = 30
            gamma_instance.session.get.return_value = mock_poly_resp
            gamma_instance.base_url = "https://gamma-api.polymarket.com"
            gamma_instance.timeout = 30
            MockClient.side_effect = [kalshi_instance, gamma_instance]

            result = enrich_settlement_criteria(candidates)

        assert "CoinDesk BTC Index" in result[0]["kalshi_settlement"]

    def test_enriches_polymarket_fields(
        self, sample_scored_candidates, mock_kalshi_settlement_response, mock_poly_settlement_response
    ):
        """Polymarket settlement should contain description + resolutionSource."""
        candidates = [copy.deepcopy(sample_scored_candidates[0])]

        mock_kalshi_resp = MagicMock()
        mock_kalshi_resp.json.return_value = mock_kalshi_settlement_response
        mock_kalshi_resp.raise_for_status.return_value = None

        mock_poly_resp = MagicMock()
        mock_poly_resp.json.return_value = mock_poly_settlement_response
        mock_poly_resp.raise_for_status.return_value = None

        with patch("src.matching.metadata_enricher.ResilientClient") as MockClient:
            kalshi_instance = MagicMock()
            gamma_instance = MagicMock()
            kalshi_instance.session.get.return_value = mock_kalshi_resp
            kalshi_instance.base_url = "https://api.elections.kalshi.com/trade-api/v2"
            kalshi_instance.timeout = 30
            gamma_instance.session.get.return_value = mock_poly_resp
            gamma_instance.base_url = "https://gamma-api.polymarket.com"
            gamma_instance.timeout = 30
            MockClient.side_effect = [kalshi_instance, gamma_instance]

            result = enrich_settlement_criteria(candidates)

        assert "BTC/USD >= $80,000" in result[0]["polymarket_settlement"]
        assert "coinbase.com" in result[0]["polymarket_settlement"]

    def test_settlement_aligned_true(
        self, sample_scored_candidates, mock_kalshi_settlement_response, mock_poly_settlement_response
    ):
        """When both APIs return data, settlement_aligned should be True (default for human review)."""
        candidates = [copy.deepcopy(sample_scored_candidates[0])]

        mock_kalshi_resp = MagicMock()
        mock_kalshi_resp.json.return_value = mock_kalshi_settlement_response
        mock_kalshi_resp.raise_for_status.return_value = None

        mock_poly_resp = MagicMock()
        mock_poly_resp.json.return_value = mock_poly_settlement_response
        mock_poly_resp.raise_for_status.return_value = None

        with patch("src.matching.metadata_enricher.ResilientClient") as MockClient:
            kalshi_instance = MagicMock()
            gamma_instance = MagicMock()
            kalshi_instance.session.get.return_value = mock_kalshi_resp
            kalshi_instance.base_url = "https://api.elections.kalshi.com/trade-api/v2"
            kalshi_instance.timeout = 30
            gamma_instance.session.get.return_value = mock_poly_resp
            gamma_instance.base_url = "https://gamma-api.polymarket.com"
            gamma_instance.timeout = 30
            MockClient.side_effect = [kalshi_instance, gamma_instance]

            result = enrich_settlement_criteria(candidates)

        assert result[0]["settlement_aligned"] is True

    def test_api_error_handling(self, sample_scored_candidates):
        """When API raises exception, fields should be FETCH_FAILED and settlement_aligned False."""
        candidates = [copy.deepcopy(sample_scored_candidates[0])]

        with patch("src.matching.metadata_enricher.ResilientClient") as MockClient:
            kalshi_instance = MagicMock()
            gamma_instance = MagicMock()
            kalshi_instance.session.get.side_effect = requests.exceptions.RequestException("timeout")
            kalshi_instance.base_url = "https://api.elections.kalshi.com/trade-api/v2"
            kalshi_instance.timeout = 30
            gamma_instance.session.get.side_effect = requests.exceptions.RequestException("timeout")
            gamma_instance.base_url = "https://gamma-api.polymarket.com"
            gamma_instance.timeout = 30
            MockClient.side_effect = [kalshi_instance, gamma_instance]

            result = enrich_settlement_criteria(candidates)

        assert result[0]["kalshi_settlement"] == "FETCH_FAILED"
        assert result[0]["polymarket_settlement"] == "FETCH_FAILED"
        assert result[0]["settlement_aligned"] is False

    def test_empty_rules(self, sample_scored_candidates):
        """When API returns empty strings for rules, fields should be set but empty."""
        candidates = [copy.deepcopy(sample_scored_candidates[0])]

        empty_kalshi = {"market": {"rules_primary": "", "rules_secondary": ""}}
        empty_poly = [{"description": "", "resolutionSource": ""}]

        mock_kalshi_resp = MagicMock()
        mock_kalshi_resp.json.return_value = empty_kalshi
        mock_kalshi_resp.raise_for_status.return_value = None

        mock_poly_resp = MagicMock()
        mock_poly_resp.json.return_value = empty_poly
        mock_poly_resp.raise_for_status.return_value = None

        with patch("src.matching.metadata_enricher.ResilientClient") as MockClient:
            kalshi_instance = MagicMock()
            gamma_instance = MagicMock()
            kalshi_instance.session.get.return_value = mock_kalshi_resp
            kalshi_instance.base_url = "https://api.elections.kalshi.com/trade-api/v2"
            kalshi_instance.timeout = 30
            gamma_instance.session.get.return_value = mock_poly_resp
            gamma_instance.base_url = "https://gamma-api.polymarket.com"
            gamma_instance.timeout = 30
            MockClient.side_effect = [kalshi_instance, gamma_instance]

            result = enrich_settlement_criteria(candidates)

        assert result[0]["kalshi_settlement"] == ""
        assert result[0]["polymarket_settlement"] == ""
        assert result[0]["settlement_aligned"] is False
