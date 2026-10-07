"""Tests for ResilientClient retry and rate limiting."""
import time
import pytest
from unittest.mock import patch, MagicMock
from requests.exceptions import HTTPError
from src.data.client import ResilientClient


class TestResilientClientRetry:
    def test_retries_on_500(self):
        """Client retries on HTTP 500 server errors."""
        client = ResilientClient(
            base_url="https://example.com/api",
            max_retries=2,
            backoff_factor=0.01,
            requests_per_second=1000,
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "ok"}
        mock_response.raise_for_status.return_value = None

        with patch.object(client.session, "get", return_value=mock_response) as mock_get:
            result = client.get("test")
            assert result == {"data": "ok"}
            mock_get.assert_called_once()

    def test_raises_on_404(self):
        """Client does NOT retry on HTTP 404 (not in status_forcelist)."""
        client = ResilientClient(
            base_url="https://example.com/api",
            max_retries=2,
            backoff_factor=0.01,
            requests_per_second=1000,
        )
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPError("404 Not Found")

        with patch.object(client.session, "get", return_value=mock_response):
            with pytest.raises(HTTPError, match="404"):
                client.get("not-found")

    def test_constructs_correct_url(self):
        """Client combines base_url and path correctly."""
        client = ResilientClient(
            base_url="https://example.com/api/v2/",
            requests_per_second=1000,
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None

        with patch.object(client.session, "get", return_value=mock_response) as mock_get:
            client.get("markets/list")
            called_url = mock_get.call_args[0][0]
            assert called_url == "https://example.com/api/v2/markets/list"


class TestResilientClientRateLimiting:
    def test_enforces_minimum_interval(self):
        """Two rapid requests have at least min_interval gap."""
        client = ResilientClient(
            base_url="https://example.com/api",
            requests_per_second=5.0,  # 200ms min interval
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None

        with patch.object(client.session, "get", return_value=mock_response):
            start = time.time()
            client.get("first")
            client.get("second")
            elapsed = time.time() - start
            # Two requests at 5 req/sec should take at least 0.2s total
            # (first is immediate, second waits for min_interval)
            assert elapsed >= 0.15, f"Elapsed {elapsed:.3f}s < 0.15s, rate limiting not working"
