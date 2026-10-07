"""Resilient HTTP client with retry, rate limiting, and logging."""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import time
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class ResilientClient:
    """HTTP client with automatic retry on transient failures and rate limiting.

    Retries on HTTP status codes: 429, 500, 502, 503, 504.
    Uses urllib3's built-in exponential backoff with configurable backoff_factor.
    Enforces minimum interval between requests for rate limiting.
    """

    def __init__(
        self,
        base_url: str,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        requests_per_second: float = 10.0,
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.min_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0
        self.timeout = timeout

        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get(self, path: str, params: dict | None = None) -> dict | list:
        """Rate-limited GET request with automatic retry.

        Returns parsed JSON response (dict or list).
        Raises requests.exceptions.HTTPError on non-retryable failures.
        """
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)

        url = f"{self.base_url}/{path.lstrip('/')}"
        logger.debug(f"GET {url} params={params}")
        self._last_request_time = time.time()

        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
