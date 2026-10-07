"""Kalshi REST order client for live (real-money) trading.

Wraps Kalshi's authenticated /trade-api/v2 endpoints. Signs every
request with RSA-PSS SHA256 using a private key loaded from disk.
Gated behind LIVE_TRADING=true env var via risk_manager — calls
fail loudly if invoked outside an armed environment.

Authentication is per the Kalshi docs:
  - Headers: KALSHI-ACCESS-KEY, KALSHI-ACCESS-TIMESTAMP, KALSHI-ACCESS-SIGNATURE
  - Signature: base64(RSA-PSS-SHA256(f"{timestamp_ms}{METHOD}{path_no_query}"))
  - PSS padding: MGF1 with SHA-256, salt_length = DIGEST_LENGTH

Credentials (env):
  KALSHI_API_KEY_ID         — UUID-like string from the Kalshi portal
  KALSHI_PRIVATE_KEY_PATH   — filesystem path to your PEM-encoded RSA key
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import requests

from src.live.risk_manager import is_live_trading_armed

logger = logging.getLogger(__name__)

KALSHI_API_BASE_PROD = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_API_BASE_DEMO = "https://demo-api.kalshi.co/trade-api/v2"

# Allow env-var override so users can practice on demo before committing
# real USD. KALSHI_ENVIRONMENT=demo points at the sandbox; anything else
# (including unset) points at prod.
def _resolve_kalshi_base() -> str:
    env = os.environ.get("KALSHI_ENVIRONMENT", "").strip().lower()
    if env == "demo":
        return KALSHI_API_BASE_DEMO
    # Explicit URL override wins over both presets.
    custom = os.environ.get("KALSHI_API_BASE", "").strip()
    if custom:
        return custom
    return KALSHI_API_BASE_PROD


KALSHI_API_BASE = KALSHI_API_BASE_PROD  # legacy alias; prefer _resolve_kalshi_base()


def _load_private_key(pem_path: Path):
    """Lazy-load and cache the cryptography private key.

    cryptography is a heavy import; deferring it keeps the slim CI
    requirements from breaking when this module is merely imported
    (e.g. for type checks). The function raises a clear error if the
    library is missing AND live trading is armed.
    """
    try:
        from cryptography.hazmat.primitives import serialization
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'cryptography' package is required for Kalshi RSA-PSS signing. "
            "Install with: pip install cryptography>=42"
        ) from exc

    with open(pem_path, "rb") as f:
        key_data = f.read()
    return serialization.load_pem_private_key(key_data, password=None)


def _sign_request(private_key, timestamp_ms: int, method: str, path: str) -> str:
    """Return base64(RSA-PSS-SHA256(timestamp + method + path_no_query))."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    path_no_query = path.split("?", 1)[0]
    message = f"{timestamp_ms}{method.upper()}{path_no_query}".encode("utf-8")
    sig = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


@dataclass
class KalshiOrderClient:
    api_key_id: str = field(default_factory=lambda: os.environ.get("KALSHI_API_KEY_ID", ""))
    private_key_path: str = field(default_factory=lambda: os.environ.get("KALSHI_PRIVATE_KEY_PATH", ""))
    api_base: str = field(default_factory=_resolve_kalshi_base)
    request_timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        self._private_key = None  # lazy

    # ------------------------------------------------------------------
    # Internal request signing
    # ------------------------------------------------------------------

    def _ensure_creds(self) -> None:
        if not self.api_key_id:
            raise RuntimeError("KALSHI_API_KEY_ID env var not set")
        if not self.private_key_path:
            raise RuntimeError("KALSHI_PRIVATE_KEY_PATH env var not set")
        if not Path(self.private_key_path).exists():
            raise FileNotFoundError(f"Kalshi private key not found: {self.private_key_path}")

    def _get_private_key(self):
        if self._private_key is None:
            self._private_key = _load_private_key(Path(self.private_key_path))
        return self._private_key

    def _signed_headers(self, method: str, path: str) -> dict[str, str]:
        self._ensure_creds()
        ts_ms = int(time.time() * 1000)
        signature = _sign_request(self._get_private_key(), ts_ms, method, path)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
            "KALSHI-ACCESS-SIGNATURE": signature,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        if not is_live_trading_armed():
            # Hard-fail in dry-run rather than silently returning fake data.
            # The strategy layer is responsible for skipping the call entirely
            # when not armed; if we got here, something is wrong.
            raise RuntimeError(
                "KalshiOrderClient called while LIVE_TRADING is not armed. "
                "Strategy layer should gate the call upstream."
            )
        headers = self._signed_headers(method, path)
        url = f"{self.api_base.rstrip('/')}{path}"
        resp = requests.request(
            method.upper(),
            url,
            headers=headers,
            data=json.dumps(body) if body is not None else None,
            timeout=self.request_timeout_seconds,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Kalshi {method.upper()} {path} -> {resp.status_code}: {resp.text[:400]}"
            )
        if not resp.content:
            return {}
        return resp.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_balance(self) -> dict[str, Any]:
        """Read account balance (cents). Sanity check that auth works."""
        return self._request("GET", "/portfolio/balance")

    def place_limit_order(
        self,
        ticker: str,
        side: Literal["yes", "no"],
        count: int,
        limit_price_cents: int,
        action: Literal["buy", "sell"] = "buy",
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Place a limit order on Kalshi.

        Args:
            ticker: market ticker, e.g. "KXWTIW-26MAY-T70.00"
            side: "yes" or "no" — which side of the binary
            count: number of contracts (each at most $1 of risk)
            limit_price_cents: integer 1..99 — the bid you're willing to pay
            action: "buy" or "sell"
            client_order_id: idempotency key. Pass the same value to retry safely.

        Returns the raw Kalshi response (contains order_id, status, etc).
        """
        if not 1 <= limit_price_cents <= 99:
            raise ValueError(f"limit_price_cents must be in [1,99], got {limit_price_cents}")
        if count <= 0:
            raise ValueError(f"count must be > 0, got {count}")
        body: dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": int(count),
            "type": "limit",
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "time_in_force": "GTC",  # good-till-cancel
        }
        # Kalshi: limit price field name depends on the side you're buying.
        # For YES leg you set yes_price; for NO leg you set no_price.
        if side == "yes":
            body["yes_price"] = int(limit_price_cents)
        else:
            body["no_price"] = int(limit_price_cents)
        logger.info("Kalshi place_limit_order: %s %s %s@%d", action, ticker, side, limit_price_cents)
        return self._request("POST", "/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/portfolio/orders/{order_id}")

    def get_order(self, order_id: str) -> dict[str, Any]:
        return self._request("GET", f"/portfolio/orders/{order_id}")
