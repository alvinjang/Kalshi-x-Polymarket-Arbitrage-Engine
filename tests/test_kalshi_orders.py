"""Tests for the Kalshi order client.

These tests cover the parts that can be verified without real
credentials or network access: signing format, gating behavior,
input validation, and request structure (via a stub requests
session).
"""
from __future__ import annotations

import base64

import pytest

# Optional dep: skip the whole module if cryptography isn't installed.
crypto = pytest.importorskip("cryptography", reason="cryptography needed for Kalshi signing")
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


@pytest.fixture
def rsa_keypair(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path = tmp_path / "kalshi.pem"
    path.write_bytes(pem)
    return key, path


def test_sign_request_roundtrip(rsa_keypair):
    """Signature produced by our helper verifies under the matching public key."""
    from src.live.kalshi_orders import _sign_request

    key, _ = rsa_keypair
    ts = 1734_000_000_000
    method = "POST"
    path = "/trade-api/v2/portfolio/orders"

    sig_b64 = _sign_request(key, ts, method, path)
    raw_sig = base64.b64decode(sig_b64)
    message = f"{ts}{method}{path}".encode("utf-8")

    # Verify with the public key — round-trip works.
    key.public_key().verify(
        raw_sig,
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )


def test_sign_strips_query_params(rsa_keypair):
    from src.live.kalshi_orders import _sign_request

    key, _ = rsa_keypair
    ts = 1734_000_000_000
    s1 = _sign_request(key, ts, "GET", "/portfolio/orders?limit=5")
    s2 = _sign_request(key, ts, "GET", "/portfolio/orders")
    # Both should derive from the same canonical message → signatures
    # are non-deterministic under PSS (salt) but both must verify the
    # *same* message. We check by verifying both against /portfolio/orders.
    message = f"{ts}GET/portfolio/orders".encode("utf-8")
    pub = key.public_key()
    for sig_b64 in (s1, s2):
        pub.verify(
            base64.b64decode(sig_b64),
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )


def test_client_blocks_when_not_armed(rsa_keypair, monkeypatch):
    """Sanity check: an unarmed client should refuse to call _request."""
    monkeypatch.delenv("LIVE_TRADING", raising=False)
    monkeypatch.delenv("EMERGENCY_HALT", raising=False)
    monkeypatch.setenv("KALSHI_API_KEY_ID", "stub")
    _, key_path = rsa_keypair
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(key_path))

    from src.live.kalshi_orders import KalshiOrderClient

    client = KalshiOrderClient()
    with pytest.raises(RuntimeError, match="LIVE_TRADING"):
        client.get_balance()


def test_place_limit_order_validates_price(rsa_keypair, monkeypatch):
    """Price validation runs before the live-arm check (input first)."""
    monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.setenv("KALSHI_API_KEY_ID", "stub")
    _, key_path = rsa_keypair
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(key_path))

    from src.live.kalshi_orders import KalshiOrderClient

    client = KalshiOrderClient()
    with pytest.raises(ValueError, match="limit_price_cents"):
        client.place_limit_order("KXWTI-MAR", "yes", count=1, limit_price_cents=0)
    with pytest.raises(ValueError, match="limit_price_cents"):
        client.place_limit_order("KXWTI-MAR", "yes", count=1, limit_price_cents=100)
    with pytest.raises(ValueError, match="count"):
        client.place_limit_order("KXWTI-MAR", "yes", count=0, limit_price_cents=50)


def test_place_limit_order_request_body(rsa_keypair, monkeypatch):
    """Verify the request body shape the client builds is what Kalshi expects."""
    monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.setenv("KALSHI_API_KEY_ID", "stub")
    _, key_path = rsa_keypair
    monkeypatch.setenv("KALSHI_PRIVATE_KEY_PATH", str(key_path))

    sent: dict = {}

    class FakeResp:
        status_code = 200
        content = b'{"order": {"order_id": "abc"}}'
        text = ""
        def json(self):
            return {"order": {"order_id": "abc"}}

    def fake_request(method, url, headers, data, timeout):
        sent["method"] = method
        sent["url"] = url
        sent["headers"] = headers
        sent["data"] = data
        return FakeResp()

    monkeypatch.setattr("src.live.kalshi_orders.requests.request", fake_request)

    from src.live.kalshi_orders import KalshiOrderClient
    client = KalshiOrderClient()
    resp = client.place_limit_order(
        "KXWTIW-26MAY-T70.00", side="yes", count=1,
        limit_price_cents=42, action="buy",
        client_order_id="test-cli-1",
    )

    assert resp == {"order": {"order_id": "abc"}}
    assert sent["method"] == "POST"
    assert sent["url"].endswith("/portfolio/orders")

    import json
    body = json.loads(sent["data"])
    assert body == {
        "ticker": "KXWTIW-26MAY-T70.00",
        "side": "yes",
        "action": "buy",
        "count": 1,
        "type": "limit",
        "client_order_id": "test-cli-1",
        "time_in_force": "GTC",
        "yes_price": 42,
    }
    assert "KALSHI-ACCESS-KEY" in sent["headers"]
    assert "KALSHI-ACCESS-SIGNATURE" in sent["headers"]
    assert "KALSHI-ACCESS-TIMESTAMP" in sent["headers"]
