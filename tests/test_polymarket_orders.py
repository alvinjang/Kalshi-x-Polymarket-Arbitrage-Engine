"""Tests for the Polymarket order client.

py-clob-client is a heavy dep we don't install in CI. These tests
exercise the gating + input-validation paths that work without the
SDK present, plus a stub-based test of the order body construction.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING", raising=False)
    monkeypatch.delenv("EMERGENCY_HALT", raising=False)
    monkeypatch.delenv("POLYMARKET_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("POLYMARKET_FUNDER_ADDRESS", raising=False)


def test_polymarket_client_blocks_when_not_armed(monkeypatch):
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0xabc")
    monkeypatch.setenv("POLYMARKET_FUNDER_ADDRESS", "0xdef")
    from src.live.polymarket_orders import PolymarketOrderClient
    client = PolymarketOrderClient()
    with pytest.raises(RuntimeError, match="LIVE_TRADING"):
        client.place_limit_order("0xtoken", "yes", size_usdc=2.0, limit_price=0.5)


def test_polymarket_validates_price_range(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0xabc")
    monkeypatch.setenv("POLYMARKET_FUNDER_ADDRESS", "0xdef")
    from src.live.polymarket_orders import PolymarketOrderClient
    client = PolymarketOrderClient()
    with pytest.raises(ValueError, match="limit_price"):
        client.place_limit_order("0xtok", "yes", size_usdc=2.0, limit_price=0.0)
    with pytest.raises(ValueError, match="limit_price"):
        client.place_limit_order("0xtok", "yes", size_usdc=2.0, limit_price=1.0)
    with pytest.raises(ValueError, match="size_usdc"):
        client.place_limit_order("0xtok", "yes", size_usdc=0.0, limit_price=0.5)


def test_polymarket_size_shares_calculation(monkeypatch):
    """The CLOB sizes orders in SHARES = usdc / price — verify the conversion."""
    monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0xabc")
    monkeypatch.setenv("POLYMARKET_FUNDER_ADDRESS", "0xdef")

    # Stub out the entire py_clob_client import surface.
    class StubOrderArgs:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class StubOrderType:
        GTC = "GTC"

    class StubClient:
        def __init__(self, *a, **kw):
            self.captured_order = None

        def set_api_creds(self, _creds):
            pass

        def create_or_derive_api_creds(self):
            return object()

        def create_order(self, order_args):
            self.captured_order = order_args
            return {"signed": True}

        def post_order(self, signed, order_type):
            return {
                "orderID": "fake-order-id",
                "captured": self.captured_order.__dict__,
                "order_type": order_type,
            }

    stub_mods = {
        "ClobClient": StubClient,
        "OrderArgs": StubOrderArgs,
        "OrderType": StubOrderType,
        "BUY": "BUY",
        "SELL": "SELL",
    }
    monkeypatch.setattr("src.live.polymarket_orders._import_clob",
                        lambda: stub_mods)

    from src.live.polymarket_orders import PolymarketOrderClient
    client = PolymarketOrderClient()

    resp = client.place_limit_order(
        token_id="0xToken123",
        side="yes",
        size_usdc=2.0,
        limit_price=0.4,
    )

    # $2 USDC at $0.40/share = 5 shares
    assert resp["captured"]["price"] == pytest.approx(0.4)
    assert resp["captured"]["size"] == pytest.approx(5.0)
    assert resp["captured"]["token_id"] == "0xToken123"
    assert resp["captured"]["side"] == "BUY"
    assert resp["order_type"] == "GTC"
