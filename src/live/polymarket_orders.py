"""Polymarket CLOB order client for live (real-money) trading.

Wraps the official py-clob-client SDK which handles all of:
  - EIP-712 typed-data signing of order payloads
  - L2 API credential derivation from a Polygon wallet private key
  - REST POST to the CLOB matching engine

Gated behind LIVE_TRADING=true via risk_manager. Calls fail loudly
if invoked outside an armed environment.

Credentials (env):
  POLYMARKET_PRIVATE_KEY   — 0x-prefixed Polygon wallet private key
  POLYMARKET_FUNDER_ADDRESS — proxy/funder wallet address (where USDC sits)
                              Often equals the wallet derived from the key,
                              but can differ when using Polymarket's
                              Safe-based proxy wallet pattern (signature_type=1).
  POLYMARKET_CHAIN_ID       — defaults to 137 (Polygon mainnet)
  POLYMARKET_HOST           — defaults to https://clob.polymarket.com

The py-clob-client library is heavy (depends on web3, eth-account,
eth-abi). It is lazy-imported so that the slim CI environment which
doesn't install it still passes when this module is merely imported.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from src.live.risk_manager import is_live_trading_armed

logger = logging.getLogger(__name__)

DEFAULT_HOST = "https://clob.polymarket.com"
DEFAULT_CHAIN_ID = 137  # Polygon mainnet
DEFAULT_SIGNATURE_TYPE = 1  # Polymarket proxy wallet


def _import_clob():
    """Lazy import. Raises a clear error if py-clob-client isn't installed."""
    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY, SELL
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "py-clob-client is required for Polymarket live trading. "
            "Install with: pip install py-clob-client>=0.18"
        ) from exc
    return {
        "ClobClient": ClobClient,
        "OrderArgs": OrderArgs,
        "OrderType": OrderType,
        "BUY": BUY,
        "SELL": SELL,
    }


@dataclass
class PolymarketOrderClient:
    private_key: str = field(default_factory=lambda: os.environ.get("POLYMARKET_PRIVATE_KEY", ""))
    funder_address: str = field(default_factory=lambda: os.environ.get("POLYMARKET_FUNDER_ADDRESS", ""))
    chain_id: int = field(default_factory=lambda: int(os.environ.get("POLYMARKET_CHAIN_ID", DEFAULT_CHAIN_ID)))
    host: str = field(default_factory=lambda: os.environ.get("POLYMARKET_HOST", DEFAULT_HOST))

    def __post_init__(self) -> None:
        self._client = None  # lazy: only construct when armed + needed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_creds(self) -> None:
        if not self.private_key:
            raise RuntimeError("POLYMARKET_PRIVATE_KEY env var not set")
        if not self.funder_address:
            raise RuntimeError("POLYMARKET_FUNDER_ADDRESS env var not set")

    def _get_client(self):
        if self._client is None:
            mods = _import_clob()
            self._ensure_creds()
            self._client = mods["ClobClient"](
                self.host,
                key=self.private_key,
                chain_id=self.chain_id,
                signature_type=DEFAULT_SIGNATURE_TYPE,
                funder=self.funder_address,
            )
            self._client.set_api_creds(self._client.create_or_derive_api_creds())
        return self._client

    def _require_armed(self) -> None:
        if not is_live_trading_armed():
            raise RuntimeError(
                "PolymarketOrderClient called while LIVE_TRADING is not armed. "
                "Strategy layer should gate the call upstream."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_balance_usdc(self) -> float:
        """Read USDC balance for the funder wallet (sanity check)."""
        self._require_armed()
        client = self._get_client()
        # py-clob-client exposes get_balance_allowance for the proxy wallet
        bal = client.get_balance_allowance()
        # bal: { "balance": "X.X", "allowance": "X.X" } (strings, USDC units)
        return float(bal.get("balance", 0.0))

    def place_limit_order(
        self,
        token_id: str,
        side: Literal["yes", "no"],
        size_usdc: float,
        limit_price: float,
    ) -> dict[str, Any]:
        """Place a limit order on Polymarket CLOB.

        Polymarket binary markets have TWO token_ids (one per outcome).
        Caller is responsible for passing the YES-token-id when buying
        YES, and the NO-token-id when buying NO.

        Args:
            token_id: ERC-1155 token id (string) for the outcome to buy.
            side: "yes" or "no" — informational; actual side selection
                  is done by which token_id you pass. We accept this
                  for symmetry with the Kalshi client; on Polymarket
                  buying YES vs buying NO is two different token_ids.
            size_usdc: dollar size of the bid, in USDC (e.g. 2.0).
            limit_price: probability between 0.01 and 0.99 inclusive.

        Returns the raw CLOB response (contains orderID, status, etc).
        """
        self._require_armed()

        if not 0.01 <= limit_price <= 0.99:
            raise ValueError(f"limit_price must be in [0.01, 0.99], got {limit_price}")
        if size_usdc <= 0:
            raise ValueError(f"size_usdc must be > 0, got {size_usdc}")

        mods = _import_clob()
        client = self._get_client()

        side_const = mods["BUY"]  # we always buy contracts in arb (long only)
        # Polymarket CLOB sizes orders in SHARES, not USD. Shares = usdc / price.
        size_shares = size_usdc / limit_price

        order_args = mods["OrderArgs"](
            token_id=token_id,
            price=float(limit_price),
            size=float(size_shares),
            side=side_const,
        )
        logger.info(
            "Polymarket place_limit_order: token=%s side=%s size_usdc=%.2f "
            "size_shares=%.2f price=%.3f",
            token_id, side, size_usdc, size_shares, limit_price,
        )
        signed = client.create_order(order_args)
        return client.post_order(signed, mods["OrderType"].GTC)

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        self._require_armed()
        return self._get_client().cancel(order_id)

    def get_order(self, order_id: str) -> dict[str, Any]:
        self._require_armed()
        return self._get_client().get_order(order_id)
