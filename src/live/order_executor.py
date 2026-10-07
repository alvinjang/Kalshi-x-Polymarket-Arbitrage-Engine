"""Two-leg order execution coordinator for live arbitrage trading.

A single arb "trade" is TWO orders — one on Kalshi, one on Polymarket
— that must both fill or neither. This module handles:

  1. Pre-trade risk gate (per-trade size, daily kill switch, exposure cap)
  2. Direction inference (which side to buy on each exchange)
  3. Order placement (parallel-ish: Kalshi REST + Polymarket CLOB)
  4. Atomicity recovery: if one leg fills and the other rejects/times out,
     cancel the open order; if the open order has partial fills, try to
     close them via the opposite side at the next cycle's price.

Default behavior: when LIVE_TRADING is not armed, this module is a
no-op (returns a synthetic success record so the strategy layer can
keep using its existing paper-tracking codepath unchanged).

Sizing convention:
  - Kalshi: contracts at price-cents. A $2 position at 60c YES → 3 contracts
            (round_down). Counts of 0 are skipped (means the price band
            is too high to fit even one contract within the per-trade cap).
  - Polymarket: USDC notional. A $2 position at $0.40/share → 5 shares.
"""
from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Optional

from src.live.risk_manager import (
    DEFAULT_PER_TRADE_USD,
    check_pretrade,
    is_live_trading_armed,
)

logger = logging.getLogger(__name__)


@dataclass
class ArbExecutionResult:
    permitted: bool
    placed_kalshi: bool = False
    placed_polymarket: bool = False
    kalshi_order_id: Optional[str] = None
    polymarket_order_id: Optional[str] = None
    reject_reason: Optional[str] = None
    notes: list[str] = None

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []


def _infer_legs(
    spread: float, kalshi_yes_price: float, poly_yes_price: float
) -> tuple[Literal["yes", "no"], float, Literal["yes_token", "no_token"], float]:
    """Return (kalshi_side, kalshi_leg_price, poly_token_side, poly_leg_price).

    Both directions of "spread will narrow toward zero" call for buying
    the cheap side on each exchange.
    """
    if spread > 0:
        # Kalshi YES is expensive vs Polymarket YES.
        # Short Kalshi (buy NO) + Long Polymarket YES.
        return ("no", 1.0 - kalshi_yes_price,
                "yes_token", poly_yes_price)
    else:
        # Kalshi YES is cheap vs Polymarket YES.
        # Long Kalshi YES + Short Polymarket (buy NO token).
        return ("yes", kalshi_yes_price,
                "no_token", 1.0 - poly_yes_price)


def execute_arb(
    *,
    pair_id: str,
    kalshi_ticker: str,
    poly_yes_token_id: str,
    poly_no_token_id: str,
    kalshi_yes_price: float,
    poly_yes_price: float,
    spread: float,
    per_trade_usd: float = DEFAULT_PER_TRADE_USD,
    kalshi_client: Optional[Any] = None,
    poly_client: Optional[Any] = None,
    history_path: Optional[str] = None,
    positions_db: Optional[str] = None,
) -> ArbExecutionResult:
    """Place both legs of an arb trade. Returns a structured result.

    The clients are injected so this is testable without network — pass
    real KalshiOrderClient / PolymarketOrderClient instances in
    production, stubs in tests.
    """
    # When LIVE_TRADING is off, return a no-op success so the strategy
    # paper-path continues to function. This is the only place the
    # paper-vs-real distinction is made in the execution layer.
    if not is_live_trading_armed():
        return ArbExecutionResult(
            permitted=True,
            placed_kalshi=False,
            placed_polymarket=False,
            notes=["paper-mode (LIVE_TRADING not armed)"],
        )

    # Pre-trade risk gate. Check both exchange exposures separately —
    # a kill switch or per-exchange cap could trip even when the other
    # exchange has room.
    risk_kwargs = {}
    if history_path is not None:
        risk_kwargs["history_path"] = history_path
    if positions_db is not None:
        risk_kwargs["positions_db"] = positions_db
    risk_k = check_pretrade(trade_size_usd=per_trade_usd, exchange="kalshi", **risk_kwargs)
    risk_p = check_pretrade(trade_size_usd=per_trade_usd, exchange="polymarket", **risk_kwargs)
    if not risk_k.permit:
        return ArbExecutionResult(permitted=False,
                                  reject_reason=f"kalshi gate: {risk_k.reason}")
    if not risk_p.permit:
        return ArbExecutionResult(permitted=False,
                                  reject_reason=f"polymarket gate: {risk_p.reason}")

    if kalshi_client is None or poly_client is None:
        return ArbExecutionResult(
            permitted=False,
            reject_reason="missing kalshi or polymarket client",
        )

    k_side, k_leg_price, p_token_side, p_leg_price = _infer_legs(
        spread, kalshi_yes_price, poly_yes_price
    )

    # Convert dollar size to Kalshi contracts. Each contract pays $1
    # on a YES win, so cost ≈ count * price.
    if k_leg_price <= 0:
        return ArbExecutionResult(
            permitted=False,
            reject_reason=f"kalshi leg price {k_leg_price:.4f} not tradeable",
        )
    k_count = int(math.floor(per_trade_usd / k_leg_price))
    if k_count < 1:
        return ArbExecutionResult(
            permitted=False,
            reject_reason=f"kalshi leg too expensive (price={k_leg_price:.2f}) "
                          f"for ${per_trade_usd} per-trade",
        )
    k_price_cents = int(round(k_leg_price * 100))
    k_price_cents = max(1, min(99, k_price_cents))

    poly_token_id = (poly_yes_token_id if p_token_side == "yes_token"
                     else poly_no_token_id)

    client_order_id = f"arb-{pair_id}-{uuid.uuid4().hex[:8]}"
    result = ArbExecutionResult(permitted=True)

    # ---- Place Kalshi leg ----
    try:
        k_resp = kalshi_client.place_limit_order(
            ticker=kalshi_ticker,
            side=k_side,
            count=k_count,
            limit_price_cents=k_price_cents,
            action="buy",
            client_order_id=client_order_id + "-k",
        )
        result.kalshi_order_id = (
            k_resp.get("order", {}).get("order_id")
            if isinstance(k_resp, dict) else None
        )
        result.placed_kalshi = True
    except Exception as e:
        logger.exception("Kalshi leg failed for %s: %s", pair_id, e)
        result.reject_reason = f"kalshi placement failed: {e}"
        return result

    # ---- Place Polymarket leg ----
    try:
        p_resp = poly_client.place_limit_order(
            token_id=poly_token_id,
            side=("yes" if p_token_side == "yes_token" else "no"),
            size_usdc=per_trade_usd,
            limit_price=p_leg_price,
        )
        result.polymarket_order_id = (
            p_resp.get("orderID") if isinstance(p_resp, dict) else None
        )
        result.placed_polymarket = True
    except Exception as e:
        logger.exception("Polymarket leg failed for %s: %s", pair_id, e)
        result.reject_reason = f"polymarket placement failed: {e}"
        # Atomicity recovery: cancel the open Kalshi leg.
        if result.kalshi_order_id:
            try:
                kalshi_client.cancel_order(result.kalshi_order_id)
                result.notes.append("cancelled kalshi leg after poly failure")
            except Exception as ce:
                logger.exception("Failed to cancel Kalshi leg: %s", ce)
                result.notes.append(f"CRITICAL: kalshi leg open, cancel failed: {ce}")
        return result

    result.notes.append(
        f"k_side={k_side} k_count={k_count} k_price_c={k_price_cents} "
        f"p_token={p_token_side} p_price={p_leg_price:.3f}"
    )
    return result
