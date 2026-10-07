#!/usr/bin/env python3
"""Place exactly ONE real order on Kalshi demo + Polymarket to validate
the full live-order code path end-to-end.

Unlike `scripts/live_check.py` (which only pings balance endpoints),
this script exercises the entire pipeline:
    features → models → strategy filter → execute_arb → both exchanges

When run with KALSHI_ENVIRONMENT=demo, this places a REAL ORDER against
Kalshi's demo environment (paper money, real markets). For Polymarket,
no demo CLOB exists, so the script will only place the Kalshi leg
unless --include-polymarket is set (which DOES use real USDC).

Safety guards:
  - Refuses to run unless LIVE_TRADING=true is in env
  - Picks at most ONE pair, places at most ONE order per exchange
  - Confirms with user before submitting (--yes to skip)
  - Defaults to Kalshi-only; Polymarket leg requires --include-polymarket

Usage:
    source ~/.env.live
    python scripts/dry_test_arb.py             # Kalshi demo only, prompts
    python scripts/dry_test_arb.py --yes       # skip confirmation
    python scripts/dry_test_arb.py --include-polymarket   # both legs (REAL USDC)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make project importable
sys.path.insert(0, str(Path(__file__).parent.parent))

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def pick_qualifying_pair() -> dict | None:
    """Find one commodity pair with all the data we need.

    Returns a dict with kalshi_ticker, poly_yes_token_id, poly_no_token_id,
    plus the latest prices/spread.
    """
    from src.live.collector import LiveCollector
    from src.live.contract_classifier import ContractClassifier
    from src.features.category import derive_category_from_ticker

    print(f"\n{BOLD}Looking for a qualifying commodity pair...{RESET}")

    mapping_path = Path("data/live/pair_mapping.json")
    if not mapping_path.exists():
        fail("data/live/pair_mapping.json missing")
        return None
    with open(mapping_path) as f:
        mapping = json.load(f)

    # Filter to pairs with token IDs (live-tradeable on Polymarket)
    enriched = {pid: e for pid, e in mapping.items()
                if e.get("polymarket_yes_token_id") and e.get("polymarket_no_token_id")}
    ok(f"{len(enriched)} pairs have YES/NO tokens")

    # Filter to commodity edge
    commodity_pids = [
        pid for pid, e in enriched.items()
        if derive_category_from_ticker(e.get("kalshi_market_id", "")) in ("oil", "commodities")
    ]
    ok(f"{len(commodity_pids)} of those are oil/commodity category")
    if not commodity_pids:
        warn("No enriched commodity pairs available. Cannot run dry test.")
        return None

    # Fetch current prices for these candidates
    print("  Fetching live prices (this takes ~10-30 seconds)...")
    collector = LiveCollector(use_live_pairs=True)
    k_prices = collector.fetch_kalshi_prices()
    p_prices = collector.fetch_polymarket_prices()
    ok(f"Fetched {len(k_prices)} Kalshi + {len(p_prices)} Polymarket prices")

    # Look for a pair where both prices are known AND spread is interesting
    best = None
    best_spread = 0.0
    for pid in commodity_pids:
        entry = enriched[pid]
        k_ticker = entry["kalshi_market_id"]
        cid = entry["polymarket_market_id"]
        k_px = k_prices.get(k_ticker)
        p_px = p_prices.get(cid)
        if k_px is None or p_px is None:
            continue
        # Sanity bounds
        if not (0.05 < k_px < 0.95 and 0.05 < p_px < 0.95):
            continue
        spread = k_px - p_px
        if abs(spread) < 0.05:
            continue  # Too small to be worth trading
        if abs(spread) > best_spread:
            best_spread = abs(spread)
            best = {
                "pair_id": pid,
                "kalshi_ticker": k_ticker,
                "polymarket_yes_token_id": entry["polymarket_yes_token_id"],
                "polymarket_no_token_id": entry["polymarket_no_token_id"],
                "kalshi_price": k_px,
                "poly_price": p_px,
                "spread": spread,
            }
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation prompt before placing")
    parser.add_argument("--include-polymarket", action="store_true",
                        help="Also place the Polymarket leg (uses REAL USDC)")
    args = parser.parse_args()

    print(f"{BOLD}=== Single-order dry test ==={RESET}\n")

    if os.environ.get("LIVE_TRADING", "").lower() != "true":
        fail("LIVE_TRADING is not 'true'. The dry test requires arming live")
        print("    trading to validate the order path. To run safely:")
        print(f"    {BOLD}export KALSHI_ENVIRONMENT=demo{RESET}")
        print(f"    {BOLD}export LIVE_TRADING=true{RESET}")
        print(f"    python scripts/dry_test_arb.py")
        return 2

    env = os.environ.get("KALSHI_ENVIRONMENT", "").lower()
    if env != "demo":
        if not args.include_polymarket:
            warn("KALSHI_ENVIRONMENT is not 'demo'. The Kalshi leg will hit PROD.")
        print(f"  Current Kalshi environment: {env or '(unset = prod)'}")
        if not args.yes:
            ans = input(f"  {BOLD}Continue against {env or 'prod'}? [y/N] {RESET}").strip().lower()
            if ans not in ("y", "yes"):
                print("Aborted.")
                return 1

    pair = pick_qualifying_pair()
    if pair is None:
        fail("No qualifying pair found right now. Try again later or relax filters.")
        return 1

    print(f"\n{BOLD}Selected pair:{RESET}")
    print(f"  pair_id:           {pair['pair_id']}")
    print(f"  kalshi_ticker:     {pair['kalshi_ticker']}")
    print(f"  kalshi YES price:  ${pair['kalshi_price']:.3f}")
    print(f"  polymarket YES px: ${pair['poly_price']:.3f}")
    print(f"  signed spread:     {pair['spread']:+.3f}")

    direction = "short_spread" if pair["spread"] > 0 else "long_spread"
    print(f"  inferred dir:      {direction}")

    # Compute legs the same way order_executor would
    from src.live.order_executor import _infer_legs
    k_side, k_leg_price, p_token_side, p_leg_price = _infer_legs(
        pair["spread"], pair["kalshi_price"], pair["poly_price"]
    )
    print(f"\n{BOLD}Planned legs (${{2}}/leg canary size):{RESET}")
    print(f"  Kalshi:      BUY {k_side.upper()} @ ${k_leg_price:.3f} ({int(k_leg_price*100)}¢)")
    print(f"  Polymarket:  BUY {p_token_side} @ ${p_leg_price:.3f}")

    if not args.yes:
        print()
        ans = input(f"  {BOLD}Place these orders? [y/N] {RESET}").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted (no orders placed).")
            return 0

    # Construct the clients lazily and call execute_arb
    print(f"\n{BOLD}Placing orders...{RESET}")
    from src.live.order_executor import execute_arb
    from src.live.kalshi_orders import KalshiOrderClient

    kalshi_client = KalshiOrderClient()

    if args.include_polymarket:
        from src.live.polymarket_orders import PolymarketOrderClient
        poly_client = PolymarketOrderClient()
    else:
        # Pass a stub that refuses to place — we only want the Kalshi leg.
        class KalshiOnlyStub:
            def place_limit_order(self, **kw):
                raise RuntimeError(
                    "POLY-LEG-SKIPPED: --include-polymarket flag not set. "
                    "Set the flag to also place the Polymarket leg (uses real USDC)."
                )
            def cancel_order(self, order_id):
                return {"cancelled": order_id}
        poly_client = KalshiOnlyStub()

    result = execute_arb(
        pair_id=pair["pair_id"],
        kalshi_ticker=pair["kalshi_ticker"],
        poly_yes_token_id=pair["polymarket_yes_token_id"],
        poly_no_token_id=pair["polymarket_no_token_id"],
        kalshi_yes_price=pair["kalshi_price"],
        poly_yes_price=pair["poly_price"],
        spread=pair["spread"],
        kalshi_client=kalshi_client,
        poly_client=poly_client,
    )

    print()
    if result.placed_kalshi:
        ok(f"Kalshi order placed: order_id={result.kalshi_order_id}")
    else:
        fail(f"Kalshi leg NOT placed: {result.reject_reason}")

    if args.include_polymarket:
        if result.placed_polymarket:
            ok(f"Polymarket order placed: order_id={result.polymarket_order_id}")
        else:
            fail(f"Polymarket leg NOT placed: {result.reject_reason}")
    else:
        warn("Polymarket leg skipped (run with --include-polymarket to enable)")

    if result.notes:
        print(f"\n  Notes: {'; '.join(result.notes)}")

    print()
    if result.placed_kalshi:
        print(f"{GREEN}{BOLD}Dry test fired a real order!{RESET} Check your Kalshi dashboard.")
        if env == "demo":
            print("    (Demo environment = paper money, but the order is real.)")
        else:
            print(f"    {YELLOW}You are in PROD. Real USD is on the line.{RESET}")
        return 0
    else:
        print(f"{RED}Dry test did NOT place an order.{RESET} See reject reason above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
