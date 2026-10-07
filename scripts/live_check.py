#!/usr/bin/env python3
"""End-to-end validation of the live-trading credentials and stack.

Run AFTER you set up your env vars but BEFORE you flip LIVE_TRADING=true
in production. This script:

    1. Verifies all required env vars are set (LIVE_TRADING aside).
    2. Pings Kalshi /portfolio/balance with the configured RSA key.
       A 200 + balance response proves Kalshi auth works.
    3. Pings Polymarket get_balance_allowance with the configured wallet.
       A non-zero or zero (but valid) USDC response proves Polymarket
       auth + funder address work.
    4. Confirms data/live/pair_mapping.json has YES/NO token IDs.
    5. Confirms the position DB and history files exist.

NO ORDERS ARE PLACED. The script intentionally does NOT set
LIVE_TRADING=true on its own — you'd toggle that when ready.

Usage:
    source ~/.env.live   # whatever file has your credentials
    python scripts/live_check.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


def check_env_vars() -> dict[str, bool]:
    """Verify required env vars are set."""
    print("\n[1/5] Environment variables")
    required = {
        "KALSHI_API_KEY_ID": "Kalshi API key UUID",
        "KALSHI_PRIVATE_KEY_PATH": "path to Kalshi RSA private key (PEM)",
        "POLYMARKET_PRIVATE_KEY": "0x-prefixed Polygon wallet private key",
        "POLYMARKET_FUNDER_ADDRESS": "Polymarket funder/proxy address",
    }
    status: dict[str, bool] = {}
    for var, desc in required.items():
        val = os.environ.get(var)
        if val:
            display = val[:8] + "..." if len(val) > 12 else val
            ok(f"{var} = {display}  ({desc})")
            status[var] = True
        else:
            fail(f"{var} UNSET  ({desc})")
            status[var] = False

    live_armed = os.environ.get("LIVE_TRADING", "").lower() == "true"
    if live_armed:
        warn("LIVE_TRADING=true is set — real orders WILL fire if strategy is run")
    else:
        ok("LIVE_TRADING is not 'true' — system is in paper mode (safe default)")
    return status


def check_kalshi() -> bool:
    print("\n[2/5] Kalshi authentication")
    if not os.environ.get("KALSHI_API_KEY_ID"):
        fail("KALSHI_API_KEY_ID not set — skipping ping")
        return False
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
    if not key_path or not Path(key_path).exists():
        fail(f"KALSHI_PRIVATE_KEY_PATH={key_path} doesn't exist — skipping ping")
        return False

    # Forcibly arm just for the duration of the balance call so the
    # client doesn't refuse to ping. This is the only place we override
    # the safety flag, and we never call place_order from here.
    prior = os.environ.get("LIVE_TRADING")
    os.environ["LIVE_TRADING"] = "true"
    try:
        from src.live.kalshi_orders import KalshiOrderClient
        client = KalshiOrderClient()
        try:
            resp = client.get_balance()
        except Exception as e:
            fail(f"Kalshi /balance failed: {e}")
            return False
        balance_cents = resp.get("balance", None)
        if balance_cents is None:
            fail(f"Kalshi response missing 'balance' field: {resp}")
            return False
        ok(f"Kalshi auth works. Balance: ${balance_cents / 100:.2f}")
        if balance_cents == 0:
            warn("Balance is $0 — fund the account before live trading")
        elif balance_cents < 5000:
            warn(f"Balance < $50 (${balance_cents/100:.2f}) — below canary spec")
        return True
    finally:
        if prior is None:
            os.environ.pop("LIVE_TRADING", None)
        else:
            os.environ["LIVE_TRADING"] = prior


def check_polymarket() -> bool:
    print("\n[3/5] Polymarket authentication")
    if not (os.environ.get("POLYMARKET_PRIVATE_KEY") and os.environ.get("POLYMARKET_FUNDER_ADDRESS")):
        fail("POLYMARKET_PRIVATE_KEY or POLYMARKET_FUNDER_ADDRESS not set — skipping")
        return False

    prior = os.environ.get("LIVE_TRADING")
    os.environ["LIVE_TRADING"] = "true"
    try:
        try:
            from src.live.polymarket_orders import PolymarketOrderClient
        except Exception as e:
            fail(f"Failed to import PolymarketOrderClient: {e}")
            return False
        client = PolymarketOrderClient()
        try:
            balance = client.get_balance_usdc()
        except RuntimeError as e:
            if "py-clob-client" in str(e):
                fail("py-clob-client not installed. Run: pip install py-clob-client>=0.18")
            else:
                fail(f"Polymarket /balance failed: {e}")
            return False
        except Exception as e:
            fail(f"Polymarket /balance failed: {e}")
            return False
        ok(f"Polymarket auth works. USDC balance: ${balance:.2f}")
        if balance == 0:
            warn("USDC balance is $0 — fund the funder address before live trading")
        elif balance < 50:
            warn(f"USDC balance < $50 (${balance:.2f}) — below canary spec")
        return True
    finally:
        if prior is None:
            os.environ.pop("LIVE_TRADING", None)
        else:
            os.environ["LIVE_TRADING"] = prior


def check_pair_mapping_tokens() -> bool:
    print("\n[4/5] pair_mapping.json YES/NO token enrichment")
    mapping_path = Path("data/live/pair_mapping.json")
    if not mapping_path.exists():
        fail(f"{mapping_path} not found")
        return False
    with open(mapping_path) as f:
        mapping = json.load(f)
    n_total = len(mapping)
    n_enriched = sum(1 for v in mapping.values()
                     if v.get("polymarket_yes_token_id"))
    pct = 100 * n_enriched / n_total if n_total else 0.0
    if n_enriched == 0:
        fail(f"NO pairs have YES/NO token IDs. Run: python scripts/enrich_pair_mapping_with_tokens.py")
        return False
    elif n_enriched < n_total * 0.5:
        warn(f"Only {n_enriched}/{n_total} ({pct:.1f}%) pairs have YES/NO tokens. "
             f"Live trades will only fire on enriched pairs.")
        return True
    else:
        ok(f"{n_enriched}/{n_total} ({pct:.1f}%) pairs have YES/NO token IDs")
        return True


def check_safety_files() -> bool:
    print("\n[5/5] Safety files")
    all_ok = True
    pos_db = Path("data/live/positions.db")
    if pos_db.exists():
        ok(f"positions DB exists: {pos_db}")
    else:
        warn(f"positions DB missing — first trade will create it: {pos_db}")
    hist = Path("data/live/position_history.jsonl")
    if hist.exists():
        ok(f"position history exists: {hist}")
    else:
        warn(f"position history missing — first exit will create it: {hist}")
    return all_ok


def main() -> int:
    print("="*60)
    print("LIVE TRADING CREDENTIAL + STACK VALIDATION")
    print("="*60)

    env_status = check_env_vars()
    env_ok = all(env_status.values())

    kalshi_ok = check_kalshi() if env_status.get("KALSHI_API_KEY_ID") else False
    poly_ok = check_polymarket() if env_status.get("POLYMARKET_PRIVATE_KEY") else False
    tokens_ok = check_pair_mapping_tokens()
    safety_ok = check_safety_files()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    summary = [
        ("Env vars", env_ok),
        ("Kalshi auth", kalshi_ok),
        ("Polymarket auth", poly_ok),
        ("Pair token IDs", tokens_ok),
        ("Safety files", safety_ok),
    ]
    for name, status in summary:
        marker = f"{GREEN}PASS{RESET}" if status else f"{RED}FAIL{RESET}"
        print(f"  {marker}  {name}")
    print()
    if all(s for _, s in summary):
        print(f"{GREEN}All checks passed.{RESET} Ready to set LIVE_TRADING=true.")
        print("Recommend: run one paper cycle first to confirm strategy still loads cleanly.")
        return 0
    else:
        print(f"{RED}Some checks failed.{RESET} Fix the above before enabling LIVE_TRADING.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
