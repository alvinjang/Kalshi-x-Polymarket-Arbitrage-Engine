"""Pre-filter metadata to realistic matching candidates.

The full metadata (2M Kalshi + 292k Polymarket) is too large for
pairwise matching. This script filters both sides to markets that
have realistic cross-platform overlap:

Kalshi: Keep only unique event-level markets (one representative per topic/date)
Polymarket: Keep only genuine econ/crypto price markets (filter noise)

Output: filtered _metadata.json files in the same directories.
"""
import json
import re
from pathlib import Path


def filter_kalshi(metadata: list[dict]) -> list[dict]:
    """Filter Kalshi to unique event-level representatives."""
    # Economics: deduplicate by event type + date
    econ_events = {}
    for m in metadata:
        if m.get("category") != "Economics":
            continue
        # Group by question pattern (strip strike values)
        q = m.get("question", "")
        # Normalize: "CPI year-over-year in Feb 2026?" -> "CPI year-over-year Feb 2026"
        key = re.sub(r"\b(above|below|between|to)\b.*?\d", "", q)
        key = re.sub(r"[?\d.]+$", "", key).strip()
        res_date = m.get("resolution_date", "")[:10]
        event_key = f"{key}_{res_date}"
        if event_key not in econ_events:
            econ_events[event_key] = m

    # Crypto: keep only BTC/ETH, deduplicate by date
    crypto_events = {}
    for m in metadata:
        if m.get("category") != "Crypto":
            continue
        q = m.get("question", "").lower()
        mid = m.get("market_id", "")
        # Only BTC and ETH
        if not any(k in q for k in ["bitcoin", "btc"]) and not any(k in q for k in ["ethereum", "eth"]):
            if not any(k in mid.upper() for k in ["BTC", "ETH"]):
                continue
        # Group by date
        res_date = m.get("resolution_date", "")[:10]
        asset = "btc" if any(k in q for k in ["bitcoin", "btc"]) or "BTC" in mid.upper() else "eth"
        event_key = f"{asset}_{res_date}"
        if event_key not in crypto_events:
            crypto_events[event_key] = m

    # Financials: keep unique events for major indices
    fin_events = {}
    for m in metadata:
        if m.get("category") != "Financials":
            continue
        q = m.get("question", "").lower()
        if not any(k in q for k in ["nasdaq", "s&p", "sp500", "dow"]):
            continue
        res_date = m.get("resolution_date", "")[:10]
        event_key = f"fin_{res_date}"
        if event_key not in fin_events:
            fin_events[event_key] = m

    filtered = list(econ_events.values()) + list(crypto_events.values()) + list(fin_events.values())
    return filtered


def filter_polymarket(metadata: list[dict]) -> list[dict]:
    """Filter Polymarket to genuine econ and crypto price markets."""
    filtered = []
    seen = set()

    for m in metadata:
        q = m.get("question", "").lower()
        mid = m.get("market_id", "")

        # Skip duplicates
        if mid in seen:
            continue

        is_econ = any(k in q for k in [
            "inflation", "cpi", "gdp", "unemployment", "fed ",
            "fomc", "rate cut", "rate hike", "recession",
            "interest rate", "treasury", "jobs report",
            "nonfarm", "payroll"
        ])

        is_crypto_price = (
            any(k in q for k in ["bitcoin", "btc", "ethereum", "eth", "solana", "sol "])
            and any(k in q for k in ["price", "above", "below", "hit", "reach", "close", "$"])
        )

        is_finance = any(k in q for k in [
            "s&p", "nasdaq", "dow", "stock market", "spy "
        ])

        if is_econ or is_crypto_price or is_finance:
            seen.add(mid)
            filtered.append(m)

    return filtered


def main():
    kalshi_path = Path("data/raw/kalshi/_metadata.json")
    poly_path = Path("data/raw/polymarket/_metadata.json")

    print("Loading metadata...")
    kalshi = json.load(open(kalshi_path))
    poly = json.load(open(poly_path))
    print(f"  Kalshi: {len(kalshi)} markets")
    print(f"  Polymarket: {len(poly)} markets")

    print("\nFiltering Kalshi...")
    kalshi_filtered = filter_kalshi(kalshi)
    print(f"  Kalshi filtered: {len(kalshi_filtered)} markets")

    by_cat = {}
    for m in kalshi_filtered:
        c = m.get("category", "unknown")
        by_cat[c] = by_cat.get(c, 0) + 1
    for c, n in sorted(by_cat.items()):
        print(f"    {c}: {n}")

    print("\nFiltering Polymarket...")
    poly_filtered = filter_polymarket(poly)
    print(f"  Polymarket filtered: {len(poly_filtered)} markets")

    by_cat = {}
    for m in poly_filtered:
        c = m.get("category", "unknown")
        by_cat[c] = by_cat.get(c, 0) + 1
    for c, n in sorted(by_cat.items()):
        print(f"    {c}: {n}")

    # Save filtered versions
    kalshi_out = Path("data/raw/kalshi/_metadata_filtered.json")
    poly_out = Path("data/raw/polymarket/_metadata_filtered.json")

    with open(kalshi_out, "w") as f:
        json.dump(kalshi_filtered, f, indent=2)
    with open(poly_out, "w") as f:
        json.dump(poly_filtered, f, indent=2)

    print(f"\nSaved filtered metadata:")
    print(f"  {kalshi_out}")
    print(f"  {poly_out}")
    print(f"\nTotal candidate comparisons: ~{len(kalshi_filtered) * len(poly_filtered):,}")


if __name__ == "__main__":
    main()
