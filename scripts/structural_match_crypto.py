"""Structural matching for crypto price markets.

Match on (asset, date, price threshold) instead of text similarity.
Kalshi KXBTC/KXETH series have specific-time price targets.
Polymarket has "Will Bitcoin be above $X on date Y?" markets.
These should match structurally.
"""
import json
import re
from datetime import datetime
from pathlib import Path


def extract_kalshi_info(market: dict) -> dict | None:
    """Extract (asset, date, threshold) from Kalshi market."""
    mid = market.get("market_id", "")
    q = market.get("question", "").lower()

    # Asset detection from market_id
    if mid.startswith(("KXBTC", "KXBTCD")):
        asset = "btc"
    elif mid.startswith(("KXETH", "KXETHD")):
        asset = "eth"
    else:
        return None

    # Skip 15M and hourly ranges
    if "15M" in mid:
        return None

    # Get resolution date
    res_date = market.get("resolution_date", "")
    if not res_date:
        return None
    try:
        dt = datetime.fromisoformat(res_date.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None

    # Extract threshold from market_id (e.g., KXBTC-26APR0219-T59199.99 -> 59199.99)
    match = re.search(r"-[TB]([\d.]+)$", mid)
    if not match:
        return None
    threshold = float(match.group(1))

    return {
        "asset": asset,
        "date": dt.date(),
        "datetime": dt,
        "threshold": threshold,
        "market_id": mid,
        "question": q,
        "resolution_date": res_date,
    }


def extract_poly_info(market: dict) -> dict | None:
    """Extract (asset, date, threshold) from Polymarket market."""
    q = market.get("question", "").lower()

    # Asset
    if "bitcoin" in q or "btc" in q:
        asset = "btc"
    elif "ethereum" in q or "eth " in q or "$eth" in q:
        asset = "eth"
    else:
        return None

    # Must be "above/below X" type
    if not any(k in q for k in ["above", "below"]):
        return None

    # Extract dollar threshold
    threshold_match = re.search(r"\$([\d,]+\.?\d*)(k)?", q)
    if not threshold_match:
        # Try without $
        threshold_match = re.search(r"above ([\d,]+)", q)
        if not threshold_match:
            return None
    threshold_str = threshold_match.group(1).replace(",", "")
    threshold = float(threshold_str)
    # Handle "k" suffix
    if threshold_match.lastindex and threshold_match.lastindex >= 2:
        if threshold_match.group(2) == "k":
            threshold *= 1000
    if threshold < 100:  # Looks like it was meant to be thousands
        threshold *= 1000

    # Get resolution date
    res_date = market.get("resolution_date", "")
    if not res_date:
        return None
    try:
        dt = datetime.fromisoformat(res_date.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None

    return {
        "asset": asset,
        "date": dt.date(),
        "datetime": dt,
        "threshold": threshold,
        "market_id": market.get("market_id", ""),
        "question": q,
        "resolution_date": res_date,
    }


def main():
    # Load metadata
    kalshi = json.load(open("data/raw/kalshi/_metadata.json"))
    poly = json.load(open("data/raw/polymarket/_metadata.json"))

    print("Extracting structural info...")

    # Extract from Kalshi crypto markets
    kalshi_crypto = [m for m in kalshi if m.get("category") == "Crypto"]
    print(f"Kalshi crypto markets: {len(kalshi_crypto)}")

    kalshi_info = []
    for m in kalshi_crypto:
        info = extract_kalshi_info(m)
        if info:
            kalshi_info.append((info, m))
    print(f"Kalshi with extractable info: {len(kalshi_info)}")

    # Extract from Polymarket
    poly_info = []
    for m in poly:
        info = extract_poly_info(m)
        if info:
            poly_info.append((info, m))
    print(f"Polymarket with extractable info: {len(poly_info)}")

    # Build Kalshi index by (asset, date)
    from collections import defaultdict
    kalshi_by_date = defaultdict(list)
    for info, m in kalshi_info:
        key = (info["asset"], info["date"])
        kalshi_by_date[key].append((info, m))

    # For each Polymarket market, find nearest Kalshi match on same date
    matched = []
    for p_info, p_market in poly_info:
        key = (p_info["asset"], p_info["date"])
        candidates = kalshi_by_date.get(key, [])

        # Find closest threshold match
        best = None
        best_diff = float("inf")
        for k_info, k_market in candidates:
            ratio = max(k_info["threshold"], p_info["threshold"]) / max(1, min(k_info["threshold"], p_info["threshold"]))
            if ratio <= 1.05:  # Within 5%
                diff = abs(k_info["threshold"] - p_info["threshold"])
                if diff < best_diff:
                    best_diff = diff
                    best = (k_info, k_market)

        if best:
            k_info, k_market = best
            matched.append({
                "kalshi_market_id": k_market["market_id"],
                "kalshi_question": k_market["question"],
                "kalshi_resolution_date": k_market["resolution_date"],
                "polymarket_market_id": p_market["market_id"],
                "polymarket_question": p_market["question"],
                "polymarket_resolution_date": p_market["resolution_date"],
                "asset": p_info["asset"],
                "match_date": str(p_info["date"]),
                "kalshi_threshold": k_info["threshold"],
                "polymarket_threshold": p_info["threshold"],
                "confidence_score": 0.95,  # Structural match = high confidence
                "keyword_score": 0.5,
                "semantic_score": 0.5,
                "category": "crypto",
                "status": "accepted",
                "review_notes": "Structural match on (asset, date, threshold)",
                "kalshi_settlement": "",
                "polymarket_settlement": "",
                "settlement_aligned": True,
                "settlement_notes": "",
            })

    print(f"\nStructural matches: {len(matched)}")

    # Dedupe: one Kalshi market can only be matched once
    seen_kalshi = set()
    deduped = []
    for m in sorted(matched, key=lambda x: abs(x["kalshi_threshold"] - x["polymarket_threshold"])):
        if m["kalshi_market_id"] in seen_kalshi:
            continue
        seen_kalshi.add(m["kalshi_market_id"])
        k_id = m["kalshi_market_id"].lower().replace("-", "")[:15]
        p_id = m["polymarket_market_id"][:10]
        m["pair_id"] = f"{k_id}-{p_id}"
        deduped.append(m)

    print(f"After dedup: {len(deduped)}")

    # Save
    output_path = Path("data/processed/crypto_structural_pairs.json")
    with open(output_path, "w") as f:
        json.dump(deduped, f, indent=2)
    print(f"Saved to {output_path}")

    # Sample
    print("\nTop 10 structural matches:")
    for m in deduped[:10]:
        print(f'  {m["asset"].upper()} {m["match_date"]} K=${m["kalshi_threshold"]:,.0f} P=${m["polymarket_threshold"]:,.0f}')
        print(f'    K: {m["kalshi_question"][:75]}')
        print(f'    P: {m["polymarket_question"][:75]}')


if __name__ == "__main__":
    main()
