"""Match crypto price markets between Kalshi and Polymarket."""
import json
import logging
import re
import sys
from pathlib import Path

from src.matching.semantic_matcher import SemanticMatcher
from src.matching.scorer import score_and_rank_candidates
from src.matching.quality_filter import filter_candidates
from src.matching.registry import deduplicate_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def filter_kalshi_crypto(metadata: list[dict]) -> list[dict]:
    """Keep only MAXY yearly high markets (cleanest match to Polymarket milestones).

    Skip the daily strike markets — too many strikes per event clogs matching.
    Focus on BTC/ETH yearly highs which align with Polymarket "Will BTC reach $X?" style.
    """
    filtered = []
    # MAXY = annual high price targets — directly matches Polymarket milestones
    matchable_prefixes = ("KXBTCMAXY", "KXETHMAXY", "KXSOLMAXY")
    for m in metadata:
        mid = m.get("market_id", "")
        if any(mid.startswith(p) for p in matchable_prefixes):
            filtered.append(m)

    # Dedupe by event prefix (strip strike suffix)
    events = {}
    for m in filtered:
        mid = m.get("market_id", "")
        prefix = re.sub(r"-[\d.]+$", "", mid)
        if prefix not in events:
            events[prefix] = m
    return list(events.values())


def filter_polymarket_crypto(metadata: list[dict]) -> list[dict]:
    """Filter Polymarket for genuine crypto price target markets."""
    filtered = []
    seen = set()
    for m in metadata:
        q = m.get("question", "").lower()
        mid = m.get("market_id", "")
        if mid in seen:
            continue

        # Must mention BTC/ETH/SOL/XRP/DOGE
        has_crypto = any(k in q for k in [
            "bitcoin", "btc", "ethereum", "eth ", "solana", "sol ",
            "ripple", "xrp", "doge", "dogecoin"
        ])
        # Must be a price target question
        has_price = ("$" in q or "above" in q or "below" in q or "hit" in q
                     or "reach" in q or "price" in q)
        # Must have a number
        has_number = any(c.isdigit() for c in q)

        if has_crypto and has_price and has_number:
            seen.add(mid)
            filtered.append(m)
    return filtered


def main():
    kalshi = json.load(open("data/raw/kalshi/_metadata.json"))
    poly = json.load(open("data/raw/polymarket/_metadata.json"))

    kalshi_crypto = [m for m in kalshi if m.get("category") == "Crypto"]
    logger.info(f"Kalshi crypto total: {len(kalshi_crypto)}")

    kalshi_filtered = filter_kalshi_crypto(kalshi_crypto)
    poly_filtered = filter_polymarket_crypto(poly)
    logger.info(f"Filtered: Kalshi {len(kalshi_filtered)}, Polymarket {len(poly_filtered)}")
    logger.info(f"Candidate comparisons: ~{len(kalshi_filtered) * len(poly_filtered):,}")

    # Normalize categories
    for m in kalshi_filtered:
        m["category"] = "Crypto"
    for m in poly_filtered:
        m["category"] = "crypto"

    matcher = SemanticMatcher()
    scored = score_and_rank_candidates(
        kalshi_filtered, poly_filtered, matcher,
        min_keyword_score=0.1, alpha=0.3,
    )
    logger.info(f"Generated {len(scored)} scored candidates")

    scored = deduplicate_pairs(scored)
    logger.info(f"After dedup: {len(scored)} unique candidates")

    scored = filter_candidates(scored)
    logger.info(f"After quality filter: {len(scored)} candidates")

    # Auto-accept pairs with confidence >= 0.70
    accepted = []
    for c in scored:
        if c.get("confidence_score", 0) < 0.70:
            continue
        c["status"] = "accepted"
        c["review_notes"] = "Auto-accepted crypto pair"
        c["kalshi_settlement"] = ""
        c["polymarket_settlement"] = ""
        c["settlement_aligned"] = True
        c["settlement_notes"] = "Enrichment skipped"
        k_id = c["kalshi_market_id"].lower().replace("-", "")[:15]
        p_id = c["polymarket_market_id"][:10]
        c["pair_id"] = f"{k_id}-{p_id}"
        accepted.append(c)

    output_path = Path("data/processed/crypto_pairs.json")
    with open(output_path, "w") as f:
        json.dump(accepted, f, indent=2)
    logger.info(f"Saved {len(accepted)} crypto pairs (confidence >= 0.70) to {output_path}")

    print(f"\n=== Crypto Matching Results ===")
    print(f"Total filtered candidates: {len(scored)}")
    print(f"Accepted (conf >= 0.70): {len(accepted)}")
    if accepted:
        scores = [p["confidence_score"] for p in accepted]
        print(f"  > 0.9: {sum(1 for s in scores if s > 0.9)}")
        print(f"  0.8-0.9: {sum(1 for s in scores if 0.8 <= s < 0.9)}")
        print(f"  0.7-0.8: {sum(1 for s in scores if 0.7 <= s < 0.8)}")
        print(f"\nTop 10 matches:")
        for p in sorted(accepted, key=lambda x: -x["confidence_score"])[:10]:
            print(f'  [{p["confidence_score"]:.3f}] {p["kalshi_question"][:60]}')
            print(f'         {p["polymarket_question"][:60]}')


if __name__ == "__main__":
    main()
