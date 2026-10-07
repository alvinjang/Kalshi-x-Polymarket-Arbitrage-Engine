"""Match climate/weather and company markets between Kalshi and Polymarket."""
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


def filter_kalshi(metadata: list[dict]) -> list[dict]:
    """Dedupe by event prefix."""
    events = {}
    for m in metadata:
        mid = m.get("market_id", "")
        prefix = re.sub(r"-[TY][\d.A-Z]+$", "", mid)
        prefix = re.sub(r"-[A-Z]+\d+$", "", prefix)
        if prefix not in events:
            events[prefix] = m
    return list(events.values())


def filter_polymarket_weather(metadata: list[dict]) -> list[dict]:
    """Filter for weather/climate/company markets."""
    keywords = [
        "temperature", "hurricane", "snow", "rain", "heat", "cold", "storm",
        "weather", "climate", "nyc temp", "tornado", "earthquake",
        "apple", "tesla", "amazon", "google", "microsoft", "nvidia", "meta",
        "ipo", "earnings", "stock", "acquisition", "merger",
    ]
    filtered = []
    seen = set()
    for m in metadata:
        q = m.get("question", "").lower()
        mid = m.get("market_id", "")
        if mid in seen:
            continue
        if not any(k in q for k in keywords):
            continue
        seen.add(mid)
        filtered.append(m)
    return filtered


def main():
    kalshi = json.load(open("data/raw/kalshi/_climate_companies_metadata.json"))
    poly = json.load(open("data/raw/polymarket/_metadata.json"))

    # Normalize categories
    for m in kalshi:
        m["category"] = "Weather" if "Climate" in m.get("category", "") else "Companies"
    for m in filter_polymarket_weather(poly):
        m["category"] = "weather"

    kalshi_filtered = filter_kalshi(kalshi)
    poly_filtered = filter_polymarket_weather(poly)
    for m in poly_filtered:
        m["category"] = "weather"

    logger.info(f"Filtered: Kalshi {len(kalshi_filtered)}, Polymarket {len(poly_filtered)}")
    logger.info(f"Candidate comparisons: ~{len(kalshi_filtered) * len(poly_filtered):,}")

    if len(kalshi_filtered) * len(poly_filtered) > 50_000_000:
        logger.warning("Too many comparisons, limiting Kalshi to 3000 most recent")
        kalshi_filtered = sorted(kalshi_filtered, key=lambda x: x.get("resolution_date", ""), reverse=True)[:3000]

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

    for c in scored:
        c["status"] = "accepted"
        c["review_notes"] = "Auto-accepted weather/companies pair"
        c["kalshi_settlement"] = ""
        c["polymarket_settlement"] = ""
        c["settlement_aligned"] = True
        c["settlement_notes"] = "Enrichment skipped"
        k_id = c["kalshi_market_id"].lower().replace("-", "")[:15]
        p_id = c["polymarket_market_id"][:10]
        c["pair_id"] = f"{k_id}-{p_id}"

    output_path = Path("data/processed/weather_companies_pairs.json")
    with open(output_path, "w") as f:
        json.dump(scored, f, indent=2)
    logger.info(f"Saved {len(scored)} pairs to {output_path}")

    print(f"\n=== Weather/Companies Matching Results ===")
    print(f"Total matched pairs: {len(scored)}")
    if scored:
        scores = [p["confidence_score"] for p in scored]
        print(f"  > 0.8: {sum(1 for s in scores if s > 0.8)}")
        print(f"  0.6-0.8: {sum(1 for s in scores if 0.6 <= s < 0.8)}")
        print(f"\nTop 10 matches:")
        for p in sorted(scored, key=lambda x: -x["confidence_score"])[:10]:
            print(f'  [{p["confidence_score"]:.3f}] {p["kalshi_question"][:60]}')
            print(f'         {p["polymarket_question"][:60]}')


if __name__ == "__main__":
    main()
