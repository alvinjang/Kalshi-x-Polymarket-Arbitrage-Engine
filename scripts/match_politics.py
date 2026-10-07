"""Match political markets between Kalshi and Polymarket.

Uses the existing matching pipeline but with political metadata files.
"""
import json
import logging
import sys
from pathlib import Path

from src.matching.semantic_matcher import SemanticMatcher
from src.matching.scorer import score_and_rank_candidates
from src.matching.quality_filter import filter_candidates
from src.matching.registry import deduplicate_pairs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    # Load filtered metadata
    kalshi = json.load(open("data/raw/kalshi/_politics_metadata_filtered.json"))
    poly = json.load(open("data/raw/polymarket/_politics_metadata_filtered.json"))
    logger.info(f"Loaded {len(kalshi)} Kalshi + {len(poly)} Polymarket political markets")

    # Run matching
    logger.info("Running semantic matching pipeline...")
    matcher = SemanticMatcher()
    scored = score_and_rank_candidates(
        kalshi, poly, matcher,
        min_keyword_score=0.1, alpha=0.3,
    )
    logger.info(f"Generated {len(scored)} scored candidates")

    if not scored:
        logger.error("No candidates found")
        sys.exit(1)

    # Deduplicate
    scored = deduplicate_pairs(scored)
    logger.info(f"After dedup: {len(scored)} unique candidates")

    # Quality filter
    scored = filter_candidates(scored)
    logger.info(f"After quality filter: {len(scored)} candidates")

    # Auto-accept all (we'll curate after)
    for c in scored:
        c["status"] = "accepted"
        c["review_notes"] = "Auto-accepted politics pair"
        c["kalshi_settlement"] = ""
        c["polymarket_settlement"] = ""
        c["settlement_aligned"] = True
        c["settlement_notes"] = "Enrichment skipped"
        k_id = c["kalshi_market_id"].lower().replace("-", "")[:15]
        p_id = c["polymarket_market_id"][:10]
        c["pair_id"] = f"{k_id}-{p_id}"

    # Save
    output_path = Path("data/processed/politics_pairs.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scored, f, indent=2)
    logger.info(f"Saved {len(scored)} politics pairs to {output_path}")

    # Summary
    print(f"\n=== Politics Matching Results ===")
    print(f"Total matched pairs: {len(scored)}")
    if scored:
        scores = [p["confidence_score"] for p in scored]
        print(f"Score distribution:")
        print(f"  > 0.8: {sum(1 for s in scores if s > 0.8)}")
        print(f"  0.6-0.8: {sum(1 for s in scores if 0.6 <= s < 0.8)}")
        print(f"\nTop 10 matches:")
        for p in sorted(scored, key=lambda x: -x["confidence_score"])[:10]:
            print(f'  [{p["confidence_score"]:.3f}] {p["kalshi_question"][:60]}')
            print(f'         {p["polymarket_question"][:60]}')


if __name__ == "__main__":
    main()
