"""Full matching pipeline: load -> match -> enrich -> curate -> save.

Usage:
    .venv/bin/python -m src.matching.run_pipeline [--kalshi-dir data/raw/kalshi] \
        [--poly-dir data/raw/polymarket] [--output data/processed/matched_pairs.json] \
        [--min-keyword-score 0.1] [--alpha 0.3] [--skip-curation] [--skip-enrichment]
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import argparse
import json
import logging
import sys
from pathlib import Path

from src.matching.keyword_matcher import generate_candidates
from src.matching.semantic_matcher import SemanticMatcher, score_candidates
from src.matching.scorer import compute_confidence, score_and_rank_candidates
from src.matching.quality_filter import filter_candidates
from src.matching.metadata_enricher import enrich_settlement_criteria
from src.matching.curator import review_candidates
from src.matching.registry import save_registry, load_registry, deduplicate_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCOPE_GATE_THRESHOLD = 30


def load_metadata(path: Path) -> list[dict]:
    """Load metadata from a platform data directory.

    Prefers _metadata_filtered.json if it exists (pre-filtered for matching).
    Falls back to _metadata.json.
    """
    filtered_path = path / "_metadata_filtered.json"
    metadata_path = path / "_metadata.json"

    if filtered_path.exists():
        target = filtered_path
    elif metadata_path.exists():
        target = metadata_path
    else:
        logger.error(
            f"No metadata file found in {path}. "
            "Run Phase 1 ingestion first."
        )
        sys.exit(1)

    with open(target) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} markets from {target}")
    return data


def report_scope_gate(accepted_pairs: list[dict]) -> None:
    """Report matched pair count and scope gate status."""
    count = len(accepted_pairs)
    print(f"\n{'='*60}")
    print("SCOPE GATE REPORT")
    print(f"{'='*60}")
    print(f"  Total accepted pairs: {count}")
    print(f"  Threshold: {SCOPE_GATE_THRESHOLD}")
    if count < SCOPE_GATE_THRESHOLD:
        print("  STATUS: WARNING -- Below threshold!")
        print("  Action: Consider dropping TFT (MOD-07) per ROADMAP")
        logger.warning(
            f"Scope gate: only {count} pairs (< {SCOPE_GATE_THRESHOLD}). "
            "TFT should be dropped."
        )
    else:
        print("  STATUS: PASS -- Sufficient pairs for all models")
        logger.info(
            f"Scope gate: {count} pairs (>= {SCOPE_GATE_THRESHOLD}). "
            "Proceeding with full model set."
        )
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full market matching pipeline"
    )
    parser.add_argument(
        "--kalshi-dir",
        type=Path,
        default=Path("data/raw/kalshi"),
        help="Kalshi raw data directory",
    )
    parser.add_argument(
        "--poly-dir",
        type=Path,
        default=Path("data/raw/polymarket"),
        help="Polymarket raw data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/matched_pairs.json"),
        help="Output path for matched pairs",
    )
    parser.add_argument(
        "--min-keyword-score",
        type=float,
        default=0.1,
        help="Minimum keyword score for candidate generation",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Keyword weight in combined score (0-1)",
    )
    parser.add_argument(
        "--skip-curation",
        action="store_true",
        help="Skip manual curation (auto-accept all candidates)",
    )
    parser.add_argument(
        "--skip-enrichment",
        action="store_true",
        help="Skip settlement criteria enrichment",
    )
    args = parser.parse_args()

    # Step 1: Load metadata
    logger.info("Step 1: Loading platform metadata...")
    kalshi_markets = load_metadata(args.kalshi_dir)
    poly_markets = load_metadata(args.poly_dir)
    logger.info(
        f"Loaded {len(kalshi_markets)} Kalshi + "
        f"{len(poly_markets)} Polymarket markets"
    )

    # Step 2: Score and rank candidates
    logger.info("Step 2: Scoring and ranking candidates...")
    matcher = SemanticMatcher()
    scored = score_and_rank_candidates(
        kalshi_markets,
        poly_markets,
        matcher,
        min_keyword_score=args.min_keyword_score,
        alpha=args.alpha,
    )
    logger.info(f"Generated {len(scored)} scored candidates")

    if not scored:
        logger.error(
            "No candidates found. Check metadata files and category mapping."
        )
        sys.exit(1)

    # Step 3: Deduplicate (one-to-many handling)
    logger.info("Step 3: Deduplicating one-to-many matches...")
    scored = deduplicate_pairs(scored)
    logger.info(f"After dedup: {len(scored)} unique candidates")

    # Step 3.5: Quality filter (temporal, direction, threshold)
    logger.info("Step 3.5: Applying quality filters...")
    scored = filter_candidates(scored)
    logger.info(f"After quality filter: {len(scored)} candidates")

    if not scored:
        logger.error(
            "No candidates survived quality filtering. "
            "Check resolution date proximity and contract compatibility."
        )
        sys.exit(1)

    # Step 4: Enrich with settlement criteria
    if not args.skip_enrichment:
        logger.info("Step 4: Enriching with settlement criteria from APIs...")
        scored = enrich_settlement_criteria(scored)
    else:
        logger.info("Step 4: Skipping settlement enrichment (--skip-enrichment)")
        for c in scored:
            c["kalshi_settlement"] = ""
            c["polymarket_settlement"] = ""
            c["settlement_aligned"] = True
            c["settlement_notes"] = "Enrichment skipped"

    # Step 5: Manual curation
    if not args.skip_curation:
        logger.info("Step 5: Starting manual curation...")
        results = review_candidates(scored, args.output)
    else:
        logger.info(
            "Step 5: Skipping curation (--skip-curation), auto-accepting all"
        )
        for c in scored:
            c["status"] = "accepted"
            c["review_notes"] = "Auto-accepted (--skip-curation)"
            k_id = c["kalshi_market_id"].lower().replace("-", "")[:15]
            p_id = c["polymarket_market_id"][:10]
            c["pair_id"] = f"{k_id}-{p_id}"
        results = scored
        save_registry(results, args.output)

    # Step 6: Scope gate report
    accepted = [r for r in results if r.get("status") == "accepted"]
    report_scope_gate(accepted)

    logger.info(f"Pipeline complete. Output: {args.output}")
    return results


if __name__ == "__main__":
    main()
