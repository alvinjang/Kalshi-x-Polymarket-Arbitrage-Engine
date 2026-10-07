"""CLI-based manual curation interface for matched pair candidates."""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import logging
from pathlib import Path

from src.matching.registry import save_registry

logger = logging.getLogger(__name__)


def review_candidates(
    candidates: list[dict],
    output_path: Path,
) -> list[dict]:
    """Interactive CLI for reviewing matched pair candidates.

    Presents each candidate with scores, settlement criteria, and options.
    Auto-saves to output_path after each decision.
    Returns list of reviewed candidates with status field.
    """
    results: list[dict] = []
    total = len(candidates)

    for i, candidate in enumerate(candidates, 1):
        print(f"\n{'='*60}")
        print(
            f"Candidate {i}/{total} "
            f"(confidence: {candidate.get('confidence_score', 0):.3f})"
        )
        print(f"{'='*60}")
        print(f"  Kalshi:       {candidate['kalshi_question']}")
        print(f"  Polymarket:   {candidate['polymarket_question']}")
        print(f"  Category:     {candidate.get('category', 'unknown')}")
        print(f"  Keyword:      {candidate.get('keyword_score', 0):.3f}")
        print(f"  Semantic:     {candidate.get('semantic_score', 0):.3f}")
        print(f"  Confidence:   {candidate.get('confidence_score', 0):.3f}")
        print(f"  K-Resolve:    {candidate.get('kalshi_resolution_date', 'N/A')}")
        print(f"  P-Resolve:    {candidate.get('polymarket_resolution_date', 'N/A')}")
        if candidate.get("kalshi_settlement"):
            k_settle = candidate["kalshi_settlement"]
            print(
                f"  K-Settlement: {k_settle[:120]}"
                f"{'...' if len(k_settle) > 120 else ''}"
            )
        if candidate.get("polymarket_settlement"):
            p_settle = candidate["polymarket_settlement"]
            print(
                f"  P-Settlement: {p_settle[:120]}"
                f"{'...' if len(p_settle) > 120 else ''}"
            )
        if candidate.get("settlement_aligned") is not None:
            print(f"  Aligned:      {candidate['settlement_aligned']}")
        if candidate.get("settlement_notes"):
            print(f"  Notes:        {candidate['settlement_notes']}")
        print()

        while True:
            choice = (
                input("  [a]ccept / [r]eject / [f]lag / [s]kip / [q]uit: ")
                .lower()
                .strip()
            )
            if choice in ("a", "r", "f", "s", "q"):
                break
            print("  Invalid choice. Use a/r/f/s/q.")

        if choice == "q":
            break

        status_map = {
            "a": "accepted",
            "r": "rejected",
            "f": "flagged",
            "s": "skipped",
        }
        candidate["status"] = status_map[choice]

        if choice in ("a", "f"):
            notes = input("  Notes (optional, Enter to skip): ").strip()
            candidate["review_notes"] = notes
        else:
            candidate["review_notes"] = ""

        # Generate pair_id slug from market IDs
        k_id = candidate["kalshi_market_id"].lower().replace("-", "")[:15]
        p_id = candidate["polymarket_market_id"][:10]
        candidate["pair_id"] = f"{k_id}-{p_id}"

        results.append(candidate)

        # Auto-save after each decision
        save_registry(results, output_path)

    return results
