"""Registry I/O for matched_pairs.json."""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_registry(pairs: list[dict], path: Path) -> None:
    """Write matched pairs to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(pairs, f, indent=2)
    logger.info(f"Saved {len(pairs)} pairs to {path}")


def load_registry(path: Path) -> list[dict]:
    """Load matched pairs from JSON file. Returns [] if file missing."""
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def deduplicate_pairs(pairs: list[dict]) -> list[dict]:
    """Keep only the highest-confidence match for each market_id.

    If the same kalshi_market_id or polymarket_market_id appears in multiple
    pairs, keep only the pair with the highest confidence_score.
    """
    # Sort by confidence descending so first-seen is highest
    sorted_pairs = sorted(
        pairs, key=lambda x: x.get("confidence_score", 0), reverse=True
    )
    seen_kalshi: set[str] = set()
    seen_poly: set[str] = set()
    deduped: list[dict] = []
    for pair in sorted_pairs:
        k_id = pair["kalshi_market_id"]
        p_id = pair["polymarket_market_id"]
        if k_id in seen_kalshi or p_id in seen_poly:
            logger.info(f"Dedup: dropping duplicate pair {k_id} <-> {p_id}")
            continue
        seen_kalshi.add(k_id)
        seen_poly.add(p_id)
        deduped.append(pair)
    return deduped
