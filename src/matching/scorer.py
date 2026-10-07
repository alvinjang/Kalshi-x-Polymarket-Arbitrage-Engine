"""Combined confidence scoring and ranking for matched candidates.

Merges keyword overlap (Stage 1) and semantic similarity (Stage 2)
into a single 0-1 confidence score using a weighted formula.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from src.matching.keyword_matcher import generate_candidates
from src.matching.semantic_matcher import SemanticMatcher, score_candidates


def compute_confidence(
    keyword_score: float,
    semantic_score: float,
    alpha: float = 0.3,
) -> float:
    """Combine keyword and semantic scores into single 0-1 confidence.

    Formula: alpha * keyword_score + (1 - alpha) * semantic_score
    Default alpha=0.3 means 30% keyword weight, 70% semantic weight.
    Semantic weighted higher because it handles paraphrases and synonyms.
    """
    return alpha * keyword_score + (1 - alpha) * semantic_score


def score_and_rank_candidates(
    kalshi_markets: list[dict],
    poly_markets: list[dict],
    matcher: SemanticMatcher,
    min_keyword_score: float = 0.1,
    alpha: float = 0.3,
) -> list[dict]:
    """Full pipeline: generate candidates, score semantically, compute confidence, rank.

    Returns list of dicts sorted by confidence_score descending. Each dict has:
    - kalshi_market_id, polymarket_market_id
    - kalshi_question, polymarket_question
    - category
    - kalshi_resolution_date, polymarket_resolution_date
    - keyword_score, semantic_score, confidence_score
    """
    # Stage 1: keyword candidates
    candidates = generate_candidates(kalshi_markets, poly_markets, min_keyword_score)
    if not candidates:
        return []

    # Stage 2: semantic scoring
    scored = score_candidates(matcher, candidates)

    # Combined confidence
    for item in scored:
        item["confidence_score"] = compute_confidence(
            item["keyword_score"], item["semantic_score"], alpha
        )

    # Sort by confidence descending
    scored.sort(key=lambda x: x["confidence_score"], reverse=True)
    return scored
