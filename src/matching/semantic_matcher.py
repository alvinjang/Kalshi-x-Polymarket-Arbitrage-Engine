"""Stage 2: Sentence-transformer semantic similarity scoring.

Uses all-MiniLM-L6-v2 to encode market questions into 384-dim embeddings
and compute pairwise cosine similarity for candidate pairs.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import numpy as np
from sentence_transformers import SentenceTransformer, util


class SemanticMatcher:
    """Wraps all-MiniLM-L6-v2 for batch encoding and pairwise similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode texts into 384-dim embeddings. Returns numpy array."""
        return self.model.encode(texts, convert_to_numpy=True, batch_size=32)

    def score_pairs(
        self,
        texts_a: list[str],
        texts_b: list[str],
    ) -> list[float]:
        """Compute pairwise cosine similarity for aligned text lists.

        texts_a[i] is compared to texts_b[i]. Returns list of floats in [0, 1].
        """
        emb_a = self.encode_batch(texts_a)
        emb_b = self.encode_batch(texts_b)

        # Compute element-wise cosine similarity (diagonal of full matrix)
        # Normalize embeddings
        norms_a = np.linalg.norm(emb_a, axis=1, keepdims=True)
        norms_b = np.linalg.norm(emb_b, axis=1, keepdims=True)
        emb_a_normed = emb_a / norms_a
        emb_b_normed = emb_b / norms_b

        # Element-wise dot product for aligned pairs
        similarities = np.sum(emb_a_normed * emb_b_normed, axis=1)

        return [float(s) for s in similarities]


def score_candidates(
    matcher: SemanticMatcher,
    candidates: list[tuple[dict, dict, float]],
) -> list[dict]:
    """Add semantic_score to each candidate tuple.

    Input: list of (kalshi_dict, poly_dict, keyword_score)
    Output: list of dicts with all fields plus semantic_score
    """
    if not candidates:
        return []

    kalshi_questions = [c[0]["question"] for c in candidates]
    poly_questions = [c[1]["question"] for c in candidates]
    scores = matcher.score_pairs(kalshi_questions, poly_questions)

    results = []
    for (km, pm, kw_score), sem_score in zip(candidates, scores):
        results.append({
            "kalshi_market_id": km["market_id"],
            "polymarket_market_id": pm["market_id"],
            "kalshi_question": km["question"],
            "polymarket_question": pm["question"],
            "category": km["category"],
            "kalshi_resolution_date": km.get("resolution_date", ""),
            "polymarket_resolution_date": pm.get("resolution_date", ""),
            "keyword_score": kw_score,
            "semantic_score": sem_score,
        })
    return results
