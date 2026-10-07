"""Stage 1: Keyword-based candidate generation with Jaccard token overlap.

Normalizes numbers (80k -> 80000), maps synonyms (btc -> bitcoin),
filters by compatible categories, and scores pairs by token overlap.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import re

# Map platform-specific category names to normalized categories
CATEGORY_MAP: dict[str, str] = {
    "Economics": "finance",
    "Financials": "finance",
    "Crypto": "crypto",
    "finance": "finance",
    "crypto": "crypto",
}

# Domain-specific synonym mapping for prediction markets
SYNONYMS: dict[str, str] = {
    "btc": "bitcoin",
    "eth": "ethereum",
    "sol": "solana",
    "fed": "federal_reserve",
    "fomc": "federal_reserve",
    "cpi": "consumer_price_index",
    "gdp": "gross_domestic_product",
}

# Common words that add no discriminative value for matching
STOP_WORDS: set[str] = {
    "will", "the", "be", "by", "in", "at", "for", "of", "a", "an",
    "to", "on", "above", "below", "over", "under", "exceed", "reach",
    "price", "this", "market", "resolves", "yes", "no", "if",
    "before", "after", "hit", "than", "more", "less",
}

# Multipliers for numeric suffixes
_SUFFIX_MULTIPLIERS: dict[str, int] = {
    "k": 1_000,
    "m": 1_000_000,
    "b": 1_000_000_000,
}


def normalize_number(s: str) -> str:
    """Normalize numeric strings with suffixes to plain integers.

    Strips $, commas, % signs. Converts suffixes: 80k -> 80000, 1.5m -> 1500000.
    Returns original string unchanged if not a recognized numeric pattern.
    """
    # Strip currency symbols, commas, percent signs
    cleaned = s.replace("$", "").replace(",", "").replace("%", "")

    # Match number with optional suffix (k, m, b)
    match = re.fullmatch(r"(\d+\.?\d*)(k|m|b)?", cleaned, re.IGNORECASE)
    if not match:
        return s

    num_str, suffix = match.group(1), match.group(2)
    value = float(num_str)

    if suffix:
        multiplier = _SUFFIX_MULTIPLIERS[suffix.lower()]
        value *= multiplier

    # Return as integer string if whole number, else float string
    if value == int(value):
        return str(int(value))
    return str(value)


def extract_key_tokens(text: str) -> set[str]:
    """Extract discriminative tokens from market question text.

    Steps:
    1. Lowercase
    2. Strip punctuation (keep alphanumeric and dots)
    3. Split on whitespace
    4. Remove stop words
    5. Apply synonym mapping
    6. Normalize numbers (80k -> 80000)
    7. Filter tokens shorter than 2 characters
    """
    lowered = text.lower()

    # Pre-process: collapse $N,NNN patterns into plain numbers before
    # general punctuation stripping would split them (e.g. "$80,000" -> "80000")
    lowered = re.sub(
        r"\$?([\d,]+\.?\d*)(k|m|b)?",
        lambda m: normalize_number(m.group(0).replace("$", "").replace(",", "")),
        lowered,
    )

    # Strip remaining punctuation (keep alphanumeric, whitespace, dots)
    cleaned = re.sub(r"[^\w\s.]", " ", lowered)
    raw_tokens = cleaned.split()

    tokens: set[str] = set()
    for token in raw_tokens:
        # Strip trailing dots
        token = token.strip(".")

        # Skip stop words
        if token in STOP_WORDS:
            continue

        # Apply synonym mapping
        token = SYNONYMS.get(token, token)

        # Normalize numbers
        token = normalize_number(token)

        # Filter short tokens
        if len(token) < 2:
            continue

        tokens.add(token)

    return tokens


def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between token sets of two texts.

    Returns |intersection| / |union|. Returns 0.0 if either set is empty.
    """
    tokens1 = extract_key_tokens(text1)
    tokens2 = extract_key_tokens(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    return len(intersection) / len(union)


def generate_candidates(
    kalshi_markets: list[dict],
    poly_markets: list[dict],
    min_keyword_score: float = 0.1,
) -> list[tuple[dict, dict, float]]:
    """Generate candidate match pairs from cross-platform markets.

    Only pairs markets in compatible categories (Crypto<->crypto,
    Economics/Financials<->finance). Filters by minimum Jaccard score.

    Returns list of (kalshi_dict, poly_dict, keyword_score) sorted
    by keyword_score descending.
    """
    candidates: list[tuple[dict, dict, float]] = []

    for km in kalshi_markets:
        km_cat = CATEGORY_MAP.get(km["category"])
        if km_cat is None:
            continue

        for pm in poly_markets:
            pm_cat = CATEGORY_MAP.get(pm["category"])
            if pm_cat is None:
                continue

            # Only pair markets in compatible categories
            if km_cat != pm_cat:
                continue

            score = jaccard_similarity(km["question"], pm["question"])
            if score >= min_keyword_score:
                candidates.append((km, pm, score))

    # Sort by keyword score descending
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates
