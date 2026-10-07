"""Post-matching quality filters for candidate pairs.

Filters out false positives caused by:
- Temporal mismatch (same question, different year)
- Contract direction mismatch (above vs below vs exactly)
- Threshold number mismatch (4.3% vs 3.4%)

Applied after scoring and deduplication, before curation.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import logging
import re
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Maximum days between resolution dates for a valid match
MAX_RESOLUTION_GAP_DAYS = 30

# Minimum confidence score to keep
MIN_CONFIDENCE = 0.6

# --- Rule 10 (asset-class consistency) token vocabularies ---
# Used to reject numerically-coincident cross-asset strikes such as
# KXWTIMAX-26DEC31-T130 ("WTI $130") vs a Bitcoin-$130K Polymarket
# market that share the numeric "130" but resolve on unrelated assets.
COMMODITY_ASSET_TOKENS: tuple[str, ...] = (
    "WTI", "CRUDE", "OIL", "DIESEL", "GAS", "GASOLINE",
    "HEATINGOIL", "HEATING OIL", "BRENT",
)
CRYPTO_ASSET_TOKENS: tuple[str, ...] = (
    "BITCOIN", "BTC", "ETHEREUM", "ETH", "DOGECOIN", "DOGE",
    "SOLANA", "SOL", "CARDANO", "ADA", "XRP", "LITECOIN", "LTC",
)
# Kalshi ticker prefixes that pre-identify asset class when title is ambiguous
KALSHI_COMMODITY_PREFIXES: tuple[str, ...] = (
    "KXWTI", "KXWTID", "KXWTIW", "KXWTIMAX",
    "KXAAAGASD", "KXAAAGASW", "KXAAAGASM",
    "KXBRENTMON", "KXDIESEL", "KXHEATINGOIL", "KXCRUDE", "KXGASOLINE",
)
KALSHI_CRYPTO_PREFIXES: tuple[str, ...] = (
    "KXBTC", "KXETH", "KXBITCOIN", "KXETHEREUM",
    "KXDOGE", "KXSOL", "KXADA", "KXXRP", "KXLTC",
)


def parse_date(date_str: str) -> datetime | None:
    """Parse ISO date string to datetime, handling various formats."""
    if not date_str:
        return None
    try:
        # Handle full ISO format with timezone
        clean = date_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except (ValueError, TypeError):
        pass
    try:
        # Handle date-only format
        return datetime.strptime(date_str[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def extract_direction(question: str) -> str | None:
    """Extract contract direction from question text.

    Returns 'above', 'below', 'exactly', 'between', or None.
    """
    q = question.lower()

    if any(k in q for k in ["above", "over", "exceed", "more than", "higher than", "at least"]):
        return "above"
    if any(k in q for k in ["below", "under", "less than", "lower than", "drop to"]):
        return "below"
    if any(k in q for k in ["exactly", "be exactly"]):
        return "exactly"
    if "between" in q:
        return "between"
    return None


def extract_threshold(question: str) -> float | None:
    """Extract the primary numeric threshold from a question.

    E.g., "Will unemployment be above 4.3%?" -> 4.3
          "Will BTC be above $80,000?" -> 80000
    """
    q = question.lower()

    # Match patterns like: above 4.3%, above $80,000, above 80000, above 4.3
    patterns = [
        r"(?:above|below|exceed|over|under|exactly|be)\s+\$?([\d,]+\.?\d*)\s*%?",
        r"\$([\d,]+\.?\d*)",
        r"([\d,]+\.?\d*)\s*%",
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            num_str = match.group(1).replace(",", "")
            try:
                return float(num_str)
            except ValueError:
                continue
    return None


def directions_compatible(dir1: str | None, dir2: str | None) -> bool:
    """Check if two contract directions are compatible for arbitrage.

    Compatible: above/above, below/below, None/anything
    Incompatible: above/below, above/exactly, below/exactly
    """
    if dir1 is None or dir2 is None:
        return True  # Can't determine, don't filter
    if dir1 == dir2:
        return True
    # above+between or below+between are borderline, allow
    if "between" in (dir1, dir2):
        return True
    return False


def thresholds_compatible(t1: float | None, t2: float | None) -> bool:
    """Check if two thresholds are close enough to be the same contract.

    Allows 5% relative tolerance or 0.5 absolute tolerance (for percentages).
    """
    if t1 is None or t2 is None:
        return True  # Can't determine, don't filter

    # Absolute tolerance for small numbers (percentages)
    if abs(t1 - t2) <= 0.5:
        return True

    # Relative tolerance for large numbers (dollar amounts)
    if t1 != 0 and t2 != 0:
        ratio = max(t1, t2) / min(t1, t2)
        if ratio <= 1.05:  # Within 5%
            return True

    return False


# --------------------------------------------------------------------
# Live-trading quality filter.
#
# Operates on the active_matches.json schema (kalshi_ticker, kalshi_title,
# poly_title, similarity, ...). Rejects the specific garbage patterns
# found in live paper-trading data:
#
#   1. KXNBAWINS-* (season-wins O/U) matched against NBA champion markets
#   2. Fed decision contracts with mismatched month/year
#   3. Cross-topic matches (e.g. cabinet visit vs presidential nominee)
#   4. Similarity below a floor
#
# Used by:
#   - src/live/collector.py::_load_live_pairs  (skip collection)
#   - src/live/strategy.py                      (skip entry)
#   - scripts/apply_match_filter.py             (offline audit)
# --------------------------------------------------------------------

# Minimum cosine similarity to accept as a match. Kept LOW on purpose:
# empirically on live paper-trading data, winning oil pairs sit in the
# 0.79-0.82 range while losing NBA-wins garbage sits at 0.73-0.74 —
# huge overlap. Similarity is NOT a reliable signal here; the structural
# rules below (ticker-prefix checks, year mismatch, cross-topic) do the
# real work. This floor is just a sanity guard for future data.
MIN_ACTIVE_SIMILARITY = 0.70

# Kalshi ticker prefixes that encode a "number-of-wins" season contract.
# These are structurally incompatible with champion/finals markets.
_SEASON_WINS_TICKER_PREFIXES = ("KXNBAWINS", "KXNFLWINS", "KXMLBWINS", "KXNHLWINS")

# Keywords that indicate a Polymarket "pick-the-winner" market. When the
# Kalshi side is a threshold/O-U, these are almost always different questions.
_DISCRETE_WINNER_KEYWORDS = (
    "nba finals",
    "nba champion",
    "nba championship",
    "super bowl",
    "world series",
    "stanley cup",
    "win the finals",
    "win the championship",
    "win the title",
)

# Keywords that indicate a political nomination/election market.
_NOMINATION_KEYWORDS = (
    "presidential nominee",
    "presidential nomination",
    "republican nominee",
    "democratic nominee",
    "nominee 2028",
    "nomination 2028",
    "win the 2028",
    "win the 2024",
    "next president",
)

# Keywords that indicate a cabinet/foreign-policy Kalshi market (very
# different from electoral markets).
_CABINET_KEYWORDS = (
    "secretary of state",
    "secretary of defense",
    "secretary of treasury",
    "visit mexico",
    "visit china",
    "visit russia",
    "state department",
)

# Keywords that indicate a "pick the leader" ranking market: who is #1
# on some leaderboard at a point in time. These are STRUCTURALLY
# incompatible with numeric-threshold contracts on the same underlying:
#
#   Kalshi: "Musk net worth > $600B?"   (threshold on a number)
#   Poly:   "Musk richest person?"      (rank #1 on a leaderboard)
#
# Both involve Musk, both move with Musk's wealth, but they resolve on
# totally different criteria and never converge.
_RANKING_KEYWORDS = (
    "richest person",
    "richest man",
    "richest woman",
    "wealthiest person",
    "wealthiest man",
    "wealthiest",
    "most valuable company",
    "biggest company",
    "largest company",
    "top team",
    "number one",
    "#1 ranked",
    "rank first",
    "ranked first",
)

# Keywords that indicate a discrete POLICY/EVENT market: will some
# one-time action happen? These are structurally incompatible with
# numeric threshold contracts on continuous statistics:
#
#   Kalshi: "Will Argentina monthly inflation be above 2.5%?"
#   Poly:   "Will Argentina dollarize by June 30, 2026?"
#
# Both mention Argentina, but one is a continuous statistic with a
# numeric threshold and the other is a one-time policy decision.
_POLICY_EVENT_KEYWORDS = (
    "dollarize",
    "legalize",
    "adopt as legal tender",
    "adopt bitcoin",
    "enact",
    "ratify",
    "impeach",
    "resign",
    "be fired",
    "declare war",
    "invade",
    "sign the",
    "sign into law",
    "pass the bill",
    "overturn",
    "repeal",
    "ceasefire",
)

# Kalshi AAA gas ticker prefixes where the LAST two letters encode a
# US state (California, Florida, New York, Texas). These state-specific
# contracts must not match Polymarket's national "gas hit $X" markets.
# Each entry is (ticker-prefix, state-name-as-it-appears-in-poly-titles).
_AAA_GAS_STATE_SUFFIXES = (
    ("CA", "california"),
    ("FL", "florida"),
    ("NY", "new york"),
    ("TX", "texas"),
)

# Map 3-letter month codes in Kalshi tickers -> month number.
_MONTH_CODES = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Full month names we look for inside Polymarket titles.
_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
}


def _extract_month_from_kalshi_ticker(ticker: str) -> int | None:
    """Return the month number encoded in a Kalshi ticker, or None.

    Looks for patterns like ``-26DEC-`` or ``-26JUN`` and returns 12 or
    6 respectively.
    """
    if not ticker:
        return None
    m = re.search(r"-\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)", ticker.upper())
    if m:
        return _MONTH_CODES[m.group(1)]
    return None


def _extract_month_from_text(text: str) -> int | None:
    """Return the first month number mentioned in a free-text title."""
    if not text:
        return None
    t = text.lower()
    for name, num in _MONTH_NAMES.items():
        # Word-boundary check so "march" doesn't match "marcher" etc.
        if re.search(rf"\b{name}\b", t):
            return num
    return None


def _kalshi_is_threshold_contract(ticker: str, title: str) -> bool:
    """Return True if this looks like a numeric-threshold Kalshi contract.

    Two signals:
      1. Ticker ends in ``-T<number>`` (Kalshi's convention for strike
         levels, e.g. KXMUSKNW-26APR30-T600, KXBTC-26APR-T100000).
      2. Title mentions a dollar/percent threshold phrase like
         "above $X", "more than $X", "exceed $X".
    """
    if not ticker and not title:
        return False
    if ticker and re.search(r"-T\d", ticker.upper()):
        return True
    t = (title or "").lower()
    threshold_phrases = (
        "above $", "above ", "more than $", "more than ",
        "exceed ", "exceeds ", "over $", "at least $",
        "at or above", "greater than ",
    )
    # Only count if the phrase is followed by a number
    for phrase in threshold_phrases:
        if phrase in t:
            # Very rough: if there's a digit within 20 chars of the phrase, call it
            idx = t.find(phrase)
            window = t[idx : idx + 40]
            if re.search(r"\d", window):
                return True
    return False


def _extract_year_from_kalshi_ticker(ticker: str) -> int | None:
    """Extract year from a Kalshi ticker.

    Kalshi tickers embed dates in patterns like:
      KXWTI-26APR08-T105.99    -> 2026
      KXFEDDECISION-27APR-H0   -> 2027
      KXSECSTATEVISIT-27-MEX   -> 2027
      KXPRESNOMD-28-AOC        -> 2028

    Returns the 4-digit year, or None if no date pattern found.
    """
    if not ticker:
        return None
    # Pattern 1: "-YY<MON>" e.g. -27APR-, -26APR08-
    m = re.search(r"-(\d{2})(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)", ticker.upper())
    if m:
        return 2000 + int(m.group(1))
    # Pattern 2: "-YY-" standalone two-digit year
    m = re.search(r"-(\d{2})-", ticker)
    if m:
        yr = int(m.group(1))
        # Reject obvious thresholds/numbers ("T35", "T40") — require year-like range
        if 20 <= yr <= 40:
            return 2000 + yr
    return None


def _extract_year_from_text(text: str) -> int | None:
    """Return the first 4-digit year mentioned in the text, or None."""
    if not text:
        return None
    m = re.search(r"\b(20\d{2})\b", text)
    if m:
        return int(m.group(1))
    return None


def _current_year() -> int:
    """Current year in UTC. Wrapped so tests can monkey-patch."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).year


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    t = text.lower()
    return any(n in t for n in needles)


def _detect_asset_class(ticker_u: str, title: str) -> str | None:
    """Return 'commodity', 'crypto', or None if asset class can't be
    confidently inferred from ticker prefix OR title tokens.

    Ticker prefix wins over title tokens when both produce a signal,
    because Kalshi tickers are authoritative and titles can contain
    stray numeric strings. Only returns 'commodity' or 'crypto' when
    the evidence is UNAMBIGUOUS — if both commodity AND crypto tokens
    appear in the title we return None (let other rules decide).
    """
    # Priority 1: Kalshi ticker prefix (most reliable). Longest-match-
    # wins is enforced implicitly by the prefix lists — a ticker like
    # KXWTIMAX hits both KXWTI and KXWTIMAX, but both map to commodity.
    for p in KALSHI_COMMODITY_PREFIXES:
        if ticker_u.startswith(p):
            return "commodity"
    for p in KALSHI_CRYPTO_PREFIXES:
        if ticker_u.startswith(p):
            return "crypto"
    # Priority 2: title token check. Uppercase the title for case-
    # insensitive match against our ALL-CAPS token tuples.
    title_u = title.upper()
    has_commodity = any(t in title_u for t in COMMODITY_ASSET_TOKENS)
    has_crypto = any(t in title_u for t in CRYPTO_ASSET_TOKENS)
    if has_commodity and not has_crypto:
        return "commodity"
    if has_crypto and not has_commodity:
        return "crypto"
    return None


def filter_active_match(match: dict) -> tuple[bool, str | None]:
    """Validate a single entry from active_matches.json.

    Returns (True, None) if the pair should be actively traded, or
    (False, reason) if it should be filtered out.

    The rules encode patterns discovered in live paper-trading data
    where systematically-losing pairs shared a structural mismatch
    that the semantic matcher missed.
    """
    ticker = (match.get("kalshi_ticker") or "").strip()
    k_title = (match.get("kalshi_title") or "").strip()
    p_title = (match.get("poly_title") or "").strip()
    similarity = match.get("similarity", 0.0) or 0.0

    # Required fields
    if not ticker:
        return False, "missing_kalshi_ticker"
    if not p_title:
        return False, "missing_poly_title"

    # --- Rule 4: similarity floor (cheap, check early) ---
    if similarity < MIN_ACTIVE_SIMILARITY:
        return False, f"low_similarity ({similarity:.2f} < {MIN_ACTIVE_SIMILARITY})"

    ticker_u = ticker.upper()
    p_title_l = p_title.lower()
    k_title_l = k_title.lower()

    # --- Rule 1: season-wins O/U vs champion/finals market ---
    if any(ticker_u.startswith(p) for p in _SEASON_WINS_TICKER_PREFIXES):
        if _has_any(p_title_l, _DISCRETE_WINNER_KEYWORDS):
            return False, "nba_season_wins_vs_champion"
        # Kalshi also phrases it as "win at least N"
        if "at least" in k_title_l and _has_any(p_title_l, ("finals", "champion", "championship")):
            return False, "nba_season_wins_vs_champion"

    # --- Rule 2: Fed decision year/month mismatch ---
    if ticker_u.startswith("KXFEDDECISION") or ticker_u.startswith("KXFED"):
        k_year = _extract_year_from_kalshi_ticker(ticker)
        p_year = _extract_year_from_text(p_title)
        if k_year is not None and p_year is not None and k_year != p_year:
            return False, f"fed_year_mismatch (kalshi={k_year}, poly={p_year})"
        # If Kalshi ticker has a year but the Poly title doesn't mention any
        # year at all AND says "in April" / "April meeting", Polymarket
        # defaults to the current Fed cycle (April 2026 in our data). A
        # 2027 Kalshi Fed contract matched against that is always wrong.
        if k_year is not None and k_year >= 2027 and p_year is None:
            # The Poly market is almost certainly the near-term Fed decision
            if "fed" in p_title_l and ("april" in p_title_l or "june" in p_title_l or "july" in p_title_l):
                return False, f"fed_year_mismatch (kalshi={k_year}, poly=implicit-2026)"
        # Month mismatch within the same year: KXFEDDECISION-26DEC vs
        # "Fed decision after the June meeting" — both 2026 but different
        # meetings, so the contracts don't overlap.
        k_month = _extract_month_from_kalshi_ticker(ticker)
        p_month = _extract_month_from_text(p_title)
        if k_month is not None and p_month is not None and k_month != p_month:
            return False, (
                f"fed_month_mismatch (kalshi_month={k_month}, "
                f"poly_month={p_month})"
            )

    # --- Rule 3: cross-topic cabinet vs nomination/election ---
    if _has_any(k_title_l, _CABINET_KEYWORDS) and _has_any(p_title_l, _NOMINATION_KEYWORDS):
        return False, "cabinet_vs_nomination"

    # --- Rule 3b: numeric threshold contract vs ranking/leader market ---
    # Kalshi "Musk net worth > $600B" vs Polymarket "Musk richest person?"
    # Both reference the same entity but resolve on different criteria.
    # Only reject when Kalshi is a threshold AND Polymarket is a ranking
    # question — symmetric threshold/threshold pairs (different strikes on
    # the same asset) are LEGITIMATE trading pairs and must not be rejected.
    if _has_any(p_title_l, _RANKING_KEYWORDS):
        if _kalshi_is_threshold_contract(ticker, k_title):
            return False, "threshold_vs_ranking"

    # --- Rule 3c: numeric threshold contract vs policy/event market ---
    # Kalshi "Argentina monthly inflation > 2.5%" vs Polymarket "Will
    # Argentina dollarize?" — both mention Argentina but one is a
    # continuous statistic and the other is a one-time policy action.
    # Same structural pattern as 3b, different vocabulary.
    if _has_any(p_title_l, _POLICY_EVENT_KEYWORDS):
        if _kalshi_is_threshold_contract(ticker, k_title):
            return False, "threshold_vs_policy_event"

    # --- Rule 3d: AAA retail gas geography/date mismatch ---
    # KXAAAGAS* is the Kalshi family for AAA-tracked US retail regular
    # gasoline ($/gallon). After the task #29 commodity fix, 205 of
    # these pairs were matched against Polymarket "Will gas hit $X by
    # April 30?" — same underlying commodity but structurally wrong:
    #
    #   a) Geography: KXAAAGAS*CA/FL/NY/TX are state averages;
    #      Polymarket gas markets are always national US. State and
    #      national averages have persistent divergences and never
    #      fully converge, so these are not arbitrage candidates.
    #
    #   b) Date: KXAAAGAS*MAX/MIN tickers resolve on Dec 31 (annual
    #      max/min), Polymarket resolves April 30. Same year, 8-month
    #      resolution gap → the Kalshi contract asks about the WHOLE
    #      year while Polymarket only covers Jan-April.
    #
    # Rule scope is intentionally narrow (KXAAAGAS prefix only) so it
    # can't false-positive on other commodity pairs. Same-month national
    # pairs (KXAAAGASW-26APR13 vs Poly April) still pass.
    if ticker_u.startswith("KXAAAGAS"):
        # (a) Geography check: state-specific ticker vs non-state Poly title
        for suffix, state_name in _AAA_GAS_STATE_SUFFIXES:
            # Match against common families: MAX<ST>, MIN<ST>, M<ST>, etc.
            if re.search(rf"KXAAAGAS[A-Z]*{suffix}(?:-|$)", ticker_u):
                if state_name not in p_title_l:
                    return False, f"aaa_gas_state_vs_national ({suffix})"
                break

        # (b) Month check: Kalshi ticker month vs Poly title month.
        # Only fires when BOTH sides produce a month — if Polymarket
        # title has no month word, we fall through and let similarity
        # be the arbiter.
        k_month = _extract_month_from_kalshi_ticker(ticker)
        p_month = _extract_month_from_text(p_title)
        if k_month is not None and p_month is not None and k_month != p_month:
            return False, (
                f"aaa_gas_month_mismatch (kalshi={k_month}, poly={p_month})"
            )

    # Extract year from the Kalshi ticker itself (authoritative) and
    # from both titles (supplementary).
    k_year_ticker = _extract_year_from_kalshi_ticker(ticker)
    k_year_title = _extract_year_from_text(k_title)
    p_year_title = _extract_year_from_text(p_title)

    # --- Rule: stale ticker (contract year already in the past) ---
    # Kalshi occasionally leaves past-dated markets as status=open; trading
    # a contract whose resolution window is behind us is meaningless. This
    # check uses the ticker-encoded year because the title may omit it.
    if k_year_ticker is not None and k_year_ticker < _current_year():
        return False, f"stale_ticker (kalshi ticker year={k_year_ticker}, now>={_current_year()})"

    # --- Rule: ticker-year vs title-year mismatch ---
    # Catches the Brazil inflation case: ticker '-25DEC-' (2025) but Poly
    # title says '2026'. The Fed-specific rule above only fires for
    # KXFED* tickers; this is the general case.
    if k_year_ticker is not None and p_year_title is not None:
        if k_year_ticker != p_year_title:
            return False, (
                f"ticker_year_mismatch (kalshi_ticker={k_year_ticker}, "
                f"poly_title={p_year_title})"
            )

    # General year-mismatch fallback for cases where both TITLES mention a
    # 4-digit year and they differ by more than one year. The >= 2 bound
    # is a safety margin so we don't reject questions that happen to
    # mention two adjacent years in passing (e.g. "2025 data, resolves
    # 2026"). The authoritative ticker-based rule above handles the
    # tighter cases.
    if k_year_title is not None and p_year_title is not None:
        if abs(k_year_title - p_year_title) >= 2:
            return False, f"year_mismatch ({k_year_title} vs {p_year_title})"

    # --- Rule 10: asset-class consistency (cross-asset numeric coincidence) ---
    # Rejects matches where one side is unambiguously a commodity contract
    # and the other is unambiguously a crypto contract (or vice versa).
    # Canonical bug: KXWTIMAX-26DEC31-T130 ("WTI $130") matched to Bitcoin-
    # $130K because both sides share the numeric "130". Semantic similarity
    # 0.707 was above the floor, but the contracts resolve on unrelated
    # assets. Only fires when BOTH sides produce a confident asset-class
    # signal — ambiguous cases pass through untouched.
    k_class = _detect_asset_class(ticker_u, k_title)
    p_class = _detect_asset_class("", p_title)  # Polymarket has no Kalshi-style ticker prefix
    if k_class is not None and p_class is not None and k_class != p_class:
        return False, f"asset_class_mismatch (kalshi={k_class}, poly={p_class})"

    return True, None


def filter_active_matches(matches: list[dict]) -> tuple[list[dict], dict]:
    """Apply filter_active_match to a list, return (passed, stats).

    Args:
        matches: list of dicts in active_matches.json schema.

    Returns:
        (passed, stats) where passed is the filtered list and stats is
        {total, passed, rejected, reasons: {reason: count}}.
    """
    passed: list[dict] = []
    reasons: dict[str, int] = {}
    rejected = 0

    for m in matches:
        ok, reason = filter_active_match(m)
        if ok:
            passed.append(m)
        else:
            rejected += 1
            key = (reason or "unknown").split(" ")[0]  # dedupe by rule name
            reasons[key] = reasons.get(key, 0) + 1

    stats = {
        "total": len(matches),
        "passed": len(passed),
        "rejected": rejected,
        "reasons": reasons,
    }
    logger.info(
        "filter_active_matches: %d -> %d (rejected %d; reasons=%s)",
        stats["total"], stats["passed"], stats["rejected"], reasons,
    )
    return passed, stats


def filter_candidates(candidates: list[dict]) -> list[dict]:
    """Apply all quality filters to scored candidates.

    Filters:
    1. Minimum confidence score
    2. Resolution date proximity (within MAX_RESOLUTION_GAP_DAYS)
    3. Contract direction compatibility
    4. Threshold number compatibility

    Returns filtered list with rejection reasons logged.
    """
    filtered = []
    stats = {
        "total": len(candidates),
        "low_confidence": 0,
        "temporal_mismatch": 0,
        "direction_mismatch": 0,
        "threshold_mismatch": 0,
        "passed": 0,
    }

    for c in candidates:
        # 1. Confidence threshold
        if c.get("confidence_score", 0) < MIN_CONFIDENCE:
            stats["low_confidence"] += 1
            continue

        # 2. Resolution date proximity
        k_date = parse_date(c.get("kalshi_resolution_date", ""))
        p_date = parse_date(c.get("polymarket_resolution_date", ""))

        if k_date and p_date:
            gap = abs((k_date - p_date).days)
            if gap > MAX_RESOLUTION_GAP_DAYS:
                stats["temporal_mismatch"] += 1
                continue

        # 3. Direction compatibility
        k_dir = extract_direction(c.get("kalshi_question", ""))
        p_dir = extract_direction(c.get("polymarket_question", ""))

        if not directions_compatible(k_dir, p_dir):
            stats["direction_mismatch"] += 1
            continue

        # 4. Threshold compatibility
        k_thresh = extract_threshold(c.get("kalshi_question", ""))
        p_thresh = extract_threshold(c.get("polymarket_question", ""))

        if not thresholds_compatible(k_thresh, p_thresh):
            stats["threshold_mismatch"] += 1
            continue

        # Add extracted metadata for curation display
        c["kalshi_direction"] = k_dir
        c["polymarket_direction"] = p_dir
        c["kalshi_threshold"] = k_thresh
        c["polymarket_threshold"] = p_thresh
        c["resolution_gap_days"] = abs((k_date - p_date).days) if k_date and p_date else None

        stats["passed"] += 1
        filtered.append(c)

    logger.info(
        f"Quality filter: {stats['total']} candidates -> {stats['passed']} passed "
        f"(rejected: {stats['low_confidence']} low confidence, "
        f"{stats['temporal_mismatch']} temporal mismatch, "
        f"{stats['direction_mismatch']} direction mismatch, "
        f"{stats['threshold_mismatch']} threshold mismatch)"
    )
    return filtered
