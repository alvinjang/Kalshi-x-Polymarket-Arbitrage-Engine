"""Asset-class category derivation from Kalshi tickers and pair IDs.

Task #25: add ``category`` as a training feature so tree models can
learn category-conditional rules (e.g. "narrow spread aggressively when
category==oil AND time_to_resolution < 48h") and linear models get
category-specific intercepts without polluting the shared weights.

Directly answers the central CLAUDE.md research question at a new axis:
*does a single unified model outperform category-conditional models,
and does the delta grow with data volume?*

Pure functions, no I/O — tested independently of the training harness.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import re

# Ordered list of (regex_or_prefix, category) rules. The FIRST match wins,
# so more specific prefixes must come before less specific ones. All
# ticker input is upper-cased before matching. Anchored to the start of
# the ticker string so "KXBTCD..." matches "KXBTC" and "KXBTCD".
#
# Keeping this as tuples (not a dict) preserves the required ordering
# for prefixes where one is a substring of another (e.g. KXBTC vs KXBTCD:
# both start with KXBTC, so the order only matters for disambiguation
# between categories, not between variants of the same category).
_RULES: tuple[tuple[str, str], ...] = (
    # --- Crypto (BTC/ETH/BNB/DOGE/XRP/HYPE and all variants) ---
    # Match order: more specific (KXBTCMAXY, KXBTCMAXMON, KXBTCMAX, KXBTCD, KXBTC)
    ("KXBTCMAXY", "crypto"),
    ("KXBTCMAXMON", "crypto"),
    ("KXBTCMAX", "crypto"),
    ("KXBTCD", "crypto"),
    ("KXBTC", "crypto"),
    ("KXETHD", "crypto"),
    ("KXETH", "crypto"),
    ("KXBNBD", "crypto"),
    ("KXBNB", "crypto"),
    ("KXDOGED", "crypto"),
    ("KXDOGE", "crypto"),
    ("KXXRPMAXMON", "crypto"),
    ("KXXRP", "crypto"),
    ("KXHYPE", "crypto"),

    # --- Oil (WTI-family + Brent + bilateral oil) ---
    # Phase 15 (2026-04-23): Extended with Brent family + remaining WTI
    # variants per 15-01-DIAGNOSTIC.md H5 PARTIAL. Without these entries,
    # newly-discovered Brent/WTI variants (once Commodities category is
    # scanned) would be classified as "other" and get the 3x threshold
    # penalty in src/live/strategy.py line 437.
    # Order matters: more-specific prefixes come before less-specific.
    ("KXBRENTMON", "oil"),
    ("KXBRENTW", "oil"),
    ("KXBRENTD", "oil"),
    ("KXBRENT", "oil"),
    ("KXWTIMAX", "oil"),
    ("KXWTIMINM", "oil"),
    ("KXWTIMIN", "oil"),
    ("KXWTIW", "oil"),
    ("KXWTIH", "oil"),
    ("KXWTIEU", "oil"),
    ("KXWTIE", "oil"),
    ("KXWTID", "oil"),
    ("KXWTI", "oil"),
    ("KXCRUDE", "oil"),
    ("KXDIESEL", "oil"),
    ("KXHEATINGOIL", "oil"),
    ("KXGASOLINE", "oil"),
    ("KXMEXCUBOIL", "oil"),

    # --- Non-oil commodities ---
    ("KXSILVERW", "commodities"),
    ("KXSILVERMON", "commodities"),
    ("KXSILVER", "commodities"),
    ("KXNATGASW", "commodities"),
    ("KXNATGAS", "commodities"),
    ("KXWHEATMON", "commodities"),
    ("KXWHEAT", "commodities"),
    ("KXCORNW", "commodities"),
    ("KXCORN", "commodities"),
    ("KXCOCOAMON", "commodities"),
    ("KXCOCOA", "commodities"),
    ("KXNICKELMON", "commodities"),
    ("KXNICKEL", "commodities"),
    ("KXLCATTLEW", "commodities"),
    ("KXLCATTLE", "commodities"),
    ("KXB200MON", "commodities"),

    # --- Inflation (US + foreign) ---
    ("KXCPICOREYOY", "inflation"),
    ("KXCPICORE", "inflation"),
    ("KXCPIYOY", "inflation"),
    ("KXCPI", "inflation"),
    ("KXSHELTERCPI", "inflation"),
    ("KXUKCPIYOY", "inflation"),
    ("KXCACPIYOY", "inflation"),
    ("KXBRAZILINF", "inflation"),
    ("KXARMOMINF", "inflation"),
    ("KXJPMOMINF", "inflation"),

    # --- GDP ---
    ("KXEZGDPQOQF", "gdp"),
    ("KXFRGDPQOQP", "gdp"),
    ("KXESGDPYOYF", "gdp"),
    ("KXDEGDPYOYF", "gdp"),
    ("KXITGDPQOQA", "gdp"),
    ("KXGDP", "gdp"),

    # --- Fed decisions + Treasury rates + foreign central banks ---
    ("KXFEDDECISION", "fed_rates"),
    ("KXFED", "fed_rates"),
    ("KX3MTBILL", "fed_rates"),
    ("KXCBDECISION", "fed_rates"),  # Chinese central bank decisions

    # --- Employment / labor ---
    ("KXLAYOFFSYINFO", "employment"),
    ("KXJOBLESSCLAIMS", "employment"),
    ("KXU3", "employment"),
    ("KXUE", "employment"),

    # --- Politics: elections and nominations ---
    ("KXPRESNOMD", "politics_election"),
    ("KXPRESNOMR", "politics_election"),
    ("KXPRESNOMFEDCHA", "politics_election"),
    ("KXTEXASGOP", "politics_election"),
    ("KXPARIS1RWINNER", "politics_election"),
    ("KXTXSENDPRIMARY", "politics_election"),  # TX Senate D primary
    ("KXTXSENRPRIMARY", "politics_election"),  # TX Senate R primary
    ("KXTHAILANDHOUSE", "politics_election"),

    # --- Politics: policy events / cabinet / geopolitics ---
    ("KXTRUMPMEET", "politics_policy"),
    ("KXRECOGPALESTINE", "politics_policy"),
    ("KXLEADERSOUT", "politics_policy"),
    ("KXSECSTATEVISIT", "politics_policy"),
    ("KXGOVTSHUTLENGTH", "politics_policy"),
    ("KXSOCDEMSEATS", "politics_policy"),
    ("KXDENMARK2ND", "politics_policy"),
    ("KXDENMARK3RD", "politics_policy"),
    ("KXDENMARKGAIN", "politics_policy"),
    ("KXDENMARK", "politics_policy"),
    ("KXJAPANHOUSE", "politics_policy"),
    ("KXLDPSEATS", "politics_policy"),     # Japan Liberal Democratic Party seat count
    ("KXNEPALHOUSE", "politics_policy"),
    ("KXECONSTATU3", "politics_policy"),   # econ policy status
    ("KXTRUMPKIDSSOTU", "politics_policy"),  # Trump kids at SOTU
    ("KXVISIT", "politics_policy"),

    # --- Sports ---
    ("KXNBAWINS", "sports"),
    ("KXNBAGAME", "sports"),
    ("KXNBATOTAL", "sports"),
    ("KXNBA", "sports"),
    ("KXNFLWINS", "sports"),
    ("KXNFL", "sports"),
    ("KXMLBWINS", "sports"),
    ("KXMLB", "sports"),
    ("KXNHLWINS", "sports"),
    ("KXNHL", "sports"),
    ("KXATP", "sports"),
    ("KXPGATOP", "sports"),
    ("KXPGA", "sports"),

    # --- FX ---
    ("KXINXY", "fx"),
    ("KXUSDIRR", "fx"),
    ("KXUSDARS", "fx"),
    ("KXUSDTRY", "fx"),

    # --- Housing / gas / wealth ---
    ("KXSEAHOMEVAL", "housing"),
    ("KXAAAGASMAX", "gas_prices"),
    ("KXAAAGASMIN", "gas_prices"),
    ("KXAAAGAS", "gas_prices"),
    ("KXTRILLIONAIRE", "wealth"),
)

# Canonical list of all category labels. Useful for one-hot encoding
# and for validating the output of derivation.
CATEGORIES: tuple[str, ...] = (
    "crypto",
    "oil",
    "commodities",
    "inflation",
    "gdp",
    "fed_rates",
    "employment",
    "politics_election",
    "politics_policy",
    "sports",
    "fx",
    "housing",
    "gas_prices",
    "wealth",
    "other",
)


def derive_category_from_ticker(ticker: str | None) -> str:
    """Return the asset-class category for a Kalshi ticker.

    Args:
        ticker: Kalshi ticker string (e.g. "KXWTI-26APR08-T105.99"),
            case-insensitive. May be empty or None.

    Returns:
        One of the strings in ``CATEGORIES``. Falls back to ``"other"``
        if no rule matches.
    """
    if not ticker:
        return "other"
    t = ticker.upper().strip()
    for prefix, category in _RULES:
        if t.startswith(prefix):
            return category
    return "other"


# pair_id format in train.parquet: lowercase truncated ticker + '-' +
# '0x' + hex poly suffix. Example: "kxbtc26jan0217b-0x511f8c84".
#
# The live collector format is different: "live_0042" — no ticker info,
# lookup has to go through active_matches.json at call time.
_LIVE_PAIR_RE = re.compile(r"^live_\d+$")


def derive_category_from_pair_id(pair_id: str | None) -> str:
    """Return the asset-class category for a training-set pair_id.

    Training pair_ids encode the Kalshi ticker as a lowercase prefix
    (``kxbtc26jan0217b-0x511f8c84`` -> KXBTC -> crypto). Live pair_ids
    (``live_0042``) do not encode the ticker and return ``"other"`` —
    callers should look up via ``active_matches.json`` in that case.
    """
    if not pair_id:
        return "other"
    if _LIVE_PAIR_RE.match(pair_id):
        return "other"
    # Take the chunk before the '-0x' hex suffix, upper-case it, and
    # treat it as a ticker prefix.
    prefix = pair_id.split("-", 1)[0]
    return derive_category_from_ticker(prefix)
