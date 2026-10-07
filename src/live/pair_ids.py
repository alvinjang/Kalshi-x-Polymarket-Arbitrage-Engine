"""Content-addressed pair_id generation for live matched markets.

**Why this module exists**

The old live-trading code used ``f"live_{i:04d}"`` where ``i`` was the
enumerate index over ``active_matches.json``. That is broken for three
reasons discovered on 2026-04-11:

  1. ``collector.py::_load_live_pairs`` and ``strategy.py`` enumerated
     DIFFERENT lists (unfiltered vs filtered by quality_filter).
     Same pair_id string meant different pairs in the two modules.

  2. Every time the quality filter changes, the filtered-list indices
     shift and every open position silently starts pointing at a
     different underlying pair.

  3. ``data/live/pair_mapping.json`` was a third static snapshot from
     April 9 with its own live_NNNN numbering that matched neither.

This module replaces the index-based scheme with a content-addressed
one, matching the format ``train.parquet`` already uses:

    kxbtc26feb0617b-0x0356fe1e
    |              | |        |
    |              | └─ poly_id prefix (first 10 hex chars)
    |              └─ hyphen separator
    └─ kalshi_ticker, lowercased, hyphens removed

With a content-addressed pair_id, the same (kalshi_ticker, poly_id)
always produces the same string across all modules and all runs
regardless of discovery or filter state.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

# Length of the poly_id prefix we keep in the pair_id. 10 chars of hex
# = 40 bits of entropy, more than enough to disambiguate the few
# thousand pairs Polymarket currently offers. Matches train.parquet.
_POLY_ID_PREFIX_LEN = 10


def make_pair_id(kalshi_ticker: str, poly_id: str) -> str:
    """Return a stable content-addressed pair_id.

    Args:
        kalshi_ticker: Kalshi market ticker (e.g. ``"KXWTI-26APR08-T107.99"``).
            Case and hyphens are normalized out so ``"KXWTI-26APR08"``
            and ``"kxwti-26apr08"`` produce the same id.
        poly_id: Polymarket identifier — ideally a 0x-prefixed hex
            conditionId, but legacy numeric Polymarket "id" values are
            also accepted. The first 10 characters are retained.

    Returns:
        A string like ``"kxwti26apr08t10799-0x43d5953d"`` that uniquely
        and stably identifies the pair across runs. Returns the empty
        string if either input is blank — callers should skip empty
        pair_ids rather than passing them around.

    Examples:
        >>> make_pair_id("KXWTI-26APR08-T107.99", "0x43d5953daec805127ff71b")
        'kxwti26apr08t10799-0x43d5953d'
        >>> make_pair_id("KXWTI-26APR08", "1712297")  # legacy numeric
        'kxwti26apr08-1712297'
        >>> make_pair_id("", "foo")
        ''
        >>> make_pair_id("KXFOO", "")
        ''
    """
    kalshi_norm = _normalize_kalshi_ticker(kalshi_ticker)
    poly_norm = _normalize_poly_id(poly_id)
    if not kalshi_norm or not poly_norm:
        return ""
    return f"{kalshi_norm}-{poly_norm}"


def _normalize_kalshi_ticker(ticker: str) -> str:
    """Lowercase and strip hyphens+periods from a Kalshi ticker.

    ``"KXWTI-26APR08-T107.99"`` → ``"kxwti26apr08t10799"``.

    The period in ``T107.99`` is stripped because parquet column names
    and some downstream code can trip over punctuation; the integer
    part alone is unique per market.
    """
    if not ticker:
        return ""
    return (
        ticker.strip()
        .lower()
        .replace("-", "")
        .replace(".", "")
        .replace(" ", "")
    )


def _normalize_poly_id(poly_id: str | int) -> str:
    """Truncate a Polymarket identifier to its content-address prefix.

    Hex conditionIds (0x-prefixed) keep the first 10 chars including
    ``0x``. Legacy numeric ids are returned as-is since they're
    already short and stable.
    """
    if poly_id is None:
        return ""
    s = str(poly_id).strip()
    if not s:
        return ""
    if s.lower().startswith("0x") and len(s) >= _POLY_ID_PREFIX_LEN:
        return s[:_POLY_ID_PREFIX_LEN].lower()
    return s
