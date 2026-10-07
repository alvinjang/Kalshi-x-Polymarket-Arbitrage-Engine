"""Regression tests for Rule 10 — asset-class consistency (COM-02).

The canonical bug this rule exists to prevent:

    Kalshi  : KXWTIMAX-26DEC31-T130  ("Will WTI crude oil reach $130 by Dec 31, 2026?")
    Poly    : 0x885a6abe...a859       ("Will Bitcoin reach $130,000 by December 31, 2026?")
    Semantic similarity: 0.707 (above MIN_ACTIVE_SIMILARITY=0.70 floor)

Both sides share the numeric string "130" and the year "2026", so the
semantic matcher accepts and no pre-Rule-10 quality filter catches it.
The contracts resolve on unrelated assets — trading the pair is
structurally meaningless.

Rule 10 must:
  1. Reject numerically-coincident cross-asset strikes (commodity vs crypto,
     either direction).
  2. NOT reject legitimate symmetric matches (commodity/commodity or
     crypto/crypto) even when the strikes are numerically identical.

These tests are the RED commit — they are written BEFORE the fix and
must FAIL on pre-fix code.
"""
# AI-assisted authorship: written with Anthropic Claude (Opus 4.7) as pair-programming assistant. All design decisions and interpretations are the authors'.

from src.matching.quality_filter import filter_active_match


# Canonical false-match fixture — the exact real-world pair that motivated
# this rule. Hex id is pinned so grep-based audits can find it.
BAD_MATCH = {
    "kalshi_ticker": "KXWTIMAX-26DEC31-T130",
    "kalshi_title": "Will WTI crude oil reach $130 by December 31, 2026?",
    "poly_id": "0x885a6abefad122348b4fbd503473d7fd1f9035d0438cf988a7591620f316a859",
    "poly_title": "Will Bitcoin reach $130,000 by December 31, 2026?",
    "similarity": 0.707,
}


def test_rule_10_rejects_kxwtimax_vs_bitcoin():
    """The canonical false match MUST be rejected with an asset-class reason."""
    accepted, reason = filter_active_match(BAD_MATCH)
    assert accepted is False, (
        f"KXWTIMAX-26DEC31-T130 vs Bitcoin-$130K must be rejected; "
        f"got accepted=True (reason={reason})"
    )
    assert "asset_class_mismatch" in (reason or ""), (
        f"Expected rejection reason containing 'asset_class_mismatch', "
        f"got: {reason!r}"
    )


def test_rule_10_accepts_symmetric_wti_pair():
    """Symmetric WTI/commodity pair must pass (no false positive on legitimate
    cross-platform commodity arbitrage)."""
    match = {
        "kalshi_ticker": "KXWTIMAX-26DEC31-T130",
        "kalshi_title": "Will WTI crude oil reach $130 by December 31, 2026?",
        "poly_id": "0xdeadbeef_symmetric_wti",
        "poly_title": "Will WTI crude oil reach $130 by December 31, 2026?",
        "similarity": 0.95,
    }
    accepted, reason = filter_active_match(match)
    assert accepted is True, (
        f"Symmetric WTI-$130 vs WTI-$130 must be accepted; "
        f"got rejected with reason={reason}"
    )
    assert reason is None, f"Expected reason=None for symmetric accept, got {reason!r}"


def test_rule_10_accepts_symmetric_bitcoin_pair():
    """Symmetric Bitcoin/crypto pair must pass (no false positive on legitimate
    cross-platform crypto arbitrage)."""
    match = {
        "kalshi_ticker": "KXBTC-26DEC31-T130000",
        "kalshi_title": "Will Bitcoin reach $130,000 by December 31, 2026?",
        "poly_id": "0xdeadbeef_symmetric_btc",
        "poly_title": "Will Bitcoin reach $130,000 by December 31, 2026?",
        "similarity": 0.97,
    }
    accepted, reason = filter_active_match(match)
    assert accepted is True, (
        f"Symmetric Bitcoin-$130K vs Bitcoin-$130K must be accepted; "
        f"got rejected with reason={reason}"
    )
    assert reason is None, f"Expected reason=None for symmetric accept, got {reason!r}"


def test_rule_10_rejects_crypto_kalshi_commodity_poly():
    """Inverse direction — Kalshi crypto ticker vs Polymarket oil market —
    must also be rejected on asset-class mismatch."""
    match = {
        "kalshi_ticker": "KXBTCD-26DEC31-T130000",
        "kalshi_title": "Will Bitcoin reach $130,000 by December 31, 2026?",
        "poly_id": "0xfeedfacecafebabe_inverse",
        "poly_title": "Will WTI crude oil reach $130 by December 31, 2026?",
        "similarity": 0.71,
    }
    accepted, reason = filter_active_match(match)
    assert accepted is False, (
        f"Kalshi crypto (KXBTCD) vs Polymarket oil must be rejected; "
        f"got accepted=True (reason={reason})"
    )
    assert "asset_class_mismatch" in (reason or ""), (
        f"Expected rejection reason containing 'asset_class_mismatch', "
        f"got: {reason!r}"
    )
