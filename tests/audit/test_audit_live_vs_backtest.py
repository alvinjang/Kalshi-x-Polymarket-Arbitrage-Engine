"""Tier 6 audit fixture test.

Proves Tier 6's Cohen's h calculation reflects a large effect for the
canonical live-vs-backtest oil-WR gap (0.36 vs 0.765). This is the
sentinel test: if cohens_h returns the wrong magnitude or sign, every
Tier 6 verdict is suspect.
"""
from __future__ import annotations

import math

from experiments.audit.audit_live_vs_backtest import cohens_h


def test_cohens_h_large_for_canonical_oil_gap():
    # Live WR ≈ 0.36, backtest WR ≈ 0.765 — should produce LARGE effect (|h| > 0.5).
    # Expected sign is negative because live (p1) is below backtest (p2).
    h = cohens_h(0.36, 0.765)
    assert h < -0.5, f"expected large negative effect (h < -0.5), got {h:.3f}"


def test_cohens_h_zero_for_equal_proportions():
    # h(p, p) = 0 by construction.
    assert abs(cohens_h(0.5, 0.5)) < 1e-9
    assert abs(cohens_h(0.36, 0.36)) < 1e-9


def test_cohens_h_symmetric_around_zero():
    # h(p1, p2) = -h(p2, p1).
    assert abs(cohens_h(0.3, 0.7) + cohens_h(0.7, 0.3)) < 1e-9
    assert abs(cohens_h(0.36, 0.765) + cohens_h(0.765, 0.36)) < 1e-9


def test_cohens_h_matches_formula_definition():
    # h = 2 * (asin(sqrt(p1)) - asin(sqrt(p2)))
    p1, p2 = 0.36, 0.765
    expected = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))
    assert abs(cohens_h(p1, p2) - expected) < 1e-12
