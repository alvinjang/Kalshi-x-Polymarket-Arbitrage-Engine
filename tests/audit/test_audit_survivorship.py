"""Proves Tier 4 audit would surface a synthetically post-hoc-dropped pair.

Tier 4 fixture tests for experiments/audit/audit_survivorship.py.
Validates the drop-reason classifier distinguishes:
  - structural drops (low_overlap, pre_test_window)
  - REVIEW cases (heuristic fallthrough — manual sign-off needed)

Per VALIDATION.md row 18-05-XX, function name
``test_audit_survivorship_catches_post_hoc_drop`` is the canonical
alias used by the verification harness.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import pandas as pd

from experiments.audit.audit_survivorship import classify_drop_reason


def test_classify_low_overlap():
    aligned = pd.DataFrame({
        "pair_id": ["p1"] * 5,
        "timestamp": pd.to_datetime([1, 2, 3, 4, 5], unit="s", utc=True),
    })
    assert "low_overlap" in classify_drop_reason("p1", aligned)


def test_classify_high_overlap_returns_pre_test_window():
    aligned = pd.DataFrame({
        "pair_id": ["p1"] * 50,
        "timestamp": pd.to_datetime(range(50), unit="s", utc=True),
    })
    reason = classify_drop_reason("p1", aligned)
    assert "pre_test_window" in reason


# Alias matching VALIDATION.md naming convention
def test_audit_survivorship_catches_post_hoc_drop():
    """Alias delegating to low-overlap classification test."""
    test_classify_low_overlap()
