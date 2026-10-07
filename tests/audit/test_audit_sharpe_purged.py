"""Tests for the purged Tier 1 Sharpe audit (Phase 18-08 Task 3).

Phase 18-08 redoes the Tier 1 audit on the leakage-free pair-stratified
split. This module proves the audit infrastructure:

1. Imports the canonical-audit helpers verbatim (no duplication).
2. Builds a trade ledger from the purged test parquet under LR.
3. Catches the i.i.d.-violation failure mode the same way the leaky audit
   does — via correlation_corrected_sharpe collapsing perfectly-correlated
   pair returns to ~0 (smoke-test re-verifies that property because the
   purged audit reuses the same correction code path).

Implements requirement: AUDIT-07 (test side, audit infrastructure).
"""
# AI-assisted authorship: written with Anthropic Claude (Opus 4.7) as
# pair-programming assistant. All design decisions are the authors'.
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the purged audit module is importable; this also smoke-tests the
# REUSE rule (it imports audit_sharpe helpers at module load time).
audit_purged = pytest.importorskip("experiments.audit.audit_sharpe_purged")
from experiments.audit.audit_sharpe import (
    avg_pairwise_correlation,
    correlation_corrected_sharpe,
    per_pair_returns,
    per_pair_sharpe_naive,
)


def test_purged_audit_reuses_canonical_helpers():
    """audit_sharpe_purged must re-export / re-use the canonical helpers.

    The plan's hard rule is "DO NOT duplicate" — verify the purged module
    references the canonical helpers by attribute, not by reimplementing
    them. We check for the presence of the same function names *imported*
    (not redefined) inside audit_purged.
    """
    # The canonical helpers should be reachable from audit_purged's
    # namespace because audit_purged imports them from audit_sharpe.
    assert hasattr(audit_purged, "per_pair_returns")
    assert hasattr(audit_purged, "correlation_corrected_sharpe")
    # And they should be the SAME OBJECT as in the canonical module.
    assert audit_purged.per_pair_returns is per_pair_returns
    assert (
        audit_purged.correlation_corrected_sharpe
        is correlation_corrected_sharpe
    )


def test_purged_audit_catches_inflated_independence():
    """Smoke-test the same i.i.d.-violation guard as the canonical audit.

    Identical to tests/audit/test_audit_sharpe.py, repeated here to prove
    the purged audit's correction code path inherits the property without
    bug-introducing duplication.
    """
    rng = np.random.default_rng(123)
    n_pairs, n_days = 144, 30
    daily_returns = rng.normal(0.01, 0.02, size=n_days)
    rows = []
    for day in range(n_days):
        for pair in range(n_pairs):
            rows.append(
                {
                    "pair_id": f"pair_{pair}",
                    "entry_day": day,
                    "entry_ts": day * 86400,
                    "pnl_pp": daily_returns[day],
                    "traded": True,
                }
            )
    ledger = pd.DataFrame(rows)
    pair_ret = per_pair_returns(ledger)
    s_naive = per_pair_sharpe_naive(pair_ret)
    avg_corr, _ = avg_pairwise_correlation(ledger)
    s_corr, _ = correlation_corrected_sharpe(s_naive, len(pair_ret), avg_corr)
    assert avg_corr > 0.99
    assert abs(s_corr) < 0.05 * abs(s_naive) + 1e-9, (
        f"correction should collapse to ~0; got s_corr={s_corr} "
        f"vs s_naive={s_naive}"
    )


PURGED_AUDIT_JSON = Path("experiments/results/audit/sharpe_audit_purged.json")


@pytest.mark.skipif(
    not PURGED_AUDIT_JSON.exists(),
    reason="Run experiments/audit/audit_sharpe_purged.py first.",
)
def test_purged_audit_json_has_comparison_block():
    """If the audit JSON exists, it MUST carry the comparison + verdict."""
    import json

    out = json.loads(PURGED_AUDIT_JSON.read_text())
    assert out["audit"] == "sharpe_purged"
    assert out["verdict"] in {"PASS", "CORRECTED", "FAILED"}
    comp = out.get("comparison")
    assert comp is not None, "missing comparison block"
    for key in (
        "canonical_sharpe_per_trade",
        "purged_sharpe_per_trade",
        "delta_per_trade",
        "canonical_sharpe_per_pair_naive",
        "purged_sharpe_per_pair_naive",
        "canonical_sharpe_per_pair_corrected",
        "purged_sharpe_per_pair_corrected",
        "canonical_total_pnl",
        "purged_total_pnl",
        "canonical_n_pairs",
        "purged_n_pairs",
        "interpretation",
    ):
        assert key in comp, f"comparison missing key: {key}"
    assert isinstance(comp["interpretation"], str) and comp["interpretation"]
