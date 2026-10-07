"""Proves Tier 1 audit catches the i.i.d.-violation failure mode.

Build a synthetic ledger where all 144 pairs realize identical returns
(perfect correlation, avg_corr = 1.0). The naive per-pair Sharpe is
positive, but the correlation correction must collapse it to ~0.

We assert: pp_sharpe_corrected < 0.05 * pp_sharpe_naive when avg_corr ≈ 1.

This is the audit-correctness test for AUDIT-01: it proves the audit
infrastructure built in experiments/audit/audit_sharpe.py would actually
catch the failure mode it was designed to catch (the inflated-by-pair-
correlation Sharpe number that motivates the entire Tier 1 audit). Without
this test, a verdict=PASS reading from sharpe_audit.json could mean either
"no inflation present" or "inflation present but audit silently failed".

Function name `test_audit_sharpe_catches_inflated_independence` matches
the VALIDATION.md naming convention; the alias
`test_correlation_correction_collapses_perfectly_correlated_pairs` from
RESEARCH.md is preserved as a comment for cross-reference.
"""
import numpy as np
import pandas as pd
from experiments.audit.audit_sharpe import (
    avg_pairwise_correlation, correlation_corrected_sharpe,
    per_pair_returns, per_pair_sharpe_naive,
)


def test_audit_sharpe_catches_inflated_independence():
    # Alias (RESEARCH.md): test_correlation_correction_collapses_perfectly_correlated_pairs
    rng = np.random.default_rng(123)
    n_pairs, n_days = 144, 30
    # Synthetic ledger: every pair earns the SAME daily return -> avg_corr = 1.0
    daily_returns = rng.normal(0.01, 0.02, size=n_days)
    rows = []
    for day in range(n_days):
        for pair in range(n_pairs):
            rows.append({
                "pair_id": f"pair_{pair}",
                "entry_day": day,
                "entry_ts": day * 86400,
                "pnl_pp": daily_returns[day],  # identical across pairs
                "traded": True,
            })
    ledger = pd.DataFrame(rows)
    pair_ret = per_pair_returns(ledger)
    s_naive = per_pair_sharpe_naive(pair_ret)
    avg_corr, _ = avg_pairwise_correlation(ledger)
    s_corr, n_eff = correlation_corrected_sharpe(s_naive, len(pair_ret), avg_corr)
    assert avg_corr > 0.99, f"synthetic data should have ~perfect corr, got {avg_corr}"
    assert n_eff < 5, f"effective N should collapse to ~1, got {n_eff}"
    assert abs(s_corr) < 0.05 * abs(s_naive) + 1e-6, (
        f"correlation correction did NOT collapse Sharpe: "
        f"naive={s_naive:.3f}, corrected={s_corr:.3f}"
    )
