"""Shared fixtures for Phase 18 audit fixture tests.

Each fixture corresponds to one audit-target failure mode:
- perfectly_correlated_pair_returns: Tier 1 (Sharpe — i.i.d. violation)
- synthetic_lookahead_feature_src: Tier 2 (Leakage — df.shift(-1))
- zero_fee_simulator_kwargs: Tier 3 (Cost realism — fee=0)
- retroactive_drop_pair_history: Tier 4 (Survivorship — post-hoc drop)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def perfectly_correlated_pair_returns() -> pd.DataFrame:
    """144 pairs × 30 days where every pair earns the same daily return.

    Used by Tier 1 audit fixture test: proves that
    correlation_corrected_sharpe collapses naive Sharpe to ~0
    when avg_pair_corr ≈ 1.0.
    """
    rng = np.random.default_rng(123)
    n_pairs, n_days = 144, 30
    daily_returns = rng.normal(0.01, 0.02, size=n_days)
    rows = []
    for day in range(n_days):
        for pair in range(n_pairs):
            rows.append({
                "pair_id": f"pair_{pair}",
                "entry_day": day,
                "entry_ts": day * 86400,
                "pnl_pp": float(daily_returns[day]),  # IDENTICAL across pairs
                "traded": True,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_lookahead_feature_src() -> str:
    """Source-code string with a textbook negative-shift leak.

    Used by Tier 2 audit fixture test: proves classify_features
    flags result["leaky_feature"] = df["spread"].shift(-1) as Leaking.
    """
    return '''
def f(df):
    result = df.copy()
    # PURE LEAK: uses df.shift(-1) which is one bar in the future.
    result["leaky_feature"] = df["spread"].shift(-1)
    return result
'''


@pytest.fixture
def zero_fee_simulator_kwargs() -> dict:
    """Kwargs that, if passed to a backtester, would silently drop fees.

    Used by Tier 3 audit fixture test: proves the cost audit's
    fee-handling check flags simulators where entry+exit cost == 0.
    """
    return {"entry_cost_pp": 0.0, "exit_cost_pp": 0.0}


@pytest.fixture
def retroactive_drop_pair_history() -> dict:
    """A pair-history record where the drop reason references the realized outcome.

    Used by Tier 4 audit fixture test: proves the survivorship audit
    flags pairs whose drop_reason mentions "loss", "negative_return",
    or any post-resolution outcome metadata.
    """
    return {
        "pair_id": "synthetic_post_hoc_drop_pair",
        "drop_reason": "post_hoc_low_return",
        "realized_return": -0.42,
        "is_retroactive": True,
    }
