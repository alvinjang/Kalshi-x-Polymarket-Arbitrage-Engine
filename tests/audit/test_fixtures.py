"""Sanity tests for Wave 0 fixtures.

Each fixture must behave as specified BEFORE downstream Wave 1+2 audit
fixture tests can rely on it. If any of these tests fails, audit fixture
tests in higher waves will be testing against incorrect inputs.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def test_perfectly_correlated_returns_has_avg_corr_near_one(perfectly_correlated_pair_returns):
    df = perfectly_correlated_pair_returns
    panel = df.pivot_table(
        index="entry_day", columns="pair_id",
        values="pnl_pp", aggfunc="sum",
    )
    corr = panel.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    triu_vals = corr.values[mask]
    triu_vals = triu_vals[~np.isnan(triu_vals)]
    avg_corr = float(triu_vals.mean())
    assert avg_corr > 0.99, (
        f"perfectly_correlated_pair_returns fixture must produce avg_corr ≈ 1.0; "
        f"got {avg_corr:.4f}"
    )


def test_synthetic_lookahead_src_contains_negative_shift(synthetic_lookahead_feature_src):
    src = synthetic_lookahead_feature_src
    assert "shift(-1)" in src, (
        "synthetic_lookahead_feature_src must literally contain 'shift(-1)' "
        "so the Tier 2 leakage classifier has something to flag"
    )
    assert "result[\"leaky_feature\"]" in src, (
        "fixture must assign to result['leaky_feature'] for classifier regex to hit"
    )


def test_zero_fee_kwargs_match_audit_target(zero_fee_simulator_kwargs):
    kw = zero_fee_simulator_kwargs
    assert kw["entry_cost_pp"] == 0.0
    assert kw["exit_cost_pp"] == 0.0


def test_retroactive_drop_marker_set(retroactive_drop_pair_history):
    rec = retroactive_drop_pair_history
    assert rec["is_retroactive"] is True
    assert rec["drop_reason"].startswith("post_hoc"), (
        f"drop_reason should start with 'post_hoc' to match Tier 4 classifier; "
        f"got {rec['drop_reason']}"
    )
