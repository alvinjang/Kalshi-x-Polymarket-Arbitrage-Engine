"""Tests for the pair-stratified train/test split builder.

Phase 18-08 Plan: the canonical 80/20 row-index split bridges all 144 pairs
(Tier 2 audit verdict=FAILED, n_embargo_violations=142). This module
re-builds the split with pair atomicity — every pair lives entirely in
train OR entirely in test, never both. These tests assert that property
and the supporting reproducibility / row-conservation invariants.

Implements requirement: AUDIT-07 (test side).
"""
# AI-assisted authorship: written with Anthropic Claude (Opus 4.7) as
# pair-programming assistant. All design decisions are the authors'.
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.audit.build_purged_split import (
    CANONICAL_TEST,
    CANONICAL_TRAIN,
    PURGED_DIR,
    load_canonical_combined,
    pair_stratified_split,
    write_split,
)

PURGED_TRAIN = PURGED_DIR / "train.parquet"
PURGED_TEST = PURGED_DIR / "test.parquet"


def _purged_split_exists() -> bool:
    return PURGED_TRAIN.exists() and PURGED_TEST.exists()


@pytest.mark.skipif(
    not _purged_split_exists(),
    reason=(
        "Run experiments/audit/build_purged_split.py first to materialize "
        "the purged train/test parquets."
    ),
)
def test_no_bridge():
    """The whole point of the plan: no pair appears in both halves."""
    train = pd.read_parquet(PURGED_TRAIN)
    test = pd.read_parquet(PURGED_TEST)
    bridging = set(train["pair_id"]) & set(test["pair_id"])
    assert bridging == set(), (
        f"Found {len(bridging)} bridging pairs in purged split: "
        f"{sorted(bridging)[:5]}{'...' if len(bridging) > 5 else ''}"
    )


def test_reproducibility():
    """Same seed -> identical split row counts and pair_id assignment."""
    df = load_canonical_combined()
    train_a, test_a = pair_stratified_split(df, seed=42, test_frac=0.20)
    train_b, test_b = pair_stratified_split(df, seed=42, test_frac=0.20)
    assert len(train_a) == len(train_b)
    assert len(test_a) == len(test_b)
    assert set(train_a["pair_id"]) == set(train_b["pair_id"])
    assert set(test_a["pair_id"]) == set(test_b["pair_id"])


def test_no_row_loss():
    """Train + test row count equals the canonical combined input."""
    df = load_canonical_combined()
    canonical_total = len(df)
    train, test = pair_stratified_split(df, seed=42, test_frac=0.20)
    assert (
        len(train) + len(test) == canonical_total
    ), (
        f"Row count drift: canonical {canonical_total} vs purged "
        f"{len(train) + len(test)}"
    )


def test_min_pairs_in_each_split():
    """Both halves must have >= 10 unique pair_ids (sanity, not degenerate)."""
    df = load_canonical_combined()
    train, test = pair_stratified_split(df, seed=42, test_frac=0.20)
    assert train["pair_id"].nunique() >= 10
    assert test["pair_id"].nunique() >= 10
