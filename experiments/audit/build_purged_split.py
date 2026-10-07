"""Pair-stratified train/test split builder (Phase 18-08).

The canonical 80/20 row-index split (data/processed/{train,test}.parquet)
bridges all 144 pairs (Tier 2 audit, Plan 18-03, leakage_audit.json
n_embargo_violations=142). Bridging means the same pair lifecycle is
seen in both train and test, so any feature with even mild
autocorrelation leaks information across the boundary.

This script rebuilds the split with **pair atomicity**: every pair lives
entirely in train OR entirely in test, never both. We concatenate the
canonical train + test parquets, group by ``pair_id``, shuffle the unique
pair list under seed=42, and assign the first 20% of pairs to test, the
remaining 80% to train. The total row count is preserved (no rows
dropped — only reassigned by pair).

Output:
    data/processed/purged_split/train.parquet
    data/processed/purged_split/test.parquet
    data/processed/purged_split/split_metadata.json

Usage:
    PYTHONPATH=. python experiments/audit/build_purged_split.py

Implements requirement: AUDIT-07 (build side).
"""
# AI-assisted authorship: written with Anthropic Claude (Opus 4.7) as
# pair-programming assistant. All design decisions are the authors'.
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.seed import set_all_seeds


CANONICAL_TRAIN = Path("data/processed/train.parquet")
CANONICAL_TEST = Path("data/processed/test.parquet")
PURGED_DIR = Path("data/processed/purged_split")
PURGED_TRAIN = PURGED_DIR / "train.parquet"
PURGED_TEST = PURGED_DIR / "test.parquet"
PURGED_METADATA = PURGED_DIR / "split_metadata.json"

DEFAULT_SEED = 42
DEFAULT_TEST_FRAC = 0.20


def load_canonical_combined() -> pd.DataFrame:
    """Load canonical train + test parquets and concatenate for re-splitting.

    A ``_origin_split`` column is added (values: 'train' / 'test') purely
    for traceability — it is dropped before the purged parquets are
    written, so downstream training code sees the same schema as the
    canonical files.
    """
    if not CANONICAL_TRAIN.exists() or not CANONICAL_TEST.exists():
        raise FileNotFoundError(
            f"Canonical split not found at {CANONICAL_TRAIN} / {CANONICAL_TEST}. "
            "Run Phase 3 (Feature Engineering) first."
        )
    train = pd.read_parquet(CANONICAL_TRAIN)
    test = pd.read_parquet(CANONICAL_TEST)
    train = train.assign(_origin_split="train")
    test = test.assign(_origin_split="test")
    combined = pd.concat([train, test], ignore_index=True)
    return combined


def pair_stratified_split(
    df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into train/test halves, atomically by ``pair_id``.

    Every pair contributes ALL its bars to one half. Seed=42 (canonical)
    fixes the shuffle so re-running produces an identical split.

    Args:
        df: DataFrame with a ``pair_id`` column.
        seed: RNG seed (default: 42).
        test_frac: Fraction of UNIQUE PAIRS (not rows) to assign to test.
            With 144 pairs and 0.20, this gives 28 test pairs / 116 train
            pairs (approximately — exact counts depend on rounding).

    Returns:
        ``(train_df, test_df)`` as fresh copies.
    """
    set_all_seeds(seed)

    # Convert to a plain numpy object array so np.random.Generator.shuffle
    # behaves correctly (ArrowStringArray triggers a UserWarning otherwise).
    pairs = np.asarray(df["pair_id"].unique(), dtype=object).copy()
    # Use a local Generator so the shuffle is independent of any
    # downstream consumers that also call np.random.* under seed=42.
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)

    n_test = int(round(len(pairs) * test_frac))
    test_pairs = set(pairs[:n_test])
    train_pairs = set(pairs[n_test:])

    assert train_pairs.isdisjoint(test_pairs), (
        "INTERNAL ERROR: train_pairs and test_pairs overlap — split builder bug."
    )
    assert len(train_pairs) + len(test_pairs) == len(pairs), (
        "INTERNAL ERROR: pair count not conserved across split."
    )

    train_df = df[df["pair_id"].isin(train_pairs)].copy()
    test_df = df[df["pair_id"].isin(test_pairs)].copy()
    return train_df, test_df


def write_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = PURGED_DIR,
    seed: int = DEFAULT_SEED,
    test_frac: float = DEFAULT_TEST_FRAC,
) -> dict:
    """Write the split parquets and a metadata JSON for traceability."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Drop the traceability column before writing so downstream code sees
    # the canonical schema.
    train_out = train_df.drop(columns=["_origin_split"], errors="ignore")
    test_out = test_df.drop(columns=["_origin_split"], errors="ignore")

    train_out.to_parquet(output_dir / "train.parquet", index=False)
    test_out.to_parquet(output_dir / "test.parquet", index=False)

    metadata = {
        "schema_version": "1.0",
        "generator": "experiments/audit/build_purged_split.py",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "test_frac": test_frac,
        "split_type": "pair_stratified",
        "n_train_pairs": int(train_out["pair_id"].nunique()),
        "n_test_pairs": int(test_out["pair_id"].nunique()),
        "n_train_rows": int(len(train_out)),
        "n_test_rows": int(len(test_out)),
        "train_pair_ids": sorted(train_out["pair_id"].unique().tolist()),
        "test_pair_ids": sorted(test_out["pair_id"].unique().tolist()),
        "no_bridge_property": "set(train.pair_id) & set(test.pair_id) == empty",
        "canonical_inputs": [
            CANONICAL_TRAIN.as_posix(),
            CANONICAL_TEST.as_posix(),
        ],
        "purpose": (
            "Replaces the canonical 80/20 row-index split (which bridges "
            "all 144 pairs, see experiments/results/audit/leakage_audit.json) "
            "with a pair-atomic 80/20 split for the leakage-free Phase 18-08 "
            "Sharpe recompute."
        ),
    }
    (output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> int:
    df = load_canonical_combined()
    n_total = len(df)
    n_pairs_total = df["pair_id"].nunique()

    train_df, test_df = pair_stratified_split(
        df, seed=DEFAULT_SEED, test_frac=DEFAULT_TEST_FRAC
    )
    metadata = write_split(train_df, test_df)

    bridging = set(train_df["pair_id"]) & set(test_df["pair_id"])
    bridge_marker = "no bridge ✓" if not bridging else f"BRIDGE FOUND ({len(bridging)})"

    print(
        f"Canonical combined: {n_total} rows, {n_pairs_total} unique pairs"
    )
    print(
        f"Purged split: {metadata['n_train_pairs']} train pairs / "
        f"{metadata['n_test_pairs']} test pairs ({bridge_marker})"
    )
    print(
        f"  -> train rows {metadata['n_train_rows']} / "
        f"test rows {metadata['n_test_rows']}"
    )
    print(f"Wrote {PURGED_TRAIN}")
    print(f"Wrote {PURGED_TEST}")
    print(f"Wrote {PURGED_METADATA}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
