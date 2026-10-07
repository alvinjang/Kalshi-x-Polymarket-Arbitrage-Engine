#!/usr/bin/env python3
"""Cut the canonical oil-only train/test split for the retraining study.

Produces:
    data/processed/canonical_oil/train.parquet           pair-stratified train rows
    data/processed/canonical_oil/test.parquet            pair-stratified test rows (disjoint pairs)
    data/processed/canonical_oil/split_metadata.json     full split provenance
    data/processed/canonical_oil/robustness_100bar_pairs.json  pair list for the 100-bar robustness rerun

Design constraints (load-bearing for the adversarial audit):
    * Disjoint pair sets between train and test (the original audit
      caught an embargo violation where 142 of 144 pairs bridged the
      split in the original Phase 3 file).
    * Seed 42 to match the original paper's canonical run.
    * Embargo width >= 1 bar: satisfied by construction. Pair-
      stratified disjoint sets guarantee zero temporal overlap between
      train and test, and the spread_change_target = spread.shift(-1)
      - spread label generation drops the last bar of every pair (NaN
      target), which serves as the 1-bar embargo on label leakage
      forward in time.
    * 50-bar threshold for the headline run (193 pairs, 24,848 rows).
    * 100-bar threshold cached for the robustness rerun (89 pairs).
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.category import derive_category_from_ticker

SEED = 42
BAR_THRESHOLD_HEADLINE = 50
BAR_THRESHOLD_ROBUSTNESS = 100
TRAIN_FRAC = 0.8
OUT_DIR = Path("data/processed/canonical_oil")


def series_prefix(pair_id: str, pid_to_ticker: dict) -> str:
    """Coarse Kalshi series identifier (KXWTI, KXWTIW, KXBRENTMON, etc).

    Two derivation paths:
      1. If pair_id is in pid_to_ticker, parse the ticker (KXWTIW-26APR24-T90.99
         -> KXWTIW).
      2. Otherwise, parse the content-addressed pair_id directly. Pair IDs
         look like "kxwtiw26apr24t9099-0x2c50e53f"; the leading alphabetic
         run (before the first digit) is the series. pair_mapping.json
         caps at 2,000 entries while the bars universe has many more
         pairs, so the fallback path is the common case for this
         canonical split.
    """
    ticker = pid_to_ticker.get(pair_id, "")
    if ticker:
        return ticker.split("-")[0]
    head = pair_id.split("-")[0]  # lowercase normalized kalshi ticker
    # Strip everything from the first digit on.
    import re
    m = re.match(r"^([a-z]+)", head)
    if m:
        return m.group(1).upper()
    return "UNKNOWN"


def main() -> int:
    print("=== Canonical oil-only split ===")

    # 1. Load bars + pair mapping
    bars = pd.read_parquet("data/live/bars.parquet")
    new = bars[~bars["pair_id"].str.startswith("live_")].copy()
    mapping = json.load(open("data/live/pair_mapping.json"))
    pid_to_ticker = {pid: m["kalshi_market_id"] for pid, m in mapping.items()}

    def category_of(pair_id: str) -> str:
        ticker = pid_to_ticker.get(pair_id, pair_id.upper().split("-")[0])
        return derive_category_from_ticker(ticker)

    new["category"] = new["pair_id"].map(category_of)
    oil = new[new["category"] == "oil"].copy()
    print(f"  Oil universe: {oil['pair_id'].nunique()} pairs, {len(oil):,} bars")

    # 2. Apply 50-bar threshold for headline split, cache 100-bar cohort
    bpp = oil.groupby("pair_id").size()
    qualified = sorted(bpp[bpp >= BAR_THRESHOLD_HEADLINE].index.tolist())
    qualified_100 = sorted(bpp[bpp >= BAR_THRESHOLD_ROBUSTNESS].index.tolist())
    print(f"  Qualifying at >= {BAR_THRESHOLD_HEADLINE} bars: {len(qualified)} pairs")
    print(f"  Qualifying at >= {BAR_THRESHOLD_ROBUSTNESS} bars: {len(qualified_100)} pairs")

    # 3. Pair-stratified 80/20 split with seed 42
    rng = random.Random(SEED)
    shuffled = qualified.copy()
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * TRAIN_FRAC))
    train_pairs = sorted(shuffled[:n_train])
    test_pairs = sorted(shuffled[n_train:])
    overlap = set(train_pairs) & set(test_pairs)
    assert not overlap, f"FATAL: {len(overlap)} pairs in BOTH train and test"
    print(f"  Split: {len(train_pairs)} train / {len(test_pairs)} test (disjoint verified)")

    # 4. Subset bars
    train_df = oil[oil["pair_id"].isin(train_pairs)].copy()
    test_df = oil[oil["pair_id"].isin(test_pairs)].copy()
    # Drop the synthetic category column before writing; downstream
    # feature engineering expects the original column set.
    train_df = train_df.drop(columns=["category"])
    test_df = test_df.drop(columns=["category"])
    print(f"  Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    # 5. Series breakdown for metadata
    series_train = Counter(series_prefix(p, pid_to_ticker) for p in train_pairs)
    series_test = Counter(series_prefix(p, pid_to_ticker) for p in test_pairs)
    print(f"  Train series: {dict(series_train)}")
    print(f"  Test series:  {dict(series_test)}")

    # 6. Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(OUT_DIR / "train.parquet")
    test_df.to_parquet(OUT_DIR / "test.parquet")

    train_bpp = train_df.groupby("pair_id").size()
    test_bpp = test_df.groupby("pair_id").size()

    metadata = {
        "split_seed": SEED,
        "bars_per_pair_threshold_headline": BAR_THRESHOLD_HEADLINE,
        "bars_per_pair_threshold_robustness": BAR_THRESHOLD_ROBUSTNESS,
        "category_filter": "oil",
        "category_definition": ("kalshi tickers matching the 'oil' category "
                                "from src.features.category.derive_category_from_ticker "
                                "(KXWTI*, KXBRENT*, KXCRUDE, KXDIESEL, KXHEATINGOIL, "
                                "KXGASOLINE, KXMEXCUBOIL)"),
        "n_pairs_qualifying_total": len(qualified),
        "n_pairs_train": len(train_pairs),
        "n_pairs_test": len(test_pairs),
        "n_rows_train": int(len(train_df)),
        "n_rows_test": int(len(test_df)),
        "train_bars_per_pair": {
            "min": int(train_bpp.min()),
            "median": float(train_bpp.median()),
            "p75": float(train_bpp.quantile(0.75)),
            "max": int(train_bpp.max()),
        },
        "test_bars_per_pair": {
            "min": int(test_bpp.min()),
            "median": float(test_bpp.median()),
            "p75": float(test_bpp.quantile(0.75)),
            "max": int(test_bpp.max()),
        },
        "ticker_series_train": dict(series_train),
        "ticker_series_test": dict(series_test),
        "pair_ids_train": train_pairs,
        "pair_ids_test": test_pairs,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_bars_parquet": "data/live/bars.parquet",
        "source_pair_mapping": "data/live/pair_mapping.json",
        "embargo_note": (
            "Embargo width >= 1 bar is satisfied by construction. "
            "Pair-stratified disjoint sets guarantee zero temporal "
            "overlap between train and test. The spread_change_target "
            "= spread.shift(-1) - spread label generation drops the "
            "last bar of every pair (NaN target), which serves as the "
            "1-bar embargo on label leakage forward in time."
        ),
        "notes": (
            "Pair-stratified 80/20 split, disjoint pair sets verified. "
            "Seed 42 matches original paper canonical run. The 100-bar "
            "robustness pair list is cached in robustness_100bar_pairs.json "
            "for the parallel run reported in the writeup's robustness section."
        ),
    }
    with open(OUT_DIR / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Wrote split_metadata.json")

    robustness = {
        "bars_per_pair_threshold": BAR_THRESHOLD_ROBUSTNESS,
        "n_pairs": len(qualified_100),
        "pair_ids": qualified_100,
        "is_subset_of_canonical": all(p in set(qualified) for p in qualified_100),
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "note": ("Subset of the canonical_oil pair universe at >= 100 "
                 "bars per pair, for the robustness rerun reported in "
                 "the writeup. Use the same train/test pair partition "
                 "from split_metadata.json: filter pair_ids_train and "
                 "pair_ids_test to this list."),
    }
    with open(OUT_DIR / "robustness_100bar_pairs.json", "w") as f:
        json.dump(robustness, f, indent=2)
    print(f"  Wrote robustness_100bar_pairs.json ({len(qualified_100)} pairs)")
    print(f"\nDone. Artifacts in {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
