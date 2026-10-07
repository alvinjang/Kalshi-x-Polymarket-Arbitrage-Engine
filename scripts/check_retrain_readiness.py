#!/usr/bin/env python3
"""Quick check: is the live data ready for Tier 2 retraining?

Run anytime:
    python scripts/check_retrain_readiness.py

Reports bars-per-pair stats for the new content-addressed pair_ids,
how many pairs have crossed each training threshold, and an ETA
for when Tier 2 (LSTM/GRU) training becomes viable.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main() -> None:
    bars_path = Path("data/live/bars.parquet")
    if not bars_path.exists():
        print("No bars.parquet yet — SCC hasn't collected any bars.")
        sys.exit(0)

    df = pd.read_parquet(bars_path)
    # Split old vs new format
    new = df[~df["pair_id"].str.startswith("live_")]
    old = df[df["pair_id"].str.startswith("live_")]

    print("=== Retrain Readiness Check ===")
    print(f"  Total bars:     {len(df):,}")
    print(f"  Old (live_NNNN): {len(old):,} bars, {old['pair_id'].nunique()} pairs (orphaned, pre-schema-fix)")
    print(f"  New (content):   {len(new):,} bars, {new['pair_id'].nunique()} pairs")
    print()

    if len(new) == 0:
        print("  No new-format bars yet. SCC needs to run a few cycles.")
        print("  Check back in ~1 hour.")
        return

    bpp = new.groupby("pair_id").size()
    print(f"  New bars/pair:  min={bpp.min()}, median={bpp.median():.0f}, "
          f"p75={bpp.quantile(0.75):.0f}, max={bpp.max()}")

    t1_ready = (bpp >= 10).sum()
    t2_ready = (bpp >= 100).sum()
    t2_close = ((bpp >= 50) & (bpp < 100)).sum()
    print()
    print(f"  Tier 1 ready (>=10 bars):   {t1_ready:,} pairs")
    print(f"  Tier 2 ready (>=100 bars):  {t2_ready:,} pairs")
    print(f"  Tier 2 close (50-99 bars):  {t2_close:,} pairs")

    # ETA: assume 4 bars/hour (15-min cycle)
    bars_per_hour = 4
    median_bars = bpp.median()
    if median_bars < 100:
        hours_to_100 = (100 - median_bars) / bars_per_hour
        print(f"\n  ETA to median pair hitting 100 bars: ~{hours_to_100:.0f} hours")
    else:
        print("\n  READY: median pair has 100+ bars. Run:")
        print("    python scripts/run_data_scaling.py --auto --include-tier2")

    # Commodity pairs specifically
    commodity_kw = ["wti", "oil", "gas", "silver", "gold", "corn",
                    "natgas", "nickel", "hoil", "copper", "wheat", "brent"]
    commodity_pids = [pid for pid in bpp.index
                      if any(kw in pid.lower() for kw in commodity_kw)]
    if commodity_pids:
        comm_bpp = bpp[commodity_pids]
        print(f"\n  Commodity pairs with bars: {len(comm_bpp)}")
        print(f"  Commodity bars/pair: min={comm_bpp.min()}, "
              f"median={comm_bpp.median():.0f}, max={comm_bpp.max()}")
    else:
        print("\n  No commodity bars yet (WTI/gold/silver — check back soon)")

    # Freshness check
    if "timestamp" in new.columns:
        latest_ts = new["timestamp"].max()
        now = int(datetime.now(timezone.utc).timestamp())
        age_min = (now - latest_ts) / 60
        print(f"\n  Most recent bar: {age_min:.0f} minutes ago", end="")
        if age_min > 30:
            print("  ⚠ SCC may not be collecting")
        else:
            print("  (healthy)")


if __name__ == "__main__":
    main()
