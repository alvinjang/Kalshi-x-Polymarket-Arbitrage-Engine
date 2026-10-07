"""Continuous market discovery for the live trading pair universe.

Runs end-to-end:
  1. Fetch currently-active Kalshi markets
  2. Fetch currently-active Polymarket markets
  3. Keyword pre-filter -> semantic matching -> structural quality filter
  4. Upsert results into data/live/active_matches.json, preserving
     existing pair_ids so the collector doesn't have to reindex.

Intended to run far less often than the trading cycle — every 2-6 hours
is plenty, since the matcher is expensive and new short-dated contracts
appear on a slow timescale anyway.

Usage:
    python scripts/discover_markets.py
    python scripts/discover_markets.py --live-dir data/live
    python scripts/discover_markets.py --similarity-threshold 0.75
    python scripts/discover_markets.py --dry-run   # print results, don't write
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Discover new Kalshi<->Polymarket arbitrage pairs",
        prog="python scripts/discover_markets.py",
    )
    parser.add_argument(
        "--live-dir",
        type=str,
        default="data/live",
        help="Directory containing active_matches.json (default: data/live)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.70,
        help="Minimum cosine similarity for a candidate match (default: 0.70)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print results without writing to active_matches.json",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("discover_markets")

    live_dir = Path(args.live_dir)

    if args.dry_run:
        # Execute the fetch + match pipeline but stop before the upsert.
        from src.live.market_discovery import (
            fetch_active_kalshi_markets,
            fetch_active_poly_markets,
            match_markets,
            upsert_active_matches,
        )

        kalshi = fetch_active_kalshi_markets()
        poly = fetch_active_poly_markets()
        new_matches = match_markets(
            kalshi, poly, similarity_threshold=args.similarity_threshold
        )

        existing_path = live_dir / "active_matches.json"
        if existing_path.exists():
            with open(existing_path) as f:
                existing = json.load(f)
        else:
            existing = []
        _, stats = upsert_active_matches(existing, new_matches)

        print("=== DRY RUN ===")
        print(f"Kalshi fetched:    {len(kalshi)}")
        print(f"Polymarket fetched: {len(poly)}")
        print(f"New matches found: {len(new_matches)}")
        print(f"Would add:         {stats['added']}")
        print(f"Would update:      {stats['updated']}")
        print(f"Would mark stale:  {stats['stale']}")
        if new_matches:
            print("\nSample new matches (first 10):")
            for m in new_matches[:10]:
                print(
                    f"  sim={m['similarity']:.3f}  "
                    f"{m['kalshi_ticker']:30s}  "
                    f"-> {m['poly_title'][:60]}"
                )
        return 0

    # Normal run
    from src.live.market_discovery import run_discovery

    try:
        stats = run_discovery(
            live_dir=live_dir,
            similarity_threshold=args.similarity_threshold,
        )
    except Exception as e:
        logger.error("Discovery run failed: %s", e, exc_info=True)
        return 1

    print("=== Discovery run complete ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
