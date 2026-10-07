#!/usr/bin/env python3
"""Rebuild data pipeline: fetch trades -> reconstruct candles -> align pairs -> save.

Full trade-based data reconstruction pipeline. Fetches raw trades from both
Kalshi and Polymarket APIs, reconstructs 4-hour OHLCV+VWAP candles with
microstructure features, aligns cross-platform pairs with forward-fill and
staleness decay, applies quality filters, and produces aligned_pairs.parquet.

Usage:
    python scripts/rebuild_data.py [--pairs-file PATH] [--output-dir PATH] [--skip-fetch] [--bar-seconds N]

Examples:
    python scripts/rebuild_data.py                          # Full pipeline
    python scripts/rebuild_data.py --skip-fetch             # Skip API fetch, use cached trades
    python scripts/rebuild_data.py --bar-seconds 3600       # 1-hour bars (not recommended)
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when running as script
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd

from src.data.client import ResilientClient
from src.data.trade_fetcher import fetch_and_save_trades
from src.data.trade_reconstructor import reconstruct_candles
from src.data.aligner import align_all_pairs

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rebuild data pipeline: fetch trades -> reconstruct candles -> align pairs -> save.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/rebuild_data.py                   Full pipeline
  python scripts/rebuild_data.py --skip-fetch      Use cached raw trades
  python scripts/rebuild_data.py --bar-seconds 3600  Use 1-hour bars
        """,
    )
    parser.add_argument(
        "--pairs-file",
        type=Path,
        default=Path("data/processed/accepted_pairs.json"),
        help="Path to accepted_pairs.json (default: data/processed/accepted_pairs.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Base output directory (default: data)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip Stage 2 (use existing raw trades on disk)",
    )
    parser.add_argument(
        "--bar-seconds",
        type=int,
        default=14400,
        help="Candle bar size in seconds (default: 14400 = 4 hours)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full rebuild data pipeline."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    total_start = time.time()
    errors: list[str] = []

    # =========================================================================
    # Stage 1: Load pairs
    # =========================================================================
    stage_start = time.time()
    logger.info("=" * 60)
    logger.info("Stage 1: Loading accepted pairs")
    logger.info("=" * 60)

    with open(args.pairs_file) as f:
        pairs = json.load(f)

    logger.info(f"Loaded {len(pairs)} accepted pairs from {args.pairs_file}")
    logger.info(f"Stage 1 complete in {time.time() - stage_start:.1f}s")

    # =========================================================================
    # Stage 2: Fetch raw trades
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Stage 2: Fetching raw trades")
    logger.info("=" * 60)

    raw_dir = args.output_dir / "raw"

    if args.skip_fetch:
        logger.info("--skip-fetch: Skipping trade fetching, using cached data")
    else:
        stage_start = time.time()

        # Create resilient clients for each platform
        kalshi_client = ResilientClient(
            base_url="https://api.elections.kalshi.com/trade-api/v2",
            requests_per_second=18.0,
            max_retries=3,
            backoff_factor=1.0,
            timeout=30,
        )
        poly_client = ResilientClient(
            base_url="https://data-api.polymarket.com",
            requests_per_second=2.0,
            max_retries=3,
            backoff_factor=2.0,
            timeout=30,
        )

        fetch_stats = fetch_and_save_trades(
            pairs, kalshi_client, poly_client, raw_dir,
        )

        logger.info(f"Trade fetch stats: {json.dumps(fetch_stats, indent=2)}")
        logger.info(f"Stage 2 complete in {time.time() - stage_start:.1f}s")

    # =========================================================================
    # Stage 3: Reconstruct candles
    # =========================================================================
    stage_start = time.time()
    logger.info("=" * 60)
    logger.info("Stage 3: Reconstructing candles from raw trades")
    logger.info("=" * 60)

    candles_dir = args.output_dir / "processed" / "candles"
    kalshi_candles_dir = candles_dir / "kalshi"
    poly_candles_dir = candles_dir / "polymarket"
    kalshi_candles_dir.mkdir(parents=True, exist_ok=True)
    poly_candles_dir.mkdir(parents=True, exist_ok=True)

    # Collect unique markets across all pairs
    kalshi_markets = set()
    poly_markets = set()
    for pair in pairs:
        kalshi_markets.add(pair["kalshi_market_id"])
        poly_markets.add(pair["polymarket_market_id"])

    kalshi_reconstructed = 0
    poly_reconstructed = 0

    # Reconstruct Kalshi candles
    for market_id in sorted(kalshi_markets):
        trades_path = raw_dir / "kalshi" / f"{market_id}_trades.parquet"
        candle_path = kalshi_candles_dir / f"{market_id}_candles.parquet"

        if not trades_path.exists():
            logger.debug(f"Kalshi {market_id}: no trade file, skipping")
            continue

        try:
            trades_df = pd.read_parquet(trades_path)
            if trades_df.empty:
                logger.debug(f"Kalshi {market_id}: empty trades, skipping")
                continue

            candles = reconstruct_candles(trades_df, bar_seconds=args.bar_seconds)
            candles.to_parquet(candle_path, index=False)
            kalshi_reconstructed += 1
        except Exception as e:
            msg = f"Kalshi {market_id}: reconstruction error: {e}"
            logger.error(msg)
            errors.append(msg)

    # Reconstruct Polymarket candles
    for market_id in sorted(poly_markets):
        trades_path = raw_dir / "polymarket" / f"{market_id}_trades.parquet"
        candle_path = poly_candles_dir / f"{market_id}_candles.parquet"

        if not trades_path.exists():
            logger.debug(f"Polymarket {market_id}: no trade file, skipping")
            continue

        try:
            trades_df = pd.read_parquet(trades_path)
            if trades_df.empty:
                logger.debug(f"Polymarket {market_id}: empty trades, skipping")
                continue

            candles = reconstruct_candles(trades_df, bar_seconds=args.bar_seconds)
            candles.to_parquet(candle_path, index=False)
            poly_reconstructed += 1
        except Exception as e:
            msg = f"Polymarket {market_id}: reconstruction error: {e}"
            logger.error(msg)
            errors.append(msg)

    logger.info(
        f"Reconstructed candles for {kalshi_reconstructed} Kalshi markets, "
        f"{poly_reconstructed} Polymarket markets"
    )
    logger.info(f"Stage 3 complete in {time.time() - stage_start:.1f}s")

    # =========================================================================
    # Stage 4: Align and validate
    # =========================================================================
    stage_start = time.time()
    logger.info("=" * 60)
    logger.info("Stage 4: Aligning cross-platform pairs and validating")
    logger.info("=" * 60)

    aligned_df, quality_report = align_all_pairs(
        pairs,
        candles_dir=candles_dir,
        bar_seconds=args.bar_seconds,
    )

    # Drop rows with NaN spreads (no valid prediction target)
    if len(aligned_df) > 0:
        before_drop = len(aligned_df)
        aligned_df = aligned_df.dropna(subset=["spread"]).reset_index(drop=True)
        logger.info(
            f"Dropped {before_drop - len(aligned_df)} rows with NaN spread; "
            f"{len(aligned_df)} rows remain"
        )

    # Save aligned dataset
    aligned_path = args.output_dir / "processed" / "aligned_pairs.parquet"
    aligned_path.parent.mkdir(parents=True, exist_ok=True)

    if len(aligned_df) > 0:
        aligned_df.to_parquet(aligned_path, compression="snappy", index=False)
        logger.info(
            f"Saved aligned dataset: {aligned_path} "
            f"({len(aligned_df)} rows, {len(aligned_df.columns)} columns)"
        )
    else:
        # Save empty parquet with correct schema
        aligned_df.to_parquet(aligned_path, compression="snappy", index=False)
        logger.warning("No pairs passed quality filters. Empty aligned dataset saved.")

    # Save quality report
    report_path = args.output_dir / "processed" / "data_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(quality_report, f, indent=2)
    logger.info(f"Saved quality report: {report_path}")

    # Log summary
    logger.info(
        f"Aligned {quality_report['aligned_pairs']} pairs, "
        f"excluded {quality_report['excluded_pairs']} pairs. "
        f"Output: {len(aligned_df)} rows, "
        f"{len(aligned_df.columns) if len(aligned_df) > 0 else 0} columns"
    )
    logger.info(f"Exclusion breakdown: {json.dumps(quality_report['exclusion_reasons'])}")
    logger.info(f"Stage 4 complete in {time.time() - stage_start:.1f}s")

    # =========================================================================
    # Summary
    # =========================================================================
    total_duration = time.time() - total_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total duration: {total_duration:.1f}s ({total_duration / 60:.1f}m)")
    logger.info(f"Pairs: {quality_report['aligned_pairs']}/{quality_report['total_pairs']} passed filters")
    logger.info(f"Output: {aligned_path}")
    logger.info(f"Report: {report_path}")

    if errors:
        logger.warning(f"Encountered {len(errors)} errors during reconstruction:")
        for err in errors:
            logger.warning(f"  - {err}")


if __name__ == "__main__":
    main()
