"""CLI script to ingest all Kalshi historical market data.

Usage:
    .venv/bin/python -m src.data.ingest_kalshi [--categories Economics Crypto Financials] [--output-dir data/raw/kalshi]

Discovers all resolved markets in the specified categories,
fetches minute-level candlesticks, and saves as parquet files.
Idempotent: re-running skips already-cached markets.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import argparse
import json
import logging
from pathlib import Path

from src.data.kalshi import KalshiAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ["Economics", "Crypto", "Financials"]
DEFAULT_OUTPUT_DIR = Path("data/raw/kalshi")


def main():
    """Run Kalshi data ingestion pipeline.

    Discovers markets via /series -> /events, saves metadata as JSON,
    then fetches candlesticks with dynamic endpoint routing based on
    /historical/cutoff and file-based caching for idempotent re-runs.
    """
    parser = argparse.ArgumentParser(description="Ingest Kalshi historical market data")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help=f"Market categories to ingest (default: {DEFAULT_CATEGORIES})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for parquet files (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    logger.info(f"Starting Kalshi ingestion: categories={args.categories}, output={args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter = KalshiAdapter()

    # First, discover and save market metadata
    markets = adapter.list_markets(args.categories)
    metadata_path = args.output_dir / "_metadata.json"
    metadata_records = [
        {
            "market_id": m.market_id,
            "question": m.question,
            "category": m.category,
            "platform": m.platform,
            "resolution_date": m.resolution_date,
            "result": m.result,
            "outcomes": m.outcomes,
        }
        for m in markets
    ]
    with open(metadata_path, "w") as f:
        json.dump(metadata_records, f, indent=2)
    logger.info(f"Saved metadata for {len(markets)} markets to {metadata_path}")

    # Then, fetch candlesticks for each market (with caching)
    # ingest_all calls _get_historical_cutoff internally and routes each market
    # to the correct endpoint (historical vs live) based on the cutoff
    stats = adapter.ingest_all(args.categories, args.output_dir)

    logger.info(f"Ingestion complete: {json.dumps(stats, indent=2)}")
    return stats


if __name__ == "__main__":
    main()
