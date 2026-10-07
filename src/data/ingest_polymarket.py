"""CLI script to ingest all Polymarket historical market data.

Usage:
    .venv/bin/python -m src.data.ingest_polymarket [--categories crypto finance] [--output-dir data/raw/polymarket]

Discovers resolved markets via Gamma API keyword filtering,
fetches price history from CLOB API (with Data API trade fallback),
and saves as parquet files. Idempotent: re-running skips already-cached markets.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import argparse
import json
import logging
from pathlib import Path

from src.data.polymarket import PolymarketAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ["crypto", "finance"]
DEFAULT_OUTPUT_DIR = Path("data/raw/polymarket")


def main():
    parser = argparse.ArgumentParser(description="Ingest Polymarket historical market data")
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

    logger.info(f"Starting Polymarket ingestion: categories={args.categories}, output={args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter = PolymarketAdapter()

    # First, discover and save market metadata (includes clobTokenIds mapping)
    markets = adapter.list_markets(args.categories)

    # Save metadata including the token ID mapping for downstream use
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
            "clob_token_ids": adapter._market_token_map.get(m.market_id, []),
        }
        for m in markets
    ]
    with open(metadata_path, "w") as f:
        json.dump(metadata_records, f, indent=2)
    logger.info(f"Saved metadata for {len(markets)} markets to {metadata_path}")

    # Then, fetch price history for each market (with caching)
    stats = adapter.ingest_all(args.categories, args.output_dir)

    # Log data source breakdown
    logger.info(f"Ingestion complete: {json.dumps(stats, indent=2)}")
    return stats


if __name__ == "__main__":
    main()
