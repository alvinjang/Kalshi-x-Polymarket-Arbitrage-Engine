"""Pull candlestick/price data only for accepted matched pairs.

Reads accepted_pairs.json, pulls hourly data from both platforms
for each market in each pair. Much faster than full ingestion
since we're only pulling ~154 markets (77 pairs × 2 platforms).
"""
import json
import logging
import time
from pathlib import Path

from src.data.kalshi import KalshiAdapter
from src.data.polymarket import PolymarketAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    pairs_path = Path("data/processed/accepted_pairs.json")
    kalshi_dir = Path("data/raw/kalshi")
    poly_dir = Path("data/raw/polymarket")

    pairs = json.load(open(pairs_path))
    logger.info(f"Loaded {len(pairs)} accepted pairs")

    # Collect unique market IDs per platform
    kalshi_ids = set()
    poly_ids = set()
    poly_token_map = {}  # market_id -> clob_token_ids

    # Load Polymarket metadata for token ID lookup
    poly_meta = json.load(open(poly_dir / "_metadata.json"))
    poly_meta_map = {m["market_id"]: m for m in poly_meta}

    for p in pairs:
        kalshi_ids.add(p["kalshi_market_id"])
        pmid = p["polymarket_market_id"]
        poly_ids.add(pmid)
        # Get clob token IDs from metadata
        if pmid in poly_meta_map:
            poly_token_map[pmid] = poly_meta_map[pmid].get("clob_token_ids", [])

    logger.info(f"Unique Kalshi markets: {len(kalshi_ids)}")
    logger.info(f"Unique Polymarket markets: {len(poly_ids)}")

    # --- Kalshi ingestion ---
    logger.info("=== Starting Kalshi candlestick pull ===")
    kalshi = KalshiAdapter()

    # Get historical cutoff
    kalshi._get_historical_cutoff()
    logger.info(f"Kalshi cutoff: {kalshi._cutoff_ts}")

    # Build lookup: market_id -> resolution_date
    kalshi_meta = json.load(open(kalshi_dir / "_metadata.json"))
    kalshi_meta_map = {m["market_id"]: m for m in kalshi_meta}

    JAN_2024 = 1704067200  # Start from Jan 2024

    # Look up series_ticker for each market via event endpoint
    # Market ticker like KXU3-26FEB-T4.3 -> event ticker KXU3-26FEB -> series KXU3
    logger.info("Looking up series_ticker for Kalshi markets...")
    event_cache = {}
    for market_id in sorted(kalshi_ids):
        # Derive event ticker: everything before the last -T or last - segment
        parts = market_id.rsplit("-", 1)
        if len(parts) == 2 and parts[1].startswith("T"):
            event_ticker = parts[0]
        else:
            event_ticker = market_id.rsplit("-", 1)[0]

        if event_ticker not in event_cache:
            try:
                data = kalshi.client.get(f"events/{event_ticker}")
                event_cache[event_ticker] = data.get("event", {}).get("series_ticker", "")
            except Exception:
                event_cache[event_ticker] = ""

        series = event_cache.get(event_ticker, "")
        if series:
            kalshi._market_series[market_id] = series

    logger.info(f"Resolved {sum(1 for v in kalshi._market_series.values() if v)} series_tickers")

    kalshi_success = 0
    kalshi_empty = 0
    for i, market_id in enumerate(sorted(kalshi_ids)):
        meta = kalshi_meta_map.get(market_id, {})
        close_time = meta.get("resolution_date", "")

        # Check cache first
        cache_path = kalshi_dir / f"{market_id}.parquet"
        if cache_path.exists():
            kalshi_success += 1
            continue

        try:
            df = kalshi.get_candlesticks_with_fallback(
                market_id,
                close_time=close_time,
                start_ts=JAN_2024,
                end_ts=int(time.time()),
            )
            if df is not None and len(df) > 0 and df["close"].notna().any():
                df.to_parquet(cache_path)
                kalshi_success += 1
                logger.info(f"  Kalshi {market_id}: {len(df)} rows")
            else:
                kalshi_empty += 1
                logger.warning(f"  Kalshi {market_id}: no usable data")
        except Exception as e:
            logger.warning(f"  Kalshi {market_id}: {e}")
            kalshi_empty += 1

        if (i + 1) % 10 == 0:
            logger.info(f"  Kalshi progress: {i+1}/{len(kalshi_ids)} ({kalshi_success} with data, {kalshi_empty} empty)")

    logger.info(f"Kalshi complete: {kalshi_success} with data, {kalshi_empty} empty out of {len(kalshi_ids)}")

    # --- Polymarket ingestion ---
    logger.info("=== Starting Polymarket price history pull ===")
    poly = PolymarketAdapter()

    # Pre-populate the token map so get_candlesticks can find token IDs
    poly._market_token_map = poly_token_map

    poly_success = 0
    poly_empty = 0
    for i, market_id in enumerate(sorted(poly_ids)):
        try:
            df = poly.get_or_fetch_candlesticks(market_id, poly_dir)
            if df is not None and len(df) > 0:
                poly_success += 1
            else:
                poly_empty += 1
            if (i + 1) % 10 == 0:
                logger.info(f"  Polymarket progress: {i+1}/{len(poly_ids)} ({poly_success} with data, {poly_empty} empty)")
        except Exception as e:
            logger.warning(f"  Polymarket {market_id}: {e}")
            poly_empty += 1

    logger.info(f"Polymarket complete: {poly_success} with data, {poly_empty} empty out of {len(poly_ids)}")

    # Summary
    print(f"\n{'='*50}")
    print("TARGETED INGESTION COMPLETE")
    print(f"{'='*50}")
    print(f"  Kalshi: {kalshi_success}/{len(kalshi_ids)} markets with data")
    print(f"  Polymarket: {poly_success}/{len(poly_ids)} markets with data")
    print(f"  Pairs with both sides: check data/raw/*/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
