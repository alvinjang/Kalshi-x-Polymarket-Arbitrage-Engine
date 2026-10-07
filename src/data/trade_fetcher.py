"""Unified trade fetching for Kalshi and Polymarket platforms.

Fetches raw trades via platform APIs, normalizes to a common schema
(timestamp, price, volume, side), and saves per-market parquet files.

This module is standalone -- it does not depend on KalshiAdapter or
PolymarketAdapter classes.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.schemas import TRADE_COLUMNS

logger = logging.getLogger(__name__)

# Polymarket Data API maximum offset (hard API limit)
MAX_TRADE_OFFSET = 15000


def fetch_kalshi_trades(market_id: str, client) -> list[dict]:
    """Fetch all trades for a Kalshi market via cursor pagination.

    Args:
        market_id: Kalshi market ticker (e.g. "KXU3-26FEB-T4.3").
        client: ResilientClient instance configured for Kalshi API.

    Returns:
        List of normalized trade dicts with keys:
        timestamp (int unix), price (float), volume (float), side (str).
        Sorted by timestamp ascending.
    """
    all_trades: list[dict] = []
    cursor = None

    while True:
        params: dict = {"ticker": market_id, "limit": 200}
        if cursor:
            params["cursor"] = cursor

        data = client.get("markets/trades", params=params)
        trades = data.get("trades", [])

        for t in trades:
            ts_str = t.get("created_time", "")
            try:
                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                ts = int(dt.timestamp())
            except (ValueError, TypeError):
                continue

            price = t.get("yes_price_dollars")
            if price is None:
                continue

            all_trades.append({
                "timestamp": ts,
                "price": float(price),
                "volume": float(t.get("count_fp", "1")),
                "side": t.get("taker_side", "unknown").lower(),
            })

        cursor = data.get("cursor", "")
        if not cursor or not trades:
            break

    # Sort by timestamp ascending
    all_trades.sort(key=lambda x: x["timestamp"])
    return all_trades


def fetch_polymarket_trades(condition_id: str, client) -> list[dict]:
    """Fetch all trades for a Polymarket market via offset pagination.

    Args:
        condition_id: Polymarket condition ID (hex string).
        client: ResilientClient instance configured for Polymarket Data API.

    Returns:
        List of normalized trade dicts with keys:
        timestamp (int unix), price (float), volume (float), side (str).
        Sorted by timestamp ascending.
    """
    all_trades: list[dict] = []
    offset = 0
    limit = 500

    while offset <= MAX_TRADE_OFFSET:
        try:
            trades = client.get(
                "trades",
                params={"market": condition_id, "limit": limit, "offset": offset},
            )
        except Exception as e:
            logger.error(f"Polymarket trades error at offset {offset}: {e}")
            break

        if not trades:
            break

        for t in trades:
            all_trades.append({
                "timestamp": int(t["timestamp"]),
                "price": float(t["price"]),
                "volume": float(t["size"]),
                "side": t.get("side", "unknown").lower(),
            })

        if len(trades) < limit:
            break
        offset += limit

    # Sort by timestamp ascending
    all_trades.sort(key=lambda x: x["timestamp"])
    return all_trades


def fetch_and_save_trades(
    pairs: list[dict],
    kalshi_client,
    poly_client,
    output_dir: Path,
) -> dict:
    """Fetch trades for all matched pairs and save as per-market parquet files.

    For each pair, fetches Kalshi and Polymarket trades. Skips markets that
    already have a cached parquet file on disk.

    Args:
        pairs: List of pair dicts with kalshi_market_id and polymarket_market_id.
        kalshi_client: ResilientClient for Kalshi API.
        poly_client: ResilientClient for Polymarket Data API.
        output_dir: Base directory for output (e.g. data/raw/).

    Returns:
        Stats dict with counts of fetched, cached, and empty markets.
    """
    stats = {
        "kalshi_fetched": 0,
        "kalshi_cached": 0,
        "kalshi_empty": 0,
        "polymarket_fetched": 0,
        "polymarket_cached": 0,
        "polymarket_empty": 0,
    }

    kalshi_dir = Path(output_dir) / "kalshi"
    poly_dir = Path(output_dir) / "polymarket"
    kalshi_dir.mkdir(parents=True, exist_ok=True)
    poly_dir.mkdir(parents=True, exist_ok=True)

    for i, pair in enumerate(pairs):
        kalshi_id = pair["kalshi_market_id"]
        poly_id = pair["polymarket_market_id"]

        # --- Kalshi ---
        kalshi_path = kalshi_dir / f"{kalshi_id}_trades.parquet"
        if kalshi_path.exists():
            stats["kalshi_cached"] += 1
        else:
            trades = fetch_kalshi_trades(kalshi_id, kalshi_client)
            if trades:
                df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
                df.to_parquet(kalshi_path, index=False)
                stats["kalshi_fetched"] += 1
            else:
                stats["kalshi_empty"] += 1

        # --- Polymarket ---
        poly_path = poly_dir / f"{poly_id}_trades.parquet"
        if poly_path.exists():
            stats["polymarket_cached"] += 1
        else:
            trades = fetch_polymarket_trades(poly_id, poly_client)
            if trades:
                df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
                df.to_parquet(poly_path, index=False)
                stats["polymarket_fetched"] += 1
            else:
                stats["polymarket_empty"] += 1

        # Log progress every 10 pairs
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(pairs)} pairs")

    logger.info(f"Trade fetch complete. Stats: {stats}")
    return stats
