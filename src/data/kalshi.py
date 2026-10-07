"""Kalshi platform adapter for market discovery and candlestick ingestion.

Discovers resolved markets in Economics, Crypto, and Financials categories
via the /series -> /events chain, fetches minute-level candlesticks with
automatic time-window chunking, and routes to historical vs live endpoints
based on the dynamic /historical/cutoff response.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import logging
import time
from pathlib import Path

import pandas as pd

from src.data.base import MarketDataAdapter
from src.data.client import ResilientClient
from src.data.schemas import MarketMetadata

logger = logging.getLogger(__name__)


class KalshiAdapter(MarketDataAdapter):
    """Adapter for Kalshi prediction market API.

    Implements MarketDataAdapter to provide:
    - Market discovery via /series -> /events pagination chain
    - Candlestick fetching with historical/live endpoint routing
    - Null OHLC handling via bid-ask midpoint fallback
    - Time-window chunking for large date ranges (max 4500 candles/request)
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    CATEGORIES = ["Economics", "Crypto", "Financials"]
    MAX_CANDLES_PER_REQUEST = 4500
    PERIOD_INTERVAL = 60  # hourly candlesticks

    def __init__(self, client: ResilientClient | None = None):
        if client is None:
            client = ResilientClient(
                base_url=self.BASE_URL,
                max_retries=3,
                backoff_factor=1.0,
                requests_per_second=18.0,  # Buffer below Kalshi's 20 req/s Basic tier
            )
        self.client = client
        self._cutoff_ts: str | None = None
        self._market_series: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Historical cutoff routing
    # ------------------------------------------------------------------

    def _get_historical_cutoff(self) -> str:
        """Fetch and cache the historical/live endpoint cutoff timestamp.

        Markets settled before this timestamp use the /historical/ endpoint.
        Markets settled after use the live endpoint.

        Returns:
            ISO8601 string of the cutoff timestamp.
        """
        if self._cutoff_ts is not None:
            return self._cutoff_ts

        data = self.client.get("historical/cutoff")
        self._cutoff_ts = data["market_settled_ts"]
        logger.info(
            f"Historical cutoff: markets settled before {self._cutoff_ts} "
            "use /historical/ endpoint"
        )
        return self._cutoff_ts

    def _is_historical(self, market_close_time: str) -> bool:
        """Determine if a market should use the historical endpoint.

        Args:
            market_close_time: ISO8601 string of the market's close_time.

        Returns:
            True if market settled before cutoff (use historical endpoint).
        """
        cutoff = self._get_historical_cutoff()
        # ISO8601 strings compare lexicographically correctly
        return market_close_time < cutoff

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    def list_markets(self, categories: list[str] | None = None) -> list[MarketMetadata]:
        """Discover resolved markets via /series -> /events pagination chain.

        Args:
            categories: Categories to query. Defaults to CATEGORIES.

        Returns:
            List of MarketMetadata for all discovered markets.
        """
        if categories is None:
            categories = self.CATEGORIES

        all_markets: list[MarketMetadata] = []

        for category in categories:
            series_data = self.client.get("series", params={"category": category})
            series_list = series_data.get("series", [])
            logger.info(f"Category '{category}': found {len(series_list)} series")

            category_count = 0
            for series in series_list:
                series_ticker = series["ticker"]
                cursor = None

                while True:
                    params: dict = {
                        "series_ticker": series_ticker,
                        "status": "settled",
                        "with_nested_markets": True,
                        "limit": 200,
                    }
                    if cursor:
                        params["cursor"] = cursor

                    events_data = self.client.get("events", params=params)
                    events = events_data.get("events", [])

                    for event in events:
                        for market in event.get("markets", []):
                            ticker = market.get("ticker", "")
                            if not ticker:
                                continue

                            metadata = MarketMetadata(
                                market_id=ticker,
                                question=market.get("title", ""),
                                category=category,
                                platform="kalshi",
                                resolution_date=market.get("close_time", ""),
                                result=market.get("result"),
                                outcomes=["Yes", "No"],
                            )
                            all_markets.append(metadata)
                            self._market_series[ticker] = series_ticker
                            category_count += 1

                    cursor = events_data.get("cursor", "")
                    if not cursor:
                        break

            logger.info(f"Category '{category}': {category_count} markets discovered")

        logger.info(f"Total markets discovered: {len(all_markets)}")
        return all_markets

    # ------------------------------------------------------------------
    # Candlestick fetching
    # ------------------------------------------------------------------

    def get_candlesticks(self, market_id: str, **kwargs) -> pd.DataFrame:
        """Fetch minute-level candlesticks with time-window chunking.

        Routes to historical or live endpoint based on /historical/cutoff.
        Handles null OHLC by computing bid-ask midpoint as fallback.

        Args:
            market_id: Kalshi market ticker.
            **kwargs:
                close_time: ISO8601 string for endpoint routing.
                start_ts: Unix seconds start (default 0).
                end_ts: Unix seconds end (default now).

        Returns:
            DataFrame with candlestick data sorted by timestamp.
        """
        start_ts = kwargs.get("start_ts", 0)
        end_ts = kwargs.get("end_ts", int(time.time()))
        close_time = kwargs.get("close_time")

        # Determine endpoint
        if close_time is not None and not self._is_historical(close_time):
            series_ticker = self._market_series.get(market_id, "")
            endpoint = f"series/{series_ticker}/markets/{market_id}/candlesticks"
        else:
            # Default to historical for resolved markets
            endpoint = f"historical/markets/{market_id}/candlesticks"

        # Time-window chunking
        chunk_seconds = self.MAX_CANDLES_PER_REQUEST * self.PERIOD_INTERVAL * 60
        all_candles: list[dict] = []
        current_start = start_ts

        while current_start < end_ts:
            chunk_end = min(current_start + chunk_seconds, end_ts)
            data = self.client.get(
                endpoint,
                params={
                    "start_ts": current_start,
                    "end_ts": chunk_end,
                    "period_interval": self.PERIOD_INTERVAL,
                },
            )
            candles = data.get("candlesticks", [])
            all_candles.extend(candles)
            current_start += chunk_seconds

        if not all_candles:
            return pd.DataFrame(
                columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "open_interest", "yes_bid_close", "yes_ask_close", "has_trades",
                ]
            )

        df = self._parse_candlesticks(all_candles)

        # Deduplicate and sort
        if not df.empty:
            df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        return df

    def _parse_candlesticks(self, candles: list[dict]) -> pd.DataFrame:
        """Parse raw Kalshi candlestick dicts into a normalized DataFrame.

        Handles null OHLC fields by computing bid-ask midpoint as fallback price.

        Args:
            candles: List of raw candlestick dicts from the API.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume,
            open_interest, yes_bid_close, yes_ask_close, has_trades.
        """
        rows = []
        for c in candles:
            timestamp = c["end_period_ts"]

            # Extract price fields
            price = c.get("price", {}) or {}
            trade_open = price.get("open")
            trade_high = price.get("high")
            trade_low = price.get("low")
            trade_close = price.get("close")

            # Extract bid/ask fields
            yes_bid = c.get("yes_bid", {}) or {}
            yes_ask = c.get("yes_ask", {}) or {}
            bid_close = yes_bid.get("close")
            ask_close = yes_ask.get("close")

            # Determine close price and has_trades
            if trade_close is not None:
                close_val = float(trade_close)
                has_trades = True
            elif bid_close is not None and ask_close is not None:
                close_val = (float(bid_close) + float(ask_close)) / 2
                has_trades = False
            else:
                close_val = None
                has_trades = False

            # Open/high/low: use trade values if available, else None
            open_val = float(trade_open) if trade_open is not None else None
            high_val = float(trade_high) if trade_high is not None else None
            low_val = float(trade_low) if trade_low is not None else None

            volume = float(c.get("volume", "0") or "0")
            open_interest = float(c.get("open_interest", "0") or "0")
            yes_bid_close = float(bid_close) if bid_close is not None else None
            yes_ask_close = float(ask_close) if ask_close is not None else None

            rows.append(
                {
                    "timestamp": timestamp,
                    "open": open_val,
                    "high": high_val,
                    "low": low_val,
                    "close": close_val,
                    "volume": volume,
                    "open_interest": open_interest,
                    "yes_bid_close": yes_bid_close,
                    "yes_ask_close": yes_ask_close,
                    "has_trades": has_trades,
                }
            )

        trades_count = sum(1 for r in rows if r["has_trades"])
        logger.info(f"Parsed {len(rows)} candlesticks, {trades_count} with trades")

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Ingestion override
    # ------------------------------------------------------------------

    def ingest_all(self, categories: list[str], output_dir: Path) -> dict:
        """Orchestrate full Kalshi ingestion with cutoff-aware endpoint routing.

        Calls _get_historical_cutoff at the start, then discovers markets
        and fetches candlesticks with correct endpoint routing per market.

        Args:
            categories: Categories to ingest.
            output_dir: Directory to write parquet files.

        Returns:
            Summary dict with counts.
        """
        # Fetch cutoff FIRST before any market processing
        self._get_historical_cutoff()

        logger.info(f"Starting Kalshi ingestion for categories: {categories}")
        markets = self.list_markets(categories)
        logger.info(f"Discovered {len(markets)} markets")

        stats = {"total": len(markets), "success": 0, "cached": 0, "empty": 0, "error": 0}

        for i, market in enumerate(markets):
            try:
                cache_path = output_dir / f"{market.market_id}.parquet"
                was_cached = cache_path.exists()

                df = self.get_or_fetch_candlesticks(
                    market.market_id, output_dir, close_time=market.resolution_date
                )
                if df is not None:
                    stats["success"] += 1
                    if was_cached:
                        stats["cached"] += 1
                else:
                    stats["empty"] += 1

                if (i + 1) % 50 == 0:
                    logger.info(f"Progress: {i + 1}/{len(markets)} markets processed")
            except Exception as e:
                logger.error(f"Error fetching {market.market_id}: {e}")
                stats["error"] += 1

        logger.info(f"Kalshi ingestion complete: {stats}")
        return stats
