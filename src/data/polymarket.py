"""Polymarket platform adapter for market discovery and price history ingestion.

Uses three separate APIs:
  - Gamma API: Market metadata and discovery (events, markets)
  - CLOB API: Pre-aggregated price history (prices-history)
  - Data API: Raw trade records (trades) as fallback
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import json
import logging
import time

import pandas as pd

from src.data.base import MarketDataAdapter
from src.data.client import ResilientClient
from src.data.schemas import MarketMetadata, CANDLESTICK_COLUMNS

logger = logging.getLogger(__name__)


class PolymarketAdapter(MarketDataAdapter):
    """Adapter for Polymarket prediction market data.

    Discovers resolved markets via Gamma API keyword filtering,
    fetches price history from CLOB API (with Data API trade fallback),
    and normalizes output to the common candlestick schema.
    """

    GAMMA_BASE = "https://gamma-api.polymarket.com"
    CLOB_BASE = "https://clob.polymarket.com"
    DATA_BASE = "https://data-api.polymarket.com"

    CHUNK_DAYS = 14
    MAX_TRADE_OFFSET = 15000

    CRYPTO_KEYWORDS = [
        "btc", "bitcoin", "eth", "ethereum", "solana", "sol ",
        "crypto", "defi", "token", "altcoin", "nft",
    ]
    FINANCE_KEYWORDS = [
        "fed", "inflation", "cpi", "gdp", "recession", "unemployment",
        "rate cut", "rate hike", "interest rate", "s&p", "nasdaq",
        "dow", "treasury", "bond", "yield", "fomc",
    ]
    KEYWORD_MAP = {
        "crypto": CRYPTO_KEYWORDS,
        "finance": FINANCE_KEYWORDS,
    }

    def __init__(
        self,
        gamma_client: ResilientClient | None = None,
        clob_client: ResilientClient | None = None,
        data_client: ResilientClient | None = None,
    ):
        self.gamma_client = gamma_client or ResilientClient(
            base_url=self.GAMMA_BASE,
            max_retries=3,
            backoff_factor=1.0,
            requests_per_second=10.0,
        )
        self.clob_client = clob_client or ResilientClient(
            base_url=self.CLOB_BASE,
            max_retries=3,
            backoff_factor=1.0,
            requests_per_second=10.0,
        )
        self.data_client = data_client or ResilientClient(
            base_url=self.DATA_BASE,
            max_retries=3,
            backoff_factor=1.0,
            requests_per_second=10.0,
        )
        self._market_token_map: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Market Discovery
    # ------------------------------------------------------------------

    def list_markets(self, categories: list[str] | None = None) -> list[MarketMetadata]:
        """Discover resolved Polymarket markets matching keyword categories.

        Uses Gamma API /events with keyword filtering on title + description.
        Category matching is determined by which keyword list matched first.
        """
        if categories is None:
            categories = list(self.KEYWORD_MAP.keys())

        # Build combined keyword list per requested category
        keyword_to_category: dict[str, str] = {}
        for cat in categories:
            for kw in self.KEYWORD_MAP.get(cat, []):
                keyword_to_category[kw] = cat

        all_keywords = list(keyword_to_category.keys())
        if not all_keywords:
            logger.warning(f"No keywords for categories: {categories}")
            return []

        markets: list[MarketMetadata] = []
        offset = 0

        while True:
            events = self.gamma_client.get(
                "events",
                params={"closed": "true", "limit": 100, "offset": offset},
            )
            if not events:
                break

            for event in events:
                title = event.get("title", "").lower()
                description = event.get("description", "").lower()
                text = f"{title} {description}"

                # Determine matched category
                matched_category = None
                for kw in all_keywords:
                    if kw in text:
                        matched_category = keyword_to_category[kw]
                        break

                if matched_category is None:
                    continue

                # Process each market within matching event
                for market in event.get("markets", []):
                    condition_id = market.get("conditionId", "")
                    if not condition_id:
                        continue

                    # Parse clobTokenIds -- CRITICAL: stringified JSON array
                    raw_token_ids = market.get("clobTokenIds", "[]")
                    try:
                        clob_token_ids = json.loads(raw_token_ids)
                    except (json.JSONDecodeError, TypeError):
                        clob_token_ids = []

                    self._market_token_map[condition_id] = clob_token_ids

                    # Parse outcomes (also stringified JSON in Gamma responses)
                    raw_outcomes = market.get("outcomes", '["Yes", "No"]')
                    if isinstance(raw_outcomes, str):
                        try:
                            outcomes = json.loads(raw_outcomes)
                        except (json.JSONDecodeError, TypeError):
                            outcomes = ["Yes", "No"]
                    else:
                        outcomes = raw_outcomes if raw_outcomes else ["Yes", "No"]

                    metadata = MarketMetadata(
                        market_id=condition_id,
                        question=market.get("question", ""),
                        category=matched_category,
                        platform="polymarket",
                        resolution_date=market.get("endDate", ""),
                        result=None,
                        outcomes=outcomes,
                    )
                    markets.append(metadata)

            offset += 100
            if len(events) < 100:
                break

        logger.info(
            f"Discovered {len(markets)} Polymarket markets across categories: {categories}"
        )
        return markets

    # ------------------------------------------------------------------
    # Price History
    # ------------------------------------------------------------------

    def get_candlesticks(self, market_id: str, **kwargs) -> pd.DataFrame:
        """Fetch price history for a market (conditionId).

        Primary: CLOB /prices-history with startTs/endTs chunking.
        Fallback: Data API /trades with OHLCV aggregation.

        Tries Yes token first, then No token if Yes returns empty.
        """
        clob_token_ids = self._market_token_map.get(market_id, [])
        if not clob_token_ids:
            logger.warning(f"No clobTokenIds for market {market_id}, returning empty")
            return pd.DataFrame(columns=CANDLESTICK_COLUMNS)

        start_ts = kwargs.get("start_ts", 0)
        end_ts = kwargs.get("end_ts", int(time.time()))

        # Try CLOB /prices-history for each token (Yes first, then No)
        for token_id in clob_token_ids:
            df = self._fetch_clob_prices(token_id, start_ts, end_ts)
            if not df.empty:
                logger.info(
                    f"CLOB prices-history returned {len(df)} rows for market "
                    f"{market_id} (token {token_id})"
                )
                return df

        # Fallback to Data API /trades
        logger.info(
            f"CLOB returned no data for market {market_id}, falling back to trades"
        )
        trades = self._fetch_trades(market_id)
        if trades:
            df = self._trades_to_ohlcv(trades, freq_minutes=60)
            if not df.empty:
                logger.info(
                    f"Trade fallback produced {len(df)} OHLCV bars for market {market_id}"
                )
                return df

        logger.warning(f"No price data from any source for market {market_id}")
        return pd.DataFrame(columns=CANDLESTICK_COLUMNS)

    def _fetch_clob_prices(
        self, token_id: str, start_ts: int, end_ts: int
    ) -> pd.DataFrame:
        """Fetch CLOB /prices-history with startTs/endTs chunking.

        Chunks into CHUNK_DAYS windows (14 days) to avoid response size limits.
        Returns DataFrame with CANDLESTICK_COLUMNS.
        """
        chunk_seconds = self.CHUNK_DAYS * 86400
        all_points: list[dict] = []
        current_start = start_ts

        while current_start < end_ts:
            current_end = min(current_start + chunk_seconds, end_ts)
            try:
                data = self.clob_client.get(
                    "prices-history",
                    params={
                        "market": token_id,
                        "startTs": current_start,
                        "endTs": current_end,
                        "fidelity": 60,
                    },
                )
                history = data.get("history", []) if isinstance(data, dict) else []
                all_points.extend(history)
            except Exception as e:
                logger.error(
                    f"CLOB prices-history error for token {token_id} "
                    f"[{current_start}-{current_end}]: {e}"
                )
            current_start = current_end

        if not all_points:
            return pd.DataFrame(columns=CANDLESTICK_COLUMNS)

        # Parse CLOB response: {"t": timestamp_int, "p": price_string}
        rows = []
        for point in all_points:
            ts = int(point["t"])
            price = float(point["p"])
            rows.append({
                "timestamp": ts,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0,
            })

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        df = df.reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Trade Fallback
    # ------------------------------------------------------------------

    def _fetch_trades(self, condition_id: str) -> list[dict]:
        """Fetch trades from Data API with offset pagination.

        Stops at MAX_TRADE_OFFSET (3000) or when fewer than 500 trades returned.
        """
        all_trades: list[dict] = []
        offset = 0
        limit = 500

        while offset <= self.MAX_TRADE_OFFSET:
            try:
                trades = self.data_client.get(
                    "trades",
                    params={
                        "market": condition_id,
                        "limit": limit,
                        "offset": offset,
                    },
                )
                if not trades:
                    break
                all_trades.extend(trades)
                if len(trades) < limit:
                    break
                offset += limit
            except Exception as e:
                logger.error(f"Data API trades error at offset {offset}: {e}")
                break

        logger.info(
            f"Fetched {len(all_trades)} trades for market {condition_id} "
            f"(final offset: {offset})"
        )
        return all_trades

    def _trades_to_ohlcv(
        self, trades: list[dict], freq_minutes: int = 60
    ) -> pd.DataFrame:
        """Aggregate raw trades into OHLCV bars at the given frequency.

        Args:
            trades: List of trade dicts with price, size, timestamp fields.
            freq_minutes: Bar frequency in minutes (default: 60 = hourly).

        Returns:
            DataFrame with CANDLESTICK_COLUMNS, sorted by timestamp.
        """
        if not trades:
            return pd.DataFrame(columns=CANDLESTICK_COLUMNS)

        df = pd.DataFrame(trades)
        df["price"] = df["price"].astype(float)
        df["size"] = df["size"].astype(float)
        df["datetime"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)

        # Resample to desired frequency
        df = df.set_index("datetime")
        resampled = df["price"].resample(f"{freq_minutes}min").agg(
            open="first", high="max", low="min", close="last"
        )
        volume = df["size"].resample(f"{freq_minutes}min").sum()
        resampled["volume"] = volume

        # Drop intervals with no trades (NaN)
        resampled = resampled.dropna(subset=["open"])

        # Convert back to Unix timestamp
        resampled = resampled.reset_index()
        resampled["timestamp"] = (
            resampled["datetime"].astype("int64") // 10**9
        ).astype(int)
        resampled = resampled[CANDLESTICK_COLUMNS]
        resampled = resampled.sort_values("timestamp").reset_index(drop=True)

        return resampled
