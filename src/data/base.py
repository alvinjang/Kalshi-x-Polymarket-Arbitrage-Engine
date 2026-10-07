"""Abstract base class for platform-specific market data adapters."""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import pandas as pd
from src.data.schemas import MarketMetadata, validate_candlestick_df

logger = logging.getLogger(__name__)


class MarketDataAdapter(ABC):
    """Common interface for platform-specific data fetching.

    Subclasses must implement:
      - list_markets(): discover available markets
      - get_candlesticks(): fetch price history for one market

    This base class provides:
      - get_or_fetch_candlesticks(): caching wrapper around get_candlesticks
      - ingest_all(): orchestrate full ingestion with caching
    """

    @abstractmethod
    def list_markets(self, categories: list[str]) -> list[MarketMetadata]:
        """Return metadata for all resolved markets in given categories."""
        ...

    @abstractmethod
    def get_candlesticks(self, market_id: str, **kwargs) -> pd.DataFrame:
        """Return candlestick/price data for a single market.

        Returns DataFrame with at minimum the CANDLESTICK_COLUMNS:
            timestamp (int): Unix seconds UTC
            open (float|None): Opening price
            high (float|None): High price
            low (float|None): Low price
            close (float|None): Closing price
            volume (float): Trade volume
        """
        ...

    def get_or_fetch_candlesticks(
        self, market_id: str, output_dir: Path, **kwargs
    ) -> pd.DataFrame | None:
        """Fetch candlesticks with file-based caching.

        If parquet file exists for this market, reads from disk.
        Otherwise fetches from API, validates, and writes to disk.
        Returns None if fetch produces empty data.
        """
        cache_path = output_dir / f"{market_id}.parquet"
        if cache_path.exists():
            logger.info(f"Cache hit: {cache_path}")
            return pd.read_parquet(cache_path)

        df = self.get_candlesticks(market_id, **kwargs)
        if df is None or len(df) == 0:
            logger.warning(f"No data for market {market_id}")
            return None

        errors = validate_candlestick_df(df)
        if errors:
            logger.error(f"Validation errors for {market_id}: {errors}")
            return None

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info(f"Cached: {cache_path} ({len(df)} rows)")
        return df

    def ingest_all(
        self, categories: list[str], output_dir: Path
    ) -> dict:
        """Orchestrate full ingestion: discover markets, fetch all candlesticks.

        Returns summary dict with counts of successes, failures, cache hits.
        """
        logger.info(f"Starting ingestion for categories: {categories}")
        markets = self.list_markets(categories)
        logger.info(f"Discovered {len(markets)} markets")

        stats = {"total": len(markets), "success": 0, "cached": 0, "empty": 0, "error": 0}

        for i, market in enumerate(markets):
            try:
                cache_path = output_dir / f"{market.market_id}.parquet"
                was_cached = cache_path.exists()

                df = self.get_or_fetch_candlesticks(market.market_id, output_dir)
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

        logger.info(f"Ingestion complete: {stats}")
        return stats
