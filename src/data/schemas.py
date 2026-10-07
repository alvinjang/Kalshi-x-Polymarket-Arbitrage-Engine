"""Parquet schemas and data validation for market data."""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class MarketMetadata:
    """Platform-agnostic market metadata."""
    market_id: str
    question: str
    category: str
    platform: str           # "kalshi" or "polymarket"
    resolution_date: str    # ISO8601 UTC string
    result: str | None      # "yes", "no", or None
    outcomes: list[str] = field(default_factory=lambda: ["Yes", "No"])


# Required columns for candlestick DataFrames written to parquet
CANDLESTICK_COLUMNS = [
    "timestamp",       # int: Unix seconds UTC
    "open",           # float or None: opening price
    "high",           # float or None: high price
    "low",            # float or None: low price
    "close",          # float or None: closing price (trade or bid-ask midpoint)
    "volume",         # float: trade volume (0 if no trades)
]

# Optional columns that adapters may include
OPTIONAL_COLUMNS = [
    "open_interest",   # float: open interest
    "yes_bid_close",   # float: yes bid at period close
    "yes_ask_close",   # float: yes ask at period close
    "has_trades",      # bool: whether any trades occurred in this period
]

# Columns for raw trade parquet files (normalized across platforms)
TRADE_COLUMNS = [
    "timestamp",   # int: Unix seconds UTC
    "price",       # float: trade price
    "volume",      # float: trade size (contracts)
    "side",        # str: "buy", "sell", or "unknown"
]

# Columns for 4-hour reconstructed candles with microstructure features
CANDLE_COLUMNS = [
    "timestamp",        # int: Unix seconds UTC, start of 4-hour bar
    "vwap",             # float: volume-weighted average price
    "open",             # float: first trade price in bar
    "high",             # float: highest trade price
    "low",              # float: lowest trade price
    "close",            # float: last trade price
    "volume",           # float: total contracts traded
    "trade_count",      # int: number of trades
    "dollar_volume",    # float: sum(price * size)
    "buy_volume",       # float: volume from buy-side takers
    "sell_volume",      # float: volume from sell-side takers
    "realized_spread",  # float: mean(ask_hits) - mean(bid_hits)
    "max_trade_size",   # float: largest single trade
    "has_trade",        # bool: True (always True for reconstructed bars)
]

METADATA_COLUMNS = [
    "market_id",       # str: platform-specific identifier
    "question",        # str: human-readable market question
    "category",        # str: economics, crypto, financials
    "platform",        # str: "kalshi" or "polymarket"
    "resolution_date", # str: ISO8601 UTC
    "result",          # str or None: "yes", "no"
]


def validate_candlestick_df(df: pd.DataFrame) -> list[str]:
    """Validate that a DataFrame has required candlestick columns.

    Returns list of error messages. Empty list means valid.
    """
    errors = []
    for col in CANDLESTICK_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")
    if len(df) == 0:
        errors.append("DataFrame is empty")
    if "timestamp" in df.columns and df["timestamp"].isna().any():
        errors.append("timestamp column contains NaN values")
    return errors
