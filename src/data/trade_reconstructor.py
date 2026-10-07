"""Trade-to-candle reconstruction with microstructure features.

Converts raw trade records into 4-hour OHLCV+VWAP candles with 14
microstructure columns. Only bars containing at least one trade are emitted.

This module is standalone -- it does not depend on KalshiAdapter or
PolymarketAdapter classes.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import logging

import pandas as pd

from src.data.schemas import CANDLE_COLUMNS

logger = logging.getLogger(__name__)

# Default bar size: 4 hours in seconds
DEFAULT_BAR_SECONDS = 14400


def reconstruct_candles(
    trades_df: pd.DataFrame,
    bar_seconds: int = DEFAULT_BAR_SECONDS,
) -> pd.DataFrame:
    """Reconstruct OHLCV+VWAP candles from raw trade records.

    Args:
        trades_df: DataFrame with columns ["timestamp", "price", "volume", "side"].
            - timestamp: int Unix seconds UTC
            - price: float trade price
            - volume: float trade size (contracts)
            - side: str "buy", "sell", or "unknown"
        bar_seconds: Bar aggregation window in seconds (default 14400 = 4 hours).

    Returns:
        DataFrame with CANDLE_COLUMNS (14 columns), sorted by timestamp ascending.
        Only bars containing at least one trade are emitted.
    """
    if trades_df.empty:
        return pd.DataFrame(columns=CANDLE_COLUMNS)

    # Work on a copy to avoid mutating input
    df = trades_df.copy()

    # Clip prices to [0.01, 0.99]
    df["price"] = df["price"].clip(0.01, 0.99)

    # Compute bar timestamp (floor to bar boundary)
    df["bar_ts"] = (df["timestamp"] // bar_seconds) * bar_seconds

    # Sort by timestamp to ensure OHLC ordering
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Aggregate per bar
    candles = []
    for bar_ts, group in df.groupby("bar_ts", sort=True):
        prices = group["price"]
        volumes = group["volume"]
        sides = group["side"]

        total_volume = volumes.sum()

        # VWAP = sum(price * volume) / sum(volume)
        dollar_vol = (prices * volumes).sum()
        vwap = dollar_vol / total_volume if total_volume > 0 else prices.mean()

        # Buy/sell volume split
        buy_mask = sides == "buy"
        sell_mask = sides == "sell"
        buy_volume = volumes[buy_mask].sum()
        sell_volume = volumes[sell_mask].sum()

        # Realized spread: mean(sell_prices) - mean(buy_prices)
        buy_prices = prices[buy_mask]
        sell_prices = prices[sell_mask]
        if len(buy_prices) > 0 and len(sell_prices) > 0:
            realized_spread = sell_prices.mean() - buy_prices.mean()
        else:
            realized_spread = 0.0

        candles.append({
            "timestamp": int(bar_ts),
            "vwap": float(vwap),
            "open": float(prices.iloc[0]),
            "high": float(prices.max()),
            "low": float(prices.min()),
            "close": float(prices.iloc[-1]),
            "volume": float(total_volume),
            "trade_count": len(group),
            "dollar_volume": float(dollar_vol),
            "buy_volume": float(buy_volume),
            "sell_volume": float(sell_volume),
            "realized_spread": float(realized_spread),
            "max_trade_size": float(volumes.max()),
            "has_trade": True,
        })

    result = pd.DataFrame(candles, columns=CANDLE_COLUMNS)
    result = result.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"Reconstructed {len(result)} candles from {len(trades_df)} trades "
        f"(bar_seconds={bar_seconds})"
    )
    return result
