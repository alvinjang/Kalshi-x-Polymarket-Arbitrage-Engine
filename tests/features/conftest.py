"""Shared test fixtures for feature engineering tests.

Creates aligned_pairs-shaped DataFrames matching the Phase 2.1 output format:
31 columns, 4-hour bars, microstructure features.
"""
import numpy as np
import pandas as pd
import pytest


# Fixed epoch for reproducible test data: 2024-01-01 00:00:00 UTC
BASE_TIMESTAMP = 1704067200
BAR_SECONDS = 14400  # 4 hours


@pytest.fixture
def aligned_pairs_df():
    """Small aligned_pairs-shaped DataFrame: 10 rows, 2 pair_ids, all 31 columns.

    pair_id "pair-A": rows 0-4 (5 bars)
    pair_id "pair-B": rows 5-9 (5 bars)

    Includes some NaN spread rows (where one platform has no data) to test
    NaN propagation in derived features.
    """
    n = 10
    pair_ids = ["pair-A"] * 5 + ["pair-B"] * 5
    # Each pair has its own time range
    timestamps_a = [BASE_TIMESTAMP + i * BAR_SECONDS for i in range(5)]
    timestamps_b = [BASE_TIMESTAMP + (i + 10) * BAR_SECONDS for i in range(5)]
    timestamps = timestamps_a + timestamps_b

    # Kalshi VWAP: pair-A has values, pair-B has some NaN
    kalshi_vwap = [0.50, 0.52, 0.55, 0.53, 0.54,
                   0.60, np.nan, 0.62, 0.61, 0.63]
    # Polymarket VWAP
    polymarket_vwap = [0.48, 0.50, 0.51, 0.50, 0.52,
                       0.58, 0.59, np.nan, 0.60, 0.61]

    # Spread: kalshi_vwap - polymarket_vwap (NaN where either is NaN)
    spread = []
    for kv, pv in zip(kalshi_vwap, polymarket_vwap):
        if np.isnan(kv) or np.isnan(pv):
            spread.append(np.nan)
        else:
            spread.append(kv - pv)

    # Volumes
    kalshi_volume = [100.0, 200.0, 150.0, 0.0, 300.0,
                     50.0, 0.0, 100.0, 200.0, 150.0]
    polymarket_volume = [80.0, 0.0, 120.0, 90.0, 200.0,
                         60.0, 70.0, 0.0, 150.0, 100.0]

    # Buy/sell volumes for order flow imbalance
    kalshi_buy_volume = [60.0, 120.0, 80.0, 0.0, 180.0,
                         30.0, 0.0, 60.0, 120.0, 90.0]
    kalshi_sell_volume = [40.0, 80.0, 70.0, 0.0, 120.0,
                          20.0, 0.0, 40.0, 80.0, 60.0]
    polymarket_buy_volume = [50.0, 0.0, 70.0, 50.0, 120.0,
                              35.0, 40.0, 0.0, 90.0, 60.0]
    polymarket_sell_volume = [30.0, 0.0, 50.0, 40.0, 80.0,
                               25.0, 30.0, 0.0, 60.0, 40.0]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "kalshi_vwap": kalshi_vwap,
        "kalshi_open": [v - 0.01 if not np.isnan(v) else np.nan for v in kalshi_vwap],
        "kalshi_high": [v + 0.02 if not np.isnan(v) else np.nan for v in kalshi_vwap],
        "kalshi_low": [v - 0.02 if not np.isnan(v) else np.nan for v in kalshi_vwap],
        "kalshi_close": [v + 0.01 if not np.isnan(v) else np.nan for v in kalshi_vwap],
        "kalshi_volume": kalshi_volume,
        "kalshi_trade_count": [10.0, 20.0, 15.0, 0.0, 30.0,
                                5.0, 0.0, 10.0, 20.0, 15.0],
        "kalshi_dollar_volume": [v * 0.5 for v in kalshi_volume],
        "kalshi_buy_volume": kalshi_buy_volume,
        "kalshi_sell_volume": kalshi_sell_volume,
        "kalshi_realized_spread": [0.02, 0.03, 0.01, np.nan, 0.02,
                                    0.01, np.nan, 0.02, 0.03, 0.01],
        "kalshi_max_trade_size": [20.0, 30.0, 25.0, 0.0, 40.0,
                                   10.0, 0.0, 20.0, 30.0, 25.0],
        "kalshi_has_trade": [True, True, True, False, True,
                              True, False, True, True, True],
        "kalshi_hours_since_last_trade": [0.0, 0.0, 0.0, 4.0, 0.0,
                                           0.0, 4.0, 0.0, 0.0, 0.0],
        "polymarket_vwap": polymarket_vwap,
        "polymarket_open": [v - 0.01 if not np.isnan(v) else np.nan for v in polymarket_vwap],
        "polymarket_high": [v + 0.02 if not np.isnan(v) else np.nan for v in polymarket_vwap],
        "polymarket_low": [v - 0.02 if not np.isnan(v) else np.nan for v in polymarket_vwap],
        "polymarket_close": [v + 0.01 if not np.isnan(v) else np.nan for v in polymarket_vwap],
        "polymarket_volume": polymarket_volume,
        "polymarket_trade_count": [8.0, 0.0, 12.0, 9.0, 20.0,
                                    6.0, 7.0, 0.0, 15.0, 10.0],
        "polymarket_dollar_volume": [v * 0.48 for v in polymarket_volume],
        "polymarket_buy_volume": polymarket_buy_volume,
        "polymarket_sell_volume": polymarket_sell_volume,
        "polymarket_realized_spread": [0.01, np.nan, 0.02, 0.01, 0.02,
                                        0.01, 0.01, np.nan, 0.02, 0.01],
        "polymarket_max_trade_size": [15.0, 0.0, 20.0, 15.0, 30.0,
                                       10.0, 12.0, 0.0, 25.0, 18.0],
        "polymarket_has_trade": [True, False, True, True, True,
                                  True, True, False, True, True],
        "polymarket_hours_since_last_trade": [0.0, 4.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 4.0, 0.0, 0.0],
        "spread": spread,
        "pair_id": pair_ids,
    })

    return df
