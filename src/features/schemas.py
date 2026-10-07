"""Feature engineering schemas and column contracts.

Defines the column contract between the aligned_pairs.parquet input
(from Phase 2.1 aligner) and the feature engineering output.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.

# Columns inherited from aligned_pairs.parquet (31 columns from Phase 2.1)
ALIGNED_COLUMNS = [
    "timestamp",
    "kalshi_vwap",
    "kalshi_open",
    "kalshi_high",
    "kalshi_low",
    "kalshi_close",
    "kalshi_volume",
    "kalshi_trade_count",
    "kalshi_dollar_volume",
    "kalshi_buy_volume",
    "kalshi_sell_volume",
    "kalshi_realized_spread",
    "kalshi_max_trade_size",
    "kalshi_has_trade",
    "kalshi_hours_since_last_trade",
    "polymarket_vwap",
    "polymarket_open",
    "polymarket_high",
    "polymarket_low",
    "polymarket_close",
    "polymarket_volume",
    "polymarket_trade_count",
    "polymarket_dollar_volume",
    "polymarket_buy_volume",
    "polymarket_sell_volume",
    "polymarket_realized_spread",
    "polymarket_max_trade_size",
    "polymarket_has_trade",
    "polymarket_hours_since_last_trade",
    "spread",
    "pair_id",
]

# New derived feature columns computed by engineering.py
DERIVED_FEATURE_COLUMNS = [
    "price_velocity",                    # spread[t] - spread[t-1] per pair
    "volume_ratio",                      # kalshi_volume / polymarket_volume
    "spread_momentum",                   # spread.rolling(3).mean() per pair
    "spread_volatility",                 # spread.rolling(3).std() per pair
    "kalshi_order_flow_imbalance",       # (buy_vol - sell_vol) / (buy_vol + sell_vol)
    "polymarket_order_flow_imbalance",   # same for polymarket
    # --- Added 2026-04-12: richer signal features ---
    "spread_momentum_6",                 # spread.rolling(6).mean() — medium-term trend
    "spread_momentum_12",                # spread.rolling(12).mean() — longer-term trend
    "spread_volatility_6",              # spread.rolling(6).std() — medium-term vol
    "spread_zscore",                     # (spread - rolling_mean) / rolling_std — how anomalous
    "spread_range",                      # rolling(6) max - min — recent trading range
    "dollar_volume_ratio",               # kalshi_dollar_vol / poly_dollar_vol
    "trade_count_ratio",                 # kalshi_trade_count / poly_trade_count
    "mid_price",                         # (kalshi_close + poly_close) / 2
    "price_divergence_pct",              # abs(spread) / mid_price — relative divergence
    # --- Quant-grade microstructure features (2026-04-12) ---
    # From academic market microstructure literature (Kyle 1985,
    # Amihud 2002, Corwin & Schultz 2012, Roll 1984, Burgi et al 2026)
    "kalshi_amihud",                     # Amihud illiquidity: |return| / dollar_volume
    "polymarket_amihud",                 # same for polymarket
    "kalshi_cs_spread",                  # Corwin-Schultz implied bid-ask spread
    "polymarket_cs_spread",              # same for polymarket
    "kalshi_hl_vol",                     # Bekker-Parkinson high-low volatility
    "polymarket_hl_vol",                 # same for polymarket
    "ofi_differential",                  # cross-platform order flow imbalance gap
    "kalshi_kyle_lambda",                # Kyle's Lambda: price impact coefficient
    "polymarket_kyle_lambda",            # same for polymarket
    "boundary_distance",                 # min(mid, 1-mid) — option gamma analog
    "longshot_score",                    # favorite-longshot bias indicator
    "kalshi_roll_spread",                # Roll (1984) implied spread from return autocov
    "polymarket_roll_spread",            # same for polymarket
]

# Full output columns = ALIGNED_COLUMNS + DERIVED_FEATURE_COLUMNS
OUTPUT_COLUMNS = ALIGNED_COLUMNS + DERIVED_FEATURE_COLUMNS
