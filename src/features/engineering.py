"""Derived feature computation from aligned_pairs data.

Computes microstructure features from Phase 2.1's aligned_pairs.parquet:
spread velocity, volume ratio, spread momentum/volatility, and order flow
imbalance for each platform.

All rolling/diff operations use groupby("pair_id") to prevent cross-pair
contamination.

Low-liquidity filtering (FEAT-03) is enforced upstream by
src/data/aligner.py (MIN_TRADES_PER_PLATFORM=20). No additional
filtering is needed here.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
import numpy as np
import pandas as pd

from src.features.schemas import OUTPUT_COLUMNS


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to aligned_pairs DataFrame.

    All rolling/diff operations are per pair_id to prevent cross-pair
    contamination. NaN inputs produce NaN outputs (no imputation here).

    Args:
        df: aligned_pairs.parquet loaded as DataFrame (31 columns).

    Returns:
        DataFrame with 31 + 6 = 37 columns.
    """
    result = df.copy()

    # 1. price_velocity: spread[t] - spread[t-1] per pair
    result["price_velocity"] = result.groupby("pair_id")["spread"].diff()

    # 2. volume_ratio: kalshi_volume / polymarket_volume
    #    Replace inf/-inf with NaN for division-by-zero safety
    result["volume_ratio"] = (
        result["kalshi_volume"] / result["polymarket_volume"]
    ).replace([np.inf, -np.inf], np.nan)

    # 3. spread_momentum: spread.rolling(3).mean() per pair
    result["spread_momentum"] = result.groupby("pair_id")["spread"].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )

    # 4. spread_volatility: spread.rolling(3).std() per pair
    result["spread_volatility"] = result.groupby("pair_id")["spread"].transform(
        lambda x: x.rolling(3, min_periods=2).std()
    )

    # 5. kalshi_order_flow_imbalance: (buy - sell) / (buy + sell)
    k_denom = result["kalshi_buy_volume"] + result["kalshi_sell_volume"]
    result["kalshi_order_flow_imbalance"] = (
        (result["kalshi_buy_volume"] - result["kalshi_sell_volume"]) / k_denom
    ).where(k_denom != 0, np.nan)

    # 6. polymarket_order_flow_imbalance: same formula for polymarket
    p_denom = result["polymarket_buy_volume"] + result["polymarket_sell_volume"]
    result["polymarket_order_flow_imbalance"] = (
        (result["polymarket_buy_volume"] - result["polymarket_sell_volume"]) / p_denom
    ).where(p_denom != 0, np.nan)

    # --- Richer signal features (2026-04-12) ---
    # Multi-window spread momentum: captures short/medium/long trends.
    # The 3-bar window above catches immediate moves; 6 and 12 catch
    # the trend the models need to distinguish real convergence from noise.
    result["spread_momentum_6"] = result.groupby("pair_id")["spread"].transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    )
    result["spread_momentum_12"] = result.groupby("pair_id")["spread"].transform(
        lambda x: x.rolling(12, min_periods=1).mean()
    )

    # Medium-window volatility (6-bar) — complements the 3-bar version.
    result["spread_volatility_6"] = result.groupby("pair_id")["spread"].transform(
        lambda x: x.rolling(6, min_periods=2).std()
    )

    # Spread z-score: how many standard deviations the current spread is
    # from its recent mean. High absolute z-score = anomalous spread =
    # potential entry signal. Uses 6-bar window for stability.
    result["spread_zscore"] = (
        (result["spread"] - result["spread_momentum_6"])
        / result["spread_volatility_6"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    # Spread range: max - min over recent 6-bar window. Wide range =
    # volatile/uncertain pair; narrow range = stable convergence.
    result["spread_range"] = result.groupby("pair_id")["spread"].transform(
        lambda x: x.rolling(6, min_periods=1).max() - x.rolling(6, min_periods=1).min()
    )

    # Liquidity ratio features: asymmetry between platforms signals
    # which side is driving price discovery.
    result["dollar_volume_ratio"] = (
        result["kalshi_dollar_volume"] / result["polymarket_dollar_volume"]
    ).replace([np.inf, -np.inf], np.nan)

    result["trade_count_ratio"] = (
        result["kalshi_trade_count"] / result["polymarket_trade_count"]
    ).replace([np.inf, -np.inf], np.nan)

    # Mid-price: where both platforms roughly agree. Used as a
    # denominator for relative divergence.
    result["mid_price"] = (result["kalshi_close"] + result["polymarket_close"]) / 2.0

    # Price divergence as a percentage of mid-price. A 0.05 spread at
    # mid=0.50 (10%) is more meaningful than 0.05 at mid=0.90 (5.6%).
    result["price_divergence_pct"] = (
        result["spread"].abs() / result["mid_price"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    # ================================================================
    # Quant-grade microstructure features (2026-04-12)
    #
    # From academic market microstructure literature used at firms like
    # Jane Street, Citadel, Two Sigma. Each feature is cited below.
    # All computable from existing OHLCV + buy/sell volume bar data.
    # ================================================================

    # --- Amihud Illiquidity (Amihud 2002) ---
    # |return| / dollar_volume. High = illiquid = prices move a lot
    # per dollar traded. Thin markets have higher Amihud ratios.
    # Computed per platform: if one side is illiquid and the other
    # isn't, that asymmetry is predictive of which side will adjust.
    k_ret = result.groupby("pair_id")["kalshi_close"].pct_change().abs()
    result["kalshi_amihud"] = (
        k_ret / result["kalshi_dollar_volume"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    p_ret = result.groupby("pair_id")["polymarket_close"].pct_change().abs()
    result["polymarket_amihud"] = (
        p_ret / result["polymarket_dollar_volume"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    # --- Corwin-Schultz Spread Estimator (Corwin & Schultz 2012) ---
    # Implied bid-ask spread from high/low prices. The daily high is
    # almost always a buy trade, the low is almost always a sell trade.
    # More efficient than close-to-close estimators because it uses
    # within-bar price action.
    def _corwin_schultz(high: pd.Series, low: pd.Series) -> pd.Series:
        """Corwin-Schultz spread from consecutive H/L bars."""
        ln_hl = np.log(high / low.replace(0, np.nan))
        ln_hl_sq = ln_hl ** 2
        # Beta uses two consecutive bars
        beta = ln_hl_sq + ln_hl_sq.shift(1)
        # Gamma uses the 2-bar high-low range
        h2 = high.rolling(2, min_periods=2).max()
        l2 = low.rolling(2, min_periods=2).min()
        gamma = np.log(h2 / l2.replace(0, np.nan)) ** 2
        denom = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        return spread.clip(lower=0)  # negative estimates → 0

    # Compute per-pair via vectorized ops (no groupby.apply needed
    # since _corwin_schultz uses only rolling ops that don't cross
    # pair boundaries when the DataFrame is sorted by pair_id).
    # We sort, compute, then restore original order.
    _sorted = result.sort_values("pair_id")
    _k_cs = _corwin_schultz(_sorted["kalshi_high"], _sorted["kalshi_low"])
    _p_cs = _corwin_schultz(_sorted["polymarket_high"], _sorted["polymarket_low"])
    result.loc[_sorted.index, "kalshi_cs_spread"] = _k_cs.values
    result.loc[_sorted.index, "polymarket_cs_spread"] = _p_cs.values

    # --- Bekker-Parkinson Volatility ---
    # Intrabar volatility from high-low range. More efficient than
    # close-to-close vol because it captures within-bar price action.
    # sigma^2 = (ln(H/L))^2 / (4*ln(2))
    result["kalshi_hl_vol"] = (
        np.log(result["kalshi_high"] / result["kalshi_low"].replace(0, np.nan)) ** 2
        / (4 * np.log(2))
    ).replace([np.inf, -np.inf], np.nan).clip(lower=0).pipe(np.sqrt)

    result["polymarket_hl_vol"] = (
        np.log(result["polymarket_high"] / result["polymarket_low"].replace(0, np.nan)) ** 2
        / (4 * np.log(2))
    ).replace([np.inf, -np.inf], np.nan).clip(lower=0).pipe(np.sqrt)

    # --- Cross-Platform OFI Differential ---
    # When Polymarket shows strong buying but Kalshi doesn't (or vice
    # versa), the uninformed platform is stale. The spread should move
    # toward the informed platform's price. This is the core cross-
    # market arbitrage signal used in the academic literature
    # (Ng, Peng, Tao, Zhou 2025).
    # Use .fillna(0) so that if one side has no OFI data (e.g. Kalshi
    # buy/sell volume is zero in historical data), the differential
    # still captures the available side's signal rather than going NaN.
    _k_ofi = result.get("kalshi_order_flow_imbalance")
    _p_ofi = result.get("polymarket_order_flow_imbalance")
    if _k_ofi is not None and _p_ofi is not None:
        result["ofi_differential"] = _p_ofi.fillna(0) - _k_ofi.fillna(0)
    else:
        result["ofi_differential"] = np.nan

    # --- Kyle's Lambda (Kyle 1985, bar-level adaptation) ---
    # Price impact coefficient: how much price moves per unit of
    # signed order flow. High lambda = informed trading is impactful.
    # Rolling 12-bar OLS: lambda = cov(return, signed_sqrt_dv) / var(signed_sqrt_dv)
    # Kyle's Lambda per platform — computed on sorted data to avoid
    # groupby.apply multi-column issues.
    for _plat in ("kalshi", "polymarket"):
        _close = f"{_plat}_close"
        _buy = f"{_plat}_buy_volume"
        _sell = f"{_plat}_sell_volume"
        _dv = f"{_plat}_dollar_volume"
        _ret = _sorted[_close].pct_change()
        _sign = np.sign(_sorted[_buy] - _sorted[_sell])
        _signed_sqrt = _sign * np.sqrt(_sorted[_dv].clip(lower=0))
        _cov = _ret.rolling(12, min_periods=4).cov(_signed_sqrt)
        _var = _signed_sqrt.rolling(12, min_periods=4).var()
        _lam = (_cov / _var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        result.loc[_sorted.index, f"{_plat}_kyle_lambda"] = _lam.values

    # --- Price Boundary Distance (prediction-market-specific) ---
    # How close the mid-price is to 0 or 1. Contracts near boundaries
    # have near-certain outcomes; spreads there are noise. Contracts
    # near 0.50 have maximum uncertainty and spreads are most
    # informative. Analogous to option "gamma" at the money.
    result["boundary_distance"] = result["mid_price"].clip(0, 1).apply(
        lambda p: min(p, 1.0 - p) if pd.notna(p) else np.nan
    )

    # --- Favorite-Longshot Bias (prediction-market-specific) ---
    # Academic research on Kalshi (Burgi, Deng, Whelan 2026) shows
    # contracts priced <$0.15 are systematically overpriced (negative
    # EV after fees) while contracts priced >$0.70 have positive EV.
    # Encode as a continuous score: negative at extremes, zero in
    # the middle.
    result["longshot_score"] = result["mid_price"].apply(
        lambda p: (0.5 - abs(p - 0.5)) / 0.5 if pd.notna(p) else np.nan
    )

    # --- Roll Spread Estimator (Roll 1984) ---
    # Implied bid-ask spread from serial return covariance. Negative
    # autocorrelation in returns arises from bid-ask bounce.
    # S_roll = 2 * sqrt(max(0, -Cov(r_t, r_{t-1})))
    # Roll spread per platform — vectorized on sorted data.
    for _plat in ("kalshi", "polymarket"):
        _close = _sorted[f"{_plat}_close"]
        _ret = np.log(_close / _close.shift(1))
        _cov = _ret.rolling(6, min_periods=3).cov(_ret.shift(1))
        _roll = 2 * np.sqrt((-_cov).clip(lower=0))
        result.loc[_sorted.index, f"{_plat}_roll_spread"] = _roll.values

    # Ensure column order matches schema, but preserve harness-only
    # columns (group_id, time_idx) when the caller passed them. Phase 3
    # train.parquet carries both; Tier 2 sequence models need group_id
    # for pair-boundary-respecting windowing, and run_baselines's
    # prepare_xy_for_seq breaks if it's silently dropped here.
    _passthrough = [c for c in ("group_id", "time_idx") if c in result.columns]
    return result[OUTPUT_COLUMNS + _passthrough]
