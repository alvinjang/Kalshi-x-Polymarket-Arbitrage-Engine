# Trade-Based Data Reconstruction — Design Spec

**Date:** 2026-04-04
**Phase:** 2.1 (inserted between Phase 2 Market Matching and Phase 3 Feature Engineering)
**Status:** Approved

## Problem

Kalshi's candlestick API returns null prices for 76/77 economics markets (zero trades, zero bid/ask in hourly candles). Polymarket's CLOB prices-history is slow (4 min/market) and returns zero volume. Both candlestick APIs are unreliable for thinly-traded prediction markets.

The `/markets/trades` endpoints on both platforms work reliably and contain the actual trade data we need.

## Solution

Rebuild the data pipeline to be **trade-first**: pull raw trade records from both platforms, reconstruct OHLCV+VWAP candles at 4-hour granularity, and align cross-platform prices with forward-fill and staleness decay.

## Architecture

```
Raw Trades (Kalshi /markets/trades + Polymarket Data API /trades)
    ↓
Per-Market Trade Parquet (data/raw/{platform}/{market_id}_trades.parquet)
    ↓
4-Hour OHLCV+VWAP Reconstruction (src/data/trade_reconstructor.py)
    ↓
Per-Platform Candle Files (data/processed/candles_{platform}.parquet)
    ↓
Cross-Platform Alignment with Forward-Fill (src/data/aligner.py)
    ↓
Aligned Pairs Dataset (data/processed/aligned_pairs.parquet)
```

## Key Design Decisions

### 1. Granularity: 4-Hour Candles

With ~0.1 trades/hour on Kalshi, most granularities produce empty candles:
- 1-minute: 99.9% empty
- 1-hour: 91% empty
- **4-hour: ~30% fill rate** — the sweet spot
- Daily: 70% fill but only ~30-60 observations per market

Reference: Lopez de Prado — "do not sample at a frequency higher than the information arrival rate."

### 2. VWAP as Primary Price

VWAP (volume-weighted average price) is more robust than Close for thin markets. When only 2-3 trades occur in a 4-hour bar, Close is arbitrary (depends on which trade happened last). VWAP reflects where most volume actually transacted.

`VWAP = sum(price_i * volume_i) / sum(volume_i)`

Both VWAP and OHLC are stored. VWAP used for spread computation, OHLC range used as volatility proxy.

### 3. Forward-Fill with 24-Hour Staleness Limit

For cross-platform alignment:
1. Create unified 4-hour time grid for each pair
2. Map each platform's candles onto grid (NaN for missing periods)
3. Forward-fill each platform independently, max 24 hours (6 bars)
4. After 24h with no new trade, price reverts to NaN (excluded from training)
5. Include `hours_since_last_trade` as a model feature

Reference: Hayashi-Yoshida estimator — use overlapping intervals for asynchronous observations.

### 4. Microstructure Features per Candle

| Column | Type | Description |
|---|---|---|
| `vwap` | float | Volume-weighted average price |
| `open` | float | First trade price in bar |
| `high` | float | Highest trade price |
| `low` | float | Lowest trade price |
| `close` | float | Last trade price |
| `volume` | float | Total contracts traded |
| `trade_count` | int | Number of trades |
| `dollar_volume` | float | sum(price * size) |
| `buy_volume` | float | Volume from buy-side takers |
| `sell_volume` | float | Volume from sell-side takers |
| `realized_spread` | float | mean(ask_hits) - mean(bid_hits) |
| `max_trade_size` | float | Largest single trade |
| `time_since_last_trade` | float | Seconds since previous bar's last trade |
| `has_trade` | bool | Was there a fresh trade in this bar? |

### 5. Trade Data Sources

**Kalshi** (`/markets/trades`):
- Cursor pagination, 200 trades/request
- Fields: `created_time`, `yes_price_dollars`, `count_fp`, `taker_side`
- Rate limit: 18 req/s
- Expected: ~15-30 min for 77 markets

**Polymarket** (Data API `/trades`):
- Offset pagination, 500 trades/request, max 15000 offset
- Fields: `timestamp`, `price`, `size`, `side`
- Rate limit: ~20 req/10s
- Expected: ~10-20 min for 77 markets

### 6. Data Quality Checks

Pre-alignment:
- Price bounds: must be in [0.01, 0.99]
- Minimum 20 trades per market (else exclude)
- Monotonic timestamps
- No duplicate trades

Post-alignment:
- Staleness ratio < 80% per pair (else exclude)
- Spread should be mean-reverting (mean |spread| < 0.3)
- Both prices converge to 0 or 1 near resolution (settlement check)

### 7. Storage

Single parquet file: `data/processed/aligned_pairs.parquet`
- ~77 pairs × ~180 candles × ~25 columns = ~350K cells
- Estimated size: ~500KB-1MB
- Snappy compression, pair_id as categorical

Raw trades stored per-market: `data/raw/{platform}/{market_id}_trades.parquet`

## Components

1. **`src/data/trade_fetcher.py`** — Unified trade fetching for both platforms
2. **`src/data/trade_reconstructor.py`** — Trade → 4-hour OHLCV+VWAP+microstructure
3. **`src/data/aligner.py`** — Cross-platform alignment with forward-fill + staleness
4. **`scripts/rebuild_data.py`** — CLI entry point: fetch → reconstruct → align → validate → save

## Expected Outcome

- **Input:** 77 accepted pairs from `data/processed/accepted_pairs.json`
- **Output:** ~50-70 usable pairs with ~10,000-13,000 aligned 4-hour observations
- **Improvement:** From 1 usable pair (strict hourly matching) to 50-70 pairs (4-hour + forward-fill)

## Impact on Downstream Phases

- **Phase 3 (Feature Engineering):** Must be updated to read `aligned_pairs.parquet` instead of computing alignment itself. The 4-hour granularity and VWAP price change the feature computation.
- **Phase 4 (Baselines):** No change needed — models consume feature matrices regardless of granularity.
- **Experiments:** Window length experiment (6h, 24h, 72h, 7d) now uses 4-hour bars: 1.5, 6, 18, 42 candles per window.
