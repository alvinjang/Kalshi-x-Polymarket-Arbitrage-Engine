"""Analyze PMXT archive data to see how many of our 3079 matched pairs have trade data.

Run after extracting the PMXT archive:
    tar -I zstd -xf data/pmxt/data.tar.zst -C data/pmxt/
    python scripts/analyze_pmxt.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np


PMXT_DIR = Path("data/pmxt/data")
PAIRS_FILE = Path("data/processed/all_pairs_v2.json")


def load_pmxt_markets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Kalshi and Polymarket market metadata from PMXT archive."""
    kalshi_path = PMXT_DIR / "kalshi" / "markets" / "markets.parquet"
    poly_path = PMXT_DIR / "polymarket" / "markets" / "markets.parquet"

    kalshi_markets = pd.read_parquet(kalshi_path) if kalshi_path.exists() else pd.DataFrame()
    poly_markets = pd.read_parquet(poly_path) if poly_path.exists() else pd.DataFrame()

    return kalshi_markets, poly_markets


def load_pmxt_trades() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load trade data (just metadata — row counts, ticker coverage)."""
    kalshi_path = PMXT_DIR / "kalshi" / "trades" / "trades.parquet"
    poly_path = PMXT_DIR / "polymarket" / "trades" / "trades.parquet"

    # Read just the ticker/market_id columns to check coverage without loading all data
    kalshi_trades = pd.DataFrame()
    poly_trades = pd.DataFrame()

    if kalshi_path.exists():
        # Read only the ticker column to save memory
        kalshi_trades = pd.read_parquet(kalshi_path, columns=["ticker"])
        print(f"Kalshi trades: {len(kalshi_trades):,} rows")
        print(f"  Unique tickers: {kalshi_trades['ticker'].nunique():,}")

    if poly_path.exists():
        # Polymarket trades use _contract as market ID
        try:
            poly_trades = pd.read_parquet(poly_path, columns=["_contract"])
            print(f"Polymarket trades: {len(poly_trades):,} rows")
            print(f"  Unique contracts: {poly_trades['_contract'].nunique():,}")
        except Exception:
            # Try alternative column names
            poly_trades = pd.read_parquet(poly_path)
            print(f"Polymarket trades: {len(poly_trades):,} rows")
            print(f"  Columns: {list(poly_trades.columns)}")

    return kalshi_trades, poly_trades


def check_pair_coverage():
    """Check how many of our 3079 matched pairs have data in PMXT."""
    # Load our matched pairs
    with open(PAIRS_FILE) as f:
        pairs = json.load(f)

    print(f"\n=== Our Matched Pairs: {len(pairs)} ===")

    # Load PMXT markets
    kalshi_markets, poly_markets = load_pmxt_markets()
    print(f"\n=== PMXT Markets ===")
    print(f"Kalshi markets: {len(kalshi_markets):,}")
    if len(kalshi_markets) > 0:
        print(f"  Columns: {list(kalshi_markets.columns)}")
        print(f"  Sample tickers: {list(kalshi_markets['ticker'].head(5))}")

    print(f"Polymarket markets: {len(poly_markets):,}")
    if len(poly_markets) > 0:
        print(f"  Columns: {list(poly_markets.columns)}")
        if 'question' in poly_markets.columns:
            print(f"  Sample questions: {list(poly_markets['question'].head(3))}")

    # Check overlap with our pairs
    our_kalshi_ids = set(p['kalshi_market_id'] for p in pairs)
    our_poly_ids = set(p['polymarket_market_id'] for p in pairs)

    if len(kalshi_markets) > 0 and 'ticker' in kalshi_markets.columns:
        pmxt_kalshi_ids = set(kalshi_markets['ticker'].unique())
        kalshi_overlap = our_kalshi_ids & pmxt_kalshi_ids
        print(f"\n=== Kalshi Market Overlap ===")
        print(f"  Our Kalshi markets: {len(our_kalshi_ids)}")
        print(f"  PMXT Kalshi markets: {len(pmxt_kalshi_ids)}")
        print(f"  Overlap: {len(kalshi_overlap)} ({len(kalshi_overlap)/len(our_kalshi_ids)*100:.1f}%)")

    if len(poly_markets) > 0:
        # Polymarket IDs might match on 'id' or 'condition_id'
        for col in ['id', 'condition_id', 'slug']:
            if col in poly_markets.columns:
                pmxt_poly_ids = set(poly_markets[col].unique())
                poly_overlap = our_poly_ids & pmxt_poly_ids
                print(f"\n=== Polymarket Market Overlap (matching on '{col}') ===")
                print(f"  Our Polymarket markets: {len(our_poly_ids)}")
                print(f"  PMXT Polymarket markets: {len(pmxt_poly_ids)}")
                print(f"  Overlap: {len(poly_overlap)} ({len(poly_overlap)/len(our_poly_ids)*100:.1f}%)")

    # Check trade data coverage
    print("\n=== Trade Data Coverage ===")
    kalshi_trades, poly_trades = load_pmxt_trades()

    if len(kalshi_trades) > 0:
        pmxt_traded_kalshi = set(kalshi_trades['ticker'].unique())
        kalshi_trade_overlap = our_kalshi_ids & pmxt_traded_kalshi
        print(f"  Kalshi pairs with PMXT trade data: {len(kalshi_trade_overlap)} / {len(our_kalshi_ids)}")

        # For overlapping pairs, how many trades?
        if kalshi_trade_overlap:
            overlap_trades = kalshi_trades[kalshi_trades['ticker'].isin(kalshi_trade_overlap)]
            trades_per_market = overlap_trades.groupby('ticker').size()
            print(f"  Trades per matched market: min={trades_per_market.min()}, "
                  f"median={trades_per_market.median():.0f}, max={trades_per_market.max()}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: PMXT Data Expansion Potential")
    print("="*60)
    print(f"  Current aligned pairs: 144")
    print(f"  Total matched pairs: {len(pairs)}")
    if len(kalshi_trades) > 0:
        print(f"  Matched pairs with PMXT Kalshi trades: {len(kalshi_trade_overlap)}")
        print(f"  Potential expansion: {len(kalshi_trade_overlap) - 144} new pairs")
        expansion = len(kalshi_trade_overlap) / 144
        print(f"  Expansion factor: {expansion:.1f}x")


if __name__ == "__main__":
    # Check if PMXT data exists
    if not PMXT_DIR.exists():
        print("PMXT data not extracted yet.")
        print("Run: tar -I zstd -xf data/pmxt/data.tar.zst -C data/pmxt/")
        exit(1)

    check_pair_coverage()
