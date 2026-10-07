"""Shared test fixtures for model tests.

Matches the Phase 3 matched-pairs schema (``polymarket_*`` column naming,
4-hour bars, VWAP as primary price, ``spread`` column precomputed).
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """20-row DataFrame with the columns baselines need.

    Columns follow the Phase 3 matched-pairs schema:
      - ``spread`` (kalshi_vwap - polymarket_vwap, precomputed)
      - ``kalshi_vwap``, ``polymarket_vwap``
      - ``kalshi_volume``, ``polymarket_volume``
      - ``spread_momentum``
      - ``kalshi_realized_spread``, ``polymarket_realized_spread``
    """
    rng = np.random.default_rng(42)
    kalshi_vwap = np.linspace(0.40, 0.60, 20) + rng.normal(0, 0.01, 20)
    polymarket_vwap = kalshi_vwap + rng.normal(0, 0.03, 20)  # noisy offset
    spread = kalshi_vwap - polymarket_vwap
    # Ensure we have a mix of positive and negative spreads
    spread[0:5] = [0.05, -0.03, 0.02, -0.04, 0.06]
    spread[5:10] = [-0.01, 0.03, -0.05, 0.04, -0.02]
    kalshi_vwap = polymarket_vwap + spread  # recompute to keep identity

    return pd.DataFrame(
        {
            "spread": spread,
            "kalshi_vwap": kalshi_vwap,
            "polymarket_vwap": polymarket_vwap,
            "kalshi_volume": rng.integers(50, 500, 20).astype(float),
            "polymarket_volume": rng.integers(50, 500, 20).astype(float),
            "spread_momentum": rng.normal(0, 0.01, 20),
            "kalshi_realized_spread": rng.normal(0, 0.005, 20),
            "polymarket_realized_spread": rng.normal(0, 0.005, 20),
        }
    )


@pytest.fixture
def sample_targets(sample_features) -> np.ndarray:
    """Realized spread-change targets for ``sample_features``.

    Simulates partial reversion (so both naive and volume baselines make
    sensible, non-trivial predictions).
    """
    rng = np.random.default_rng(7)
    return (-0.6 * sample_features["spread"].values
            + rng.normal(0, 0.01, len(sample_features)))
