"""Volume baseline: the higher-volume platform's price is assumed correct.

The predicted spread change is a volume-weighted reversion: the stronger
one platform's volume dominates, the closer the prediction is to full
reversion. When the two platforms have equal volume, the prediction is
a half reversion (no directional signal).
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import BasePredictor


REQUIRED_VOLUME_COLUMNS = ("kalshi_volume", "polymarket_volume")


class VolumePredictor(BasePredictor):
    """Predict spread change as ``-spread * volume_ratio``.

    ``volume_ratio = max(kalshi_volume, polymarket_volume) / total_volume``
    so the ratio lies in [0.5, 1.0]. When one platform dominates, the
    prediction approaches full reversion; when volumes are equal, the
    prediction is half reversion.

    Requires ``spread``, ``kalshi_volume``, and ``polymarket_volume``
    columns in the feature DataFrame.
    """

    @property
    def name(self) -> str:
        return "Volume (Higher Volume Correct)"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "spread" not in X.columns:
            raise ValueError(
                "VolumePredictor requires a 'spread' column in X"
            )
        missing = [c for c in REQUIRED_VOLUME_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(
                f"VolumePredictor requires volume columns: {missing}"
            )

        spread = X["spread"].to_numpy(dtype=float)
        kalshi_volume = X["kalshi_volume"].to_numpy(dtype=float)
        polymarket_volume = X["polymarket_volume"].to_numpy(dtype=float)

        total_volume = kalshi_volume + polymarket_volume
        max_volume = np.maximum(kalshi_volume, polymarket_volume)

        # Guard against divide-by-zero when both volumes are zero.
        with np.errstate(divide="ignore", invalid="ignore"):
            volume_ratio = np.where(
                total_volume > 0.0, max_volume / total_volume, 0.0
            )

        return -spread * volume_ratio
