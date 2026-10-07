"""Naive baseline: the spread always closes fully to zero by resolution.

This is the lower-bound reference every other model must beat. It uses
no features beyond the current ``spread`` value and performs no learning.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import BasePredictor


class NaivePredictor(BasePredictor):
    """Predict that the spread reverts fully to zero in the next period.

    Predicted spread change = ``-current_spread``. Requires the feature
    DataFrame to carry a precomputed ``spread`` column (as produced by
    the Phase 3 matched-pairs pipeline).
    """

    @property
    def name(self) -> str:
        return "Naive (Spread Closes)"

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if "spread" not in X.columns:
            raise ValueError(
                "NaivePredictor requires a 'spread' column in X"
            )
        return -X["spread"].to_numpy(dtype=float)
