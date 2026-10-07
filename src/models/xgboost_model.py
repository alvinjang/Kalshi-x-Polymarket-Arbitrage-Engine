"""XGBoost baseline wrapping xgboost.XGBRegressor.

Strongest tabular baseline and likely best Tier 1 performer. Trains on the
flat feature matrix and predicts spread change. Accepts hyperparameters
(n_estimators, max_depth, learning_rate, random_state) via the constructor
so experiment scripts can override defaults.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.models.base import BasePredictor


class XGBoostPredictor(BasePredictor):
    """Gradient-boosted trees over the full feature matrix.

    Defaults: ``n_estimators=100``, ``max_depth=6``, ``learning_rate=0.1``,
    ``random_state=42``. Calling ``predict`` before ``fit`` raises
    ``RuntimeError``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self._model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbosity=0,
        )
        self._fitted = False

    @property
    def name(self) -> str:
        return "XGBoost"

    def fit(
        self, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> "XGBoostPredictor":
        self._model.fit(X_train.values, np.asarray(y_train, dtype=float))
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fit before predict")
        return np.asarray(self._model.predict(X.values), dtype=float)
