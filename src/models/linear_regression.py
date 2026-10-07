"""Linear Regression baseline wrapping sklearn.linear_model.LinearRegression.

First-class regression baseline per CLAUDE.md: trains on the flat feature
matrix and predicts spread change. Plugs into the shared evaluate() pipeline
via ``BasePredictor``.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.models.base import BasePredictor


class LinearRegressionPredictor(BasePredictor):
    """Ordinary least-squares regression over the full feature matrix.

    The model is trained to predict the per-row spread-change target.
    Calling ``predict`` before ``fit`` raises ``RuntimeError``.
    """

    def __init__(self) -> None:
        self._model = LinearRegression()
        self._fitted = False

    @property
    def name(self) -> str:
        return "Linear Regression"

    def fit(
        self, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> "LinearRegressionPredictor":
        self._model.fit(X_train.values, np.asarray(y_train, dtype=float))
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fit before predict")
        return np.asarray(self._model.predict(X.values), dtype=float)
