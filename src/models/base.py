"""BasePredictor ABC: the contract every model in this project implements.

Every spread-prediction model -- naive baselines, regression, recurrent
networks, RL policies -- inherits from ``BasePredictor`` and therefore
plugs into the shared evaluation pipeline (regression metrics + profit
simulation) via ``BasePredictor.evaluate``.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.profit_sim import simulate_profit


class BasePredictor(ABC):
    """Abstract base class for all spread-change predictors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name used in results tables."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted spread change values, one per row of ``X``.

        Args:
            X: Feature DataFrame. Concrete models document required columns.

        Returns:
            1-D ndarray of predicted spread changes, length ``len(X)``.
        """
        ...

    def fit(
        self, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> "BasePredictor":
        """Train the model. Default implementation is a no-op.

        Baseline predictors override nothing. Trainable models override
        this method. Returns ``self`` for chaining.
        """
        return self

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        threshold: float = 0.02,
        timestamps: np.ndarray | None = None,
    ) -> dict:
        """Predict on ``X_test`` and compute full evaluation results.

        Combines ``compute_regression_metrics`` and ``simulate_profit`` so
        every model is compared on the same set of numbers.

        Args:
            X_test: Test feature DataFrame.
            y_test: Ground-truth spread change array.
            threshold: Trading threshold for profit simulation.
            timestamps: Per-row timestamps for panel-aware Sharpe.
                When provided, returns are aggregated by timestamp
                before computing the annualized Sharpe ratio.

        Returns:
            Dict containing both regression metrics
            (``rmse``, ``mae``, ``directional_accuracy``) and trading
            metrics (``total_pnl``, ``num_trades``, ``win_rate``,
            ``sharpe_ratio``, ``pnl_series``).
        """
        predictions = self.predict(X_test)
        y_test = np.asarray(y_test, dtype=float)
        metrics = compute_regression_metrics(y_test, predictions)
        profit = simulate_profit(
            predictions, y_test, threshold=threshold, timestamps=timestamps
        )
        return {**metrics, **profit}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save trained model to pickle file.

        Creates parent directories if they do not exist.

        Args:
            path: Destination file path (e.g. ``models/deployed/lr.pkl``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "BasePredictor":
        """Load a trained model from a pickle file.

        Args:
            path: Path to the pickle file.

        Returns:
            The deserialized ``BasePredictor`` subclass instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            TypeError: If the loaded object is not a ``BasePredictor``.
        """
        path = Path(path)
        with open(path, "rb") as f:
            model = pickle.load(f)
        if not isinstance(model, BasePredictor):
            raise TypeError(
                f"Loaded object is {type(model).__name__}, not BasePredictor"
            )
        return model
