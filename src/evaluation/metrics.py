"""Regression metrics for spread prediction model evaluation.

Provides a single entry point, ``compute_regression_metrics``, which every
model tier (regression, recurrent, RL) uses to produce the same set of
numbers for head-to-head comparison.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import numpy as np


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute RMSE, MAE, and directional accuracy.

    Args:
        y_true: Ground-truth spread change values, shape (n,).
        y_pred: Predicted spread change values, shape (n,).

    Returns:
        Dict with keys ``rmse``, ``mae``, ``directional_accuracy``.

        - ``rmse``: sqrt(mean((y_true - y_pred)^2))
        - ``mae``: mean(|y_true - y_pred|)
        - ``directional_accuracy``: fraction of samples where
          sign(y_true) == sign(y_pred). Samples where ``y_true == 0`` are
          excluded (no direction to predict). Returns 0.0 if all truths are
          zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    diff = y_true - y_pred
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    # Directional accuracy: exclude truths with no direction.
    direction_mask = y_true != 0.0
    if direction_mask.sum() == 0:
        directional_accuracy = 0.0
    else:
        correct = np.sign(y_true[direction_mask]) == np.sign(
            y_pred[direction_mask]
        )
        directional_accuracy = float(correct.mean())

    return {
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": directional_accuracy,
    }
