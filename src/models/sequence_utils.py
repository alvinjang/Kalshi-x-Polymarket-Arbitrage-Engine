"""Shared utilities for sequence (recurrent) models.

Provides windowing, early stopping, reproducibility seeding, device
selection, and StandardScaler helpers used by both GRU and LSTM models.

Exports:
    create_sequences   -- sliding-window builder respecting pair_id boundaries
    EarlyStopping      -- patience-based training stopper
    set_seed           -- seeds numpy + torch for reproducibility
    get_device         -- auto-selects CUDA or CPU
    fit_feature_scaler -- fits StandardScaler with bool casting + zero-variance guard
    apply_feature_scaler -- applies a fitted scaler with bool casting
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.utils.seed import set_all_seeds


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
    group_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create lookback windows from ``X`` and aligned targets from ``y``.

    Windows that would cross group (pair_id) boundaries are dropped.
    Groups with fewer than ``lookback`` rows contribute zero windows.

    Args:
        X: Feature array, shape ``(n_rows, n_features)``.  May also be a
            :class:`pd.DataFrame` (converted internally).
        y: Target array, shape ``(n_rows,)``.
        lookback: Number of time-steps per window.
        group_ids: Group label per row, shape ``(n_rows,)``.

    Returns:
        ``(X_seq, y_seq)`` where ``X_seq`` has shape
        ``(n_windows, lookback, n_features)`` and ``y_seq`` has shape
        ``(n_windows,)``.  ``y_seq[i]`` is the target aligned with the
        **last** row of window *i*.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    group_ids = np.asarray(group_ids)

    # Preserve first-occurrence order of groups
    seen: OrderedDict[int, None] = OrderedDict()
    for gid in group_ids:
        seen.setdefault(gid, None)

    X_windows: list[np.ndarray] = []
    y_targets: list[float] = []

    for gid in seen:
        mask = group_ids == gid
        X_group = X[mask]
        y_group = y[mask]
        n = len(X_group)

        if n < lookback:
            continue

        for i in range(lookback - 1, n):
            window = X_group[i - lookback + 1 : i + 1]
            X_windows.append(window)
            y_targets.append(y_group[i])

    if len(X_windows) == 0:
        n_features = X.shape[1] if X.ndim == 2 else 1
        return (
            np.empty((0, lookback, n_features), dtype=float),
            np.empty((0,), dtype=float),
        )

    X_seq = np.stack(X_windows, axis=0)
    y_seq = np.array(y_targets, dtype=float)
    return X_seq, y_seq


class EarlyStopping:
    """Patience-based early stopping on validation loss.

    Tracks the best validation loss seen so far. After ``patience``
    consecutive epochs without improvement exceeding ``min_delta``,
    :meth:`step` returns ``True``.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0

    def step(self, val_loss: float) -> bool:
        """Record a validation loss and return whether to stop.

        Returns:
            ``True`` if training should stop (patience exhausted).
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def set_seed(seed: int) -> None:
    """Backward-compatible wrapper. Delegates to src/utils/seed.py.

    Previously seeded numpy + torch directly; now delegates to
    ``set_all_seeds`` which covers all 9 RNG sources including Python
    ``random``, PYTHONHASHSEED, and torch.use_deterministic_algorithms.
    """
    set_all_seeds(seed)


def get_device() -> torch.device:
    """Return a CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_feature_scaler(
    X: pd.DataFrame,
    bool_cols: list[str],
) -> StandardScaler:
    """Fit a :class:`StandardScaler` on ``X`` with safety guards.

    1. Casts ``bool_cols`` to float ``{0.0, 1.0}``.
    2. **Rejects zero-variance columns** with a :class:`ValueError`
       listing the offending names (prevents silent NaN poisoning).
    3. Fits and returns the scaler.

    Args:
        X: Feature DataFrame.
        bool_cols: Column names containing bool values to cast to float.

    Returns:
        Fitted :class:`StandardScaler`.

    Raises:
        ValueError: If any column has zero variance (std == 0).
    """
    X = X.copy()
    for col in bool_cols:
        X[col] = X[col].astype(float)

    # Zero-variance safety guard
    std = X.std(axis=0).to_numpy()
    if (std == 0).any():
        bad_cols = X.columns[std == 0].tolist()
        raise ValueError(
            "fit_feature_scaler: zero-variance columns detected "
            f"(would produce NaN after scaling): {bad_cols}"
        )

    scaler = StandardScaler()
    scaler.fit(X.values)
    return scaler


def apply_feature_scaler(
    X: pd.DataFrame,
    scaler: StandardScaler,
    bool_cols: list[str],
) -> np.ndarray:
    """Transform ``X`` using a previously fitted scaler.

    Casts ``bool_cols`` to float before applying the transform.

    Args:
        X: Feature DataFrame (same columns as fit).
        scaler: A fitted :class:`StandardScaler`.
        bool_cols: Column names containing bool values to cast to float.

    Returns:
        Scaled array of shape ``(n_rows, n_features)``.
    """
    X = X.copy()
    for col in bool_cols:
        X[col] = X[col].astype(float)
    return scaler.transform(X.values)
