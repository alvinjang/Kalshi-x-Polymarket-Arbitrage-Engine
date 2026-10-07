"""EnsemblePredictor: weighted-average ensemble with optional concordance gate.

Formalizes what ``src/live/strategy.py`` does ad-hoc (LR+XGB weighted average
with sign-agreement filter) into a reusable, parameterized, picklable
``BasePredictor`` subclass. Used by the Phase 13 concordance audit and weight
sweep, and by downstream ensemble experiments.

Design contract:
    - Members are NOT pre-fitted. ``fit(X, y)`` trains every member on the
      same data passed in.
    - Weights are normalized inside ``predict()``; call sites pass raw
      weights and the class handles the division by the sum.
    - Concordance mode ``"strict"`` zeros-out rows where any two members
      disagree on sign; ``"none"`` emits the weighted average as-is.
    - Picklability comes for free via ``BasePredictor.save/load``; all
      supported member types (LR, XGB, LSTM/GRU, naive) are pickle-safe.

This module MUST NOT import from ``src/live/`` or ``experiments/`` and MUST
NOT touch ``src/live/strategy.py`` (ENSM-05 safety guard).
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.models.base import BasePredictor
from src.utils.seed import set_all_seeds


class EnsemblePredictor(BasePredictor):
    """Weighted-average ensemble with optional sign-concordance gate.

    Parameters
    ----------
    members
        List of ``BasePredictor`` instances to combine. Must be non-empty.
        Members are fit internally by :meth:`fit`; do NOT pre-fit.
    weights
        Optional list of non-negative floats, one per member. If omitted,
        defaults to uniform weights ``[1.0, 1.0, ..., 1.0]``. Weights are
        normalized to sum to 1 inside :meth:`predict`; call sites pass the
        raw values. Weights summing to zero raise ``ValueError``.
    concordance_mode
        ``"none"`` (default) returns the weighted average unconditionally.
        ``"strict"`` emits ``0.0`` for any row where the member sign
        predictions do not all agree.
    seed
        RNG seed forwarded to :func:`set_all_seeds` at the start of
        :meth:`fit` so that members with stochastic training
        (XGBoost, GRU, LSTM) are reproducible.

    Raises
    ------
    ValueError
        If ``members`` is empty, if ``weights`` length mismatches
        ``members``, if any weight is negative, or if the weights sum
        to zero.
    """

    def __init__(
        self,
        members: List[BasePredictor],
        weights: Optional[List[float]] = None,
        concordance_mode: str = "none",
        seed: int = 42,
    ) -> None:
        if not members:
            raise ValueError("EnsemblePredictor requires at least one member")

        if weights is None:
            weights = [1.0] * len(members)

        if len(weights) != len(members):
            raise ValueError(
                f"weights length {len(weights)} != members length {len(members)}"
            )

        weights_arr = np.asarray(weights, dtype=np.float64)
        if np.any(weights_arr < 0):
            raise ValueError("weights must be non-negative")
        if weights_arr.sum() <= 0:
            raise ValueError("weights must sum to a positive value")

        if concordance_mode not in ("none", "strict"):
            raise ValueError(
                f"concordance_mode must be 'none' or 'strict', got "
                f"{concordance_mode!r}"
            )

        self._members: List[BasePredictor] = list(members)
        self._weights: np.ndarray = weights_arr
        self._concordance_mode: str = concordance_mode
        self._seed: int = int(seed)
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # BasePredictor contract
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        member_names = "+".join(m.name for m in self._members)
        return f"Ensemble({member_names}, {self._concordance_mode})"

    def fit(
        self, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> "EnsemblePredictor":
        """Train every member on the same ``(X_train, y_train)``.

        Seeds all RNGs before training so stochastic members are
        reproducible. Returns ``self`` for chaining.
        """
        set_all_seeds(self._seed)
        for member in self._members:
            member.fit(X_train, y_train)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Produce the ensemble prediction for each row of ``X``.

        Computes the per-member prediction matrix, normalizes the stored
        weights to sum to 1, and returns the weighted column-wise average.
        In ``"strict"`` concordance mode, any row where members disagree
        on sign is emitted as ``0.0``.
        """
        if not self._fitted:
            raise RuntimeError("EnsemblePredictor must be fit before predict")

        # Stack member predictions into (n_members, n_rows)
        preds = np.stack(
            [np.asarray(m.predict(X), dtype=float) for m in self._members]
        )

        # Normalize weights inside predict so call sites pass raw values
        weights = self._weights / self._weights.sum()

        # Weighted average along the member axis -> (n_rows,)
        weighted = (weights[:, None] * preds).sum(axis=0)

        if self._concordance_mode == "strict":
            signs = np.sign(preds)
            # All members agree with member 0's sign, per row
            agree = np.all(signs == signs[[0], :], axis=0)
            return np.where(agree, weighted, 0.0)

        return weighted
