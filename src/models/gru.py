"""GRU recurrent model for spread-change prediction (Tier 2 baseline).

Wraps a single-layer GRU with input dropout and a linear output head.
Inherits from ``BasePredictor`` so it plugs directly into the shared
evaluation pipeline (regression metrics + profit simulation).

Hyperparameters are locked per CONTEXT.md decisions D6 (architecture)
and D8 (training protocol).  The model caches training data during
``fit()`` and builds lookback windows internally during ``predict()``
using warm-up stitching so it returns one prediction per input row
(row-aligned with Tier 1 for direct comparison).

Exports:
    GRUPredictor -- Tier 2 recurrent baseline for spread-change prediction
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.base import BasePredictor
from src.models.sequence_utils import (
    create_sequences,
    EarlyStopping,
    set_seed,
    get_device,
    fit_feature_scaler,
    apply_feature_scaler,
)


# ---------------------------------------------------------------------------
# Internal PyTorch module
# ---------------------------------------------------------------------------

class _GRUModule(nn.Module):
    """Single-layer GRU with input dropout and linear output head.

    Architecture (from CONTEXT.md D6):
        input  -> Dropout(p) -> GRU(hidden_size, 1 layer) -> Linear(hidden_size, 1)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, lookback, n_features)``.

        Returns:
            Predictions of shape ``(batch,)``.
        """
        x = self.dropout(x)
        h, _ = self.gru(x)
        last = h[:, -1, :]  # use last timestep hidden state
        return self.out(last).squeeze(-1)  # (batch,)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class GRUPredictor(BasePredictor):
    """GRU-based spread-change predictor (Tier 2 recurrent baseline).

    Hyperparameters default to the values locked in CONTEXT.md D6 + D8:
      - hidden_size=64, num_layers=1, dropout=0.3
      - AdamW lr=1e-3, weight_decay=1e-4
      - batch_size=64, max_epochs=100, patience=10
      - lookback=6 (24 hours at 4-hour bars), grad_clip=1.0

    ``fit()`` and ``predict()`` require a ``group_id`` column in ``X``
    for pair-boundary-respecting windowing.  A ``ValueError`` is raised
    immediately if the column is missing.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.3,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        min_delta: float = 1e-4,
        lookback: int = 6,
        grad_clip: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._min_delta = min_delta
        self._lookback = lookback
        self._grad_clip = grad_clip
        self._random_state = random_state

        self._fitted = False
        self._model: _GRUModule | None = None
        self._scaler = None
        self._cached_train: dict | None = None
        self._padded_pairs: list = []

    @property
    def name(self) -> str:
        return "GRU"

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> "GRUPredictor":
        """Train the GRU on ``X_train`` / ``y_train``.

        Args:
            X_train: Feature DataFrame.  **Must** contain a ``group_id``
                column for pair-boundary windowing.
            y_train: Target array of spread changes, shape ``(n,)``.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If ``group_id`` column is missing from *X_train*.
        """
        # --- group_id guard (MUST be first step) ---
        if "group_id" not in X_train.columns:
            raise ValueError(
                "GRUPredictor.fit requires 'group_id' column in X_train for "
                "pair-boundary-respecting windowing. Pass df[feature_cols + ['group_id']]."
            )

        set_seed(self._random_state)
        device = get_device()

        y_train = np.asarray(y_train, dtype=float)

        # Extract group_ids and feature columns
        group_ids = X_train["group_id"].to_numpy()
        feature_cols = [c for c in X_train.columns if c != "group_id"]

        # Identify bool columns by dtype
        bool_cols = [c for c in feature_cols if X_train[c].dtype == bool]

        # Fit and cache scaler on train features
        self._scaler = fit_feature_scaler(X_train[feature_cols], bool_cols)
        X_scaled = apply_feature_scaler(X_train[feature_cols], self._scaler, bool_cols)

        # --- 90/10 within-pair chronological val split ---
        unique_gids = np.unique(group_ids)
        train_mask = np.ones(len(X_train), dtype=bool)
        val_mask = np.zeros(len(X_train), dtype=bool)

        for gid in unique_gids:
            gid_indices = np.where(group_ids == gid)[0]
            n_gid = len(gid_indices)
            n_val = max(1, int(n_gid * 0.1))
            val_start = n_gid - n_val
            val_indices = gid_indices[val_start:]
            train_mask[val_indices] = False
            val_mask[val_indices] = True

        X_scaled_train = X_scaled[train_mask]
        y_train_split = y_train[train_mask]
        group_ids_train = group_ids[train_mask]

        X_scaled_val = X_scaled[val_mask]
        y_val_split = y_train[val_mask]
        group_ids_val = group_ids[val_mask]

        # Build windows
        X_seq_train, y_seq_train = create_sequences(
            X_scaled_train, y_train_split, self._lookback, group_ids_train
        )
        X_seq_val, y_seq_val = create_sequences(
            X_scaled_val, y_val_split, self._lookback, group_ids_val
        )

        has_val = len(X_seq_val) > 0

        # Convert to tensors (use from_numpy to avoid segfault with
        # torch.tensor on large float64 arrays on some platforms)
        X_train_t = torch.from_numpy(
            np.ascontiguousarray(X_seq_train, dtype=np.float32)
        ).to(device)
        y_train_t = torch.from_numpy(
            np.ascontiguousarray(y_seq_train, dtype=np.float32)
        ).to(device)
        if has_val:
            X_val_t = torch.from_numpy(
                np.ascontiguousarray(X_seq_val, dtype=np.float32)
            ).to(device)
            y_val_t = torch.from_numpy(
                np.ascontiguousarray(y_seq_val, dtype=np.float32)
            ).to(device)

        # Initialize model
        n_features = len(feature_cols)
        self._model = _GRUModule(
            n_features=n_features,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.MSELoss()
        stopper = EarlyStopping(patience=self._patience, min_delta=self._min_delta)

        # --- Training loop ---
        n_train = len(X_train_t)
        for epoch in range(self._max_epochs):
            self._model.train()

            # Shuffle train indices
            perm = torch.randperm(n_train, device=device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, self._batch_size):
                idx = perm[start : start + self._batch_size]
                xb = X_train_t[idx]
                yb = y_train_t[idx]

                optimizer.zero_grad()
                pred = self._model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), max_norm=self._grad_clip
                )
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # Validation
            if has_val:
                self._model.eval()
                with torch.no_grad():
                    val_pred = self._model(X_val_t)
                    val_loss = criterion(val_pred, y_val_t).item()
                scheduler.step(val_loss)
                print(
                    f"[GRU] epoch {epoch + 1}/{self._max_epochs}  "
                    f"train_loss={avg_train_loss:.6f}  val_loss={val_loss:.6f}"
                )
                if stopper.step(val_loss):
                    print(f"[GRU] early stopping at epoch {epoch + 1}")
                    break
            else:
                print(
                    f"[GRU] epoch {epoch + 1}/{self._max_epochs}  "
                    f"train_loss={avg_train_loss:.6f}  (no val data)"
                )

        # Cache training data for warm-up stitching during predict()
        self._cached_train = {
            "X_scaled": X_scaled,
            "group_ids": group_ids,
            "feature_cols": feature_cols,
            "bool_cols": bool_cols,
        }
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict spread changes for every row in ``X``.

        Uses warm-up stitching: for each group_id in ``X``, the cached
        training rows for that group (if any) are prepended so the first
        test rows also get full-length lookback windows.  If total rows
        available (train + X) are still less than ``lookback``, the
        front is padded by repeating the first row and the event is
        logged in ``self._padded_pairs``.

        Args:
            X: Feature DataFrame.  **Must** contain a ``group_id``
                column.

        Returns:
            1-D ndarray of predictions, shape ``(len(X),)``.

        Raises:
            RuntimeError: If model has not been fit.
            ValueError: If ``group_id`` column is missing.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before predict")

        if "group_id" not in X.columns:
            raise ValueError(
                "GRUPredictor.predict requires 'group_id' column in X for "
                "pair-boundary-respecting windowing."
            )

        device = get_device()
        feature_cols = self._cached_train["feature_cols"]
        bool_cols = self._cached_train["bool_cols"]
        cached_X_scaled = self._cached_train["X_scaled"]
        cached_group_ids = self._cached_train["group_ids"]

        # Scale test features using the TRAIN-fitted scaler (no re-fit)
        X_scaled = apply_feature_scaler(X[feature_cols], self._scaler, bool_cols)

        group_ids_test = X["group_id"].to_numpy()
        n_test = len(X)

        # Build one lookback window per test row using warm-up stitching
        all_windows: list[np.ndarray] = []

        # Process by group to respect pair boundaries
        # Preserve row order: iterate over unique gids in order of first appearance.
        # group_ids are accepted as-is (int OR str): content-addressed pair_ids
        # like "kxwti26apr27t9099-0x2c50e53f" are now used as group_ids in the
        # live pipeline, so we cannot assume integer keys.
        seen_gids: list = []
        gid_test_indices: dict = {}
        for i, gid in enumerate(group_ids_test):
            if gid not in gid_test_indices:
                seen_gids.append(gid)
                gid_test_indices[gid] = []
            gid_test_indices[gid].append(i)

        # Pre-build output array to fill in order
        predictions = np.zeros(n_test, dtype=float)
        windows_by_row: dict = {}

        for gid in seen_gids:
            test_indices = gid_test_indices[gid]
            test_rows = X_scaled[test_indices]  # (n_test_group, n_features)

            # Get cached train rows for this group (warm-up)
            train_mask = cached_group_ids == gid
            if train_mask.any():
                train_rows = cached_X_scaled[train_mask]
                stitched = np.vstack([train_rows, test_rows])
            else:
                stitched = test_rows

            n_available = len(stitched)

            # Check if padding is needed
            if n_available < self._lookback:
                self._padded_pairs.append(gid)
                print(
                    f"WARN [GRU]: padding applied for group_id={gid}, "
                    f"n_rows_available={n_available}, lookback={self._lookback}"
                )
                # Pad front by repeating first row
                n_pad = self._lookback - n_available
                pad_rows = np.tile(stitched[0:1], (n_pad, 1))
                stitched = np.vstack([pad_rows, stitched])

            # For each test row, extract the window ending at that row
            # stitched has: [train_rows..., test_rows...]
            # The test rows start at offset = len(stitched) - len(test_rows)
            test_start_in_stitched = len(stitched) - len(test_rows)

            for local_idx, global_idx in enumerate(test_indices):
                end_pos = test_start_in_stitched + local_idx + 1
                start_pos = end_pos - self._lookback
                if start_pos < 0:
                    start_pos = 0
                window = stitched[start_pos:end_pos]

                # If window is shorter than lookback (shouldn't happen after padding, but safety)
                if len(window) < self._lookback:
                    n_pad = self._lookback - len(window)
                    pad = np.tile(window[0:1], (n_pad, 1))
                    window = np.vstack([pad, window])

                all_windows.append(window)
                windows_by_row[global_idx] = window

        if self._padded_pairs:
            print(
                f"[GRU] predict: {len(self._padded_pairs)} pair(s) "
                f"required warm-up padding"
            )

        # Batch forward through model
        if len(all_windows) == 0:
            return np.zeros(n_test, dtype=float)

        # Reconstruct windows in original row order
        ordered_windows = []
        for i in range(n_test):
            ordered_windows.append(windows_by_row[i])

        X_windows = np.stack(ordered_windows, axis=0)  # (n_test, lookback, n_features)
        X_tensor = torch.from_numpy(
            np.ascontiguousarray(X_windows, dtype=np.float32)
        ).to(device)

        self._model.eval()
        with torch.no_grad():
            # Process in batches to avoid memory issues
            preds_list = []
            for start in range(0, n_test, self._batch_size):
                batch = X_tensor[start : start + self._batch_size]
                batch_preds = self._model(batch)
                preds_list.append(batch_preds.cpu().numpy())

            predictions = np.concatenate(preds_list, axis=0)

        return predictions.astype(float)
