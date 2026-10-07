"""LSTM predictor for spread-change prediction (Tier 2 recurrent alternative).

Wraps a single-layer unidirectional LSTM with input dropout and a linear
output head.  Structurally identical to GRUPredictor but uses ``nn.LSTM``
with ``hidden_size=32`` (smaller because LSTM has ~40% more parameters per
unit than GRU at the same hidden size).

Training protocol: AdamW + ReduceLROnPlateau + gradient clipping + early
stopping, all controlled by constructor hyperparameters.  Windowing,
scaling, and warm-up stitching are internal -- callers pass flat DataFrames.

Exports:
    LSTMPredictor -- Tier 2 recurrent model implementing BasePredictor
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


class _LSTMModule(nn.Module):
    """Single-layer unidirectional LSTM with input dropout."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        h, _ = self.lstm(x)
        last = h[:, -1, :]
        return self.out(last).squeeze(-1)


# ---------------------------------------------------------------------------
# Public predictor class
# ---------------------------------------------------------------------------


class LSTMPredictor(BasePredictor):
    """Tier 2 LSTM spread-change predictor.

    Hyperparameter defaults match CONTEXT.md decisions D7 (architecture)
    and D8 (training protocol).

    Args:
        hidden_size: LSTM hidden dimension (default 32).
        num_layers: Number of stacked LSTM layers (default 1).
        dropout: Input dropout rate (default 0.3).
        learning_rate: AdamW learning rate (default 1e-3).
        weight_decay: AdamW L2 penalty (default 1e-4).
        batch_size: Mini-batch size for training (default 64).
        max_epochs: Maximum training epochs (default 100).
        patience: Early-stopping patience in epochs (default 10).
        min_delta: Minimum improvement for early stopping (default 1e-4).
        lookback: Sequence window length in bars (default 6).
        grad_clip: Max gradient norm for clipping (default 1.0).
        random_state: Seed for reproducibility (default 42).
    """

    def __init__(
        self,
        hidden_size: int = 32,
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

        self._model: _LSTMModule | None = None
        self._fitted: bool = False
        self._scaler = None
        self._device = get_device()
        self._padded_pairs: list = []

        # Cached training data for warm-up stitching during predict
        self._cached_train: dict | None = None

    # ------------------------------------------------------------------
    # BasePredictor interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "LSTM"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
    ) -> "LSTMPredictor":
        """Train the LSTM on windowed sequences built from *X_train*.

        Args:
            X_train: Feature DataFrame **with** ``group_id`` column.
            y_train: Target array aligned row-by-row with *X_train*.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If ``group_id`` column is missing from *X_train*.
        """
        # --- group_id guard ---
        if "group_id" not in X_train.columns:
            raise ValueError(
                "LSTMPredictor.fit requires 'group_id' column in X_train for "
                "pair-boundary-respecting windowing. Pass df[feature_cols + ['group_id']]."
            )

        set_seed(self._random_state)

        # Separate group_id from features
        group_ids = X_train["group_id"].values
        feature_cols = [c for c in X_train.columns if c != "group_id"]

        # Detect bool columns by dtype
        bool_cols = [
            c for c in feature_cols
            if X_train[c].dtype == bool or X_train[c].dtype == "boolean"
        ]

        # Fit and cache scaler on training features
        self._scaler = fit_feature_scaler(X_train[feature_cols], bool_cols)

        # Scale features
        X_scaled = apply_feature_scaler(X_train[feature_cols], self._scaler, bool_cols)
        y_arr = np.asarray(y_train, dtype=float)

        # Cache training data for warm-up stitching in predict
        self._cached_train = {
            "X_scaled": X_scaled,
            "y": y_arr,
            "group_ids": group_ids,
            "feature_cols": feature_cols,
            "bool_cols": bool_cols,
        }

        # 90/10 within-pair chronological val split
        train_idx: list[int] = []
        val_idx: list[int] = []
        unique_groups = list(dict.fromkeys(group_ids))
        for gid in unique_groups:
            mask = np.where(group_ids == gid)[0]
            n = len(mask)
            split_point = max(1, int(n * 0.9))
            train_idx.extend(mask[:split_point].tolist())
            val_idx.extend(mask[split_point:].tolist())

        X_tr = X_scaled[train_idx]
        y_tr = y_arr[train_idx]
        gids_tr = group_ids[train_idx]

        X_val = X_scaled[val_idx]
        y_val = y_arr[val_idx]
        gids_val = group_ids[val_idx]

        # Build windows
        X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, self._lookback, gids_tr)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, self._lookback, gids_val)

        n_features = X_scaled.shape[1]

        # Build model
        self._model = _LSTMModule(
            n_features=n_features,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._dropout,
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self._learning_rate,
            weight_decay=self._weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        criterion = nn.MSELoss()
        early_stop = EarlyStopping(
            patience=self._patience, min_delta=self._min_delta
        )

        # Convert to tensors (use from_numpy to avoid segfault with
        # torch.tensor on large float64 arrays on some platforms)
        X_tr_t = torch.from_numpy(
            np.ascontiguousarray(X_tr_seq, dtype=np.float32)
        ).to(self._device)
        y_tr_t = torch.from_numpy(
            np.ascontiguousarray(y_tr_seq, dtype=np.float32)
        ).to(self._device)

        has_val = len(X_val_seq) > 0
        if has_val:
            X_val_t = torch.from_numpy(
                np.ascontiguousarray(X_val_seq, dtype=np.float32)
            ).to(self._device)
            y_val_t = torch.from_numpy(
                np.ascontiguousarray(y_val_seq, dtype=np.float32)
            ).to(self._device)

        n_train = len(X_tr_t)

        for epoch in range(self._max_epochs):
            self._model.train()

            # Shuffle training data
            perm = torch.randperm(n_train, device=self._device)
            X_tr_t = X_tr_t[perm]
            y_tr_t = y_tr_t[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, self._batch_size):
                end = min(start + self._batch_size, n_train)
                X_batch = X_tr_t[start:end]
                y_batch = y_tr_t[start:end]

                optimizer.zero_grad()
                preds = self._model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(
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
                    val_preds = self._model(X_val_t)
                    val_loss = criterion(val_preds, y_val_t).item()
            else:
                val_loss = avg_train_loss

            print(
                f"[LSTM] epoch {epoch + 1}/{self._max_epochs}  "
                f"train_loss={avg_train_loss:.6f}  val_loss={val_loss:.6f}"
            )

            scheduler.step(val_loss)

            if early_stop.step(val_loss):
                print(f"[LSTM] early stopping at epoch {epoch + 1}")
                break

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict spread changes for each row of *X*.

        Uses warm-up stitching: for each group, concatenates cached
        training data with *X* rows to form complete lookback windows,
        then returns one prediction per input row.

        Args:
            X: Feature DataFrame **with** ``group_id`` column.

        Returns:
            1-D ndarray of shape ``(len(X),)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``group_id`` column is missing from *X*.
        """
        if not self._fitted:
            raise RuntimeError(
                "LSTMPredictor.predict called before fit. "
                "Train the model first."
            )

        # --- group_id guard ---
        if "group_id" not in X.columns:
            raise ValueError(
                "LSTMPredictor.predict requires 'group_id' column in X for "
                "pair-boundary-respecting windowing."
            )

        group_ids = X["group_id"].values
        feature_cols = self._cached_train["feature_cols"]
        bool_cols = self._cached_train["bool_cols"]

        # Reuse train-fit scaler (no re-fit on test)
        X_scaled = apply_feature_scaler(X[feature_cols], self._scaler, bool_cols)

        # Reset padded pairs tracking for this predict call
        self._padded_pairs = []

        # Warm-up stitching: build per-group predictions
        predictions = np.zeros(len(X), dtype=float)
        unique_groups = list(dict.fromkeys(group_ids))

        cached_gids = self._cached_train["group_ids"]
        cached_X = self._cached_train["X_scaled"]

        self._model.eval()

        for gid in unique_groups:
            test_mask = np.where(group_ids == gid)[0]
            test_rows = X_scaled[test_mask]
            n_test = len(test_rows)

            # Get cached training rows for this group (warm-up context)
            train_mask = np.where(cached_gids == gid)[0]
            if len(train_mask) > 0:
                train_rows = cached_X[train_mask]
                stitched = np.concatenate([train_rows, test_rows], axis=0)
            else:
                stitched = test_rows

            n_stitched = len(stitched)

            # Padding: if stitched sequence is shorter than lookback
            if n_stitched < self._lookback:
                pad_needed = self._lookback - n_stitched
                pad_row = stitched[0:1]  # repeat first row
                padding = np.repeat(pad_row, pad_needed, axis=0)
                stitched = np.concatenate([padding, stitched], axis=0)
                self._padded_pairs.append(gid)
                print(
                    f"WARN [LSTM]: padding applied for group_id={gid}, "
                    f"n_rows_available={n_stitched}, lookback={self._lookback}"
                )
                n_stitched = len(stitched)

            # Build windows for the test rows only
            # The last n_test rows in stitched correspond to test rows
            offset = n_stitched - n_test

            group_preds = []
            with torch.no_grad():
                for i in range(n_test):
                    # Window ends at offset + i (inclusive)
                    end_idx = offset + i + 1
                    start_idx = end_idx - self._lookback

                    if start_idx < 0:
                        # Need padding for this individual row
                        available = stitched[:end_idx]
                        pad_needed = self._lookback - len(available)
                        pad_row = available[0:1]
                        padding = np.repeat(pad_row, pad_needed, axis=0)
                        window = np.concatenate([padding, available], axis=0)
                        if gid not in self._padded_pairs:
                            self._padded_pairs.append(gid)
                            print(
                                f"WARN [LSTM]: padding applied for group_id={gid}, "
                                f"n_rows_available={end_idx}, lookback={self._lookback}"
                            )
                    else:
                        window = stitched[start_idx:end_idx]

                    window_t = torch.from_numpy(
                        np.ascontiguousarray(
                            window[np.newaxis, :, :], dtype=np.float32
                        )
                    ).to(self._device)
                    pred = self._model(window_t).cpu().numpy().item()
                    group_preds.append(pred)

            predictions[test_mask] = group_preds

        if self._padded_pairs:
            print(
                f"[LSTM] predict: {len(self._padded_pairs)} pair(s) "
                "required warm-up padding"
            )

        return predictions
