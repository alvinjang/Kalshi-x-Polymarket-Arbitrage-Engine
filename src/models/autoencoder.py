"""Autoencoder anomaly detector for spread trading signal filtering (MOD-09).

Point-based deterministic autoencoder that reconstructs 31-dimensional
feature vectors through a 4-dimensional bottleneck.  Bars with high
reconstruction error (above the 95th percentile of training errors)
are flagged as anomalous.

This is NOT a BasePredictor -- it is a signal filter used by
PPO-Filtered (MOD-10) to focus RL trading on anomalous spread patterns.

Architecture (from CONTEXT.md):
    Encoder: Linear(31->16, ReLU, BN) -> Linear(16->8, ReLU, BN) -> Linear(8->4)
    Decoder: Linear(4->8, ReLU, BN) -> Linear(8->16, ReLU, BN) -> Linear(16->31)

Exports:
    AnomalyDetectorAutoencoder -- anomaly detection via reconstruction error
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.sequence_utils import (
    EarlyStopping,
    fit_feature_scaler,
    apply_feature_scaler,
    set_seed,
    get_device,
)


# ---------------------------------------------------------------------------
# Internal PyTorch module
# ---------------------------------------------------------------------------

class _AutoencoderModule(nn.Module):
    """Symmetric autoencoder with BatchNorm and ReLU activations.

    Architecture:
        Encoder: Linear(input_dim->16, ReLU, BN) -> Linear(16->8, ReLU, BN) -> Linear(8->bottleneck_dim)
        Decoder: Linear(bottleneck_dim->8, ReLU, BN) -> Linear(8->16, ReLU, BN) -> Linear(16->input_dim)

    Forward returns both the reconstruction and the bottleneck activations
    (for potential downstream use).
    """

    def __init__(self, input_dim: int = 31, bottleneck_dim: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 8),
            nn.ReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, input_dim),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (reconstruction, bottleneck).

        Args:
            x: Input tensor of shape ``(batch, input_dim)``.

        Returns:
            Tuple of ``(reconstruction, bottleneck)`` where reconstruction
            has the same shape as input and bottleneck has shape
            ``(batch, bottleneck_dim)``.
        """
        bottleneck = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        return reconstruction, bottleneck


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class AnomalyDetectorAutoencoder:
    """Autoencoder-based anomaly detector for spread trading signal filtering.

    Trains on normal spread feature vectors and flags anomalous bars via
    reconstruction error thresholding.  NOT a BasePredictor -- this is a
    utility/filter component, not a spread predictor.

    Per CONTEXT.md locked decisions:
      - Architecture: 31->16->8->4->8->16->31 with BN + ReLU
      - Loss: MSE(input, reconstruction)
      - Optimizer: Adam, lr=1e-3
      - Batch size: 32, max epochs: 200, early stopping patience=20
      - Threshold: 95th percentile of training reconstruction errors

    Args:
        input_dim: Input feature dimensionality (default 31).
        bottleneck_dim: Bottleneck layer size (default 4).
        lr: Adam learning rate (default 1e-3).
        batch_size: Training batch size (default 32).
        max_epochs: Maximum training epochs (default 200).
        patience: Early stopping patience (default 20).
        min_delta: Early stopping minimum improvement (default 1e-4).
        random_state: Seed for reproducibility (default 42).
    """

    def __init__(
        self,
        input_dim: int = 31,
        bottleneck_dim: int = 4,
        lr: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 200,
        patience: int = 20,
        min_delta: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        self._input_dim = input_dim
        self._bottleneck_dim = bottleneck_dim
        self._lr = lr
        self._batch_size = batch_size
        self._max_epochs = max_epochs
        self._patience = patience
        self._min_delta = min_delta
        self._random_state = random_state

        self._model: _AutoencoderModule | None = None
        self._scaler = None
        self._bool_cols: list[str] = []
        self._feature_cols: list[str] = []
        self.threshold_: float | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        feature_cols: list[str],
    ) -> "AnomalyDetectorAutoencoder":
        """Train the autoencoder on normal feature patterns.

        Performs a 90/10 chronological validation split (last 10% of rows),
        trains with MSE loss and early stopping, then automatically sets
        the anomaly threshold at the 95th percentile of training errors.

        Args:
            X_train: Feature DataFrame containing the columns in
                ``feature_cols``.
            feature_cols: Column names to use as features.

        Returns:
            ``self`` for method chaining.
        """
        set_seed(self._random_state)
        device = get_device()

        self._feature_cols = list(feature_cols)

        # Identify bool columns by dtype
        self._bool_cols = [
            c for c in feature_cols if X_train[c].dtype == bool
        ]

        # Fit scaler and transform features
        self._scaler = fit_feature_scaler(
            X_train[feature_cols], self._bool_cols
        )
        X_scaled = apply_feature_scaler(
            X_train[feature_cols], self._scaler, self._bool_cols
        )

        # --- 90/10 chronological val split (last 10% of rows) ---
        n_total = len(X_scaled)
        n_val = max(1, int(n_total * 0.1))
        n_train = n_total - n_val

        X_train_arr = X_scaled[:n_train]
        X_val_arr = X_scaled[n_train:]

        # Convert to tensors
        X_train_t = torch.tensor(X_train_arr, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val_arr, dtype=torch.float32).to(device)

        # Build model
        self._model = _AutoencoderModule(
            input_dim=self._input_dim,
            bottleneck_dim=self._bottleneck_dim,
        ).to(device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr)
        criterion = nn.MSELoss()
        stopper = EarlyStopping(
            patience=self._patience, min_delta=self._min_delta
        )

        # --- Training loop ---
        n_samples = len(X_train_t)
        for epoch in range(self._max_epochs):
            self._model.train()

            # Shuffle training indices
            perm = torch.randperm(n_samples, device=device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self._batch_size):
                idx = perm[start : start + self._batch_size]
                xb = X_train_t[idx]

                optimizer.zero_grad()
                reconstruction, _bottleneck = self._model(xb)
                loss = criterion(reconstruction, xb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_recon, _ = self._model(X_val_t)
                val_loss = criterion(val_recon, X_val_t).item()

            if stopper.step(val_loss):
                print(
                    f"[Autoencoder] early stopping at epoch {epoch + 1}"
                )
                break

        # Set threshold automatically from training data
        self.set_threshold(X_train)

        return self

    # ------------------------------------------------------------------
    # compute_reconstruction_error
    # ------------------------------------------------------------------

    def compute_reconstruction_error(
        self, X: pd.DataFrame
    ) -> np.ndarray:
        """Compute per-sample reconstruction error (MSE per row).

        Args:
            X: Feature DataFrame (same columns as fit).

        Returns:
            1-D ndarray of per-sample MSE values, shape ``(n_rows,)``.
        """
        device = get_device()
        X_scaled = apply_feature_scaler(
            X[self._feature_cols], self._scaler, self._bool_cols
        )
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        self._model.eval()
        with torch.no_grad():
            reconstruction, _ = self._model(X_tensor)

        # Per-sample MSE: mean across features dimension
        errors = (
            (X_tensor - reconstruction).pow(2).mean(dim=1).cpu().numpy()
        )
        return errors

    # ------------------------------------------------------------------
    # set_threshold
    # ------------------------------------------------------------------

    def set_threshold(
        self, X_train: pd.DataFrame, percentile: float = 95.0
    ) -> None:
        """Set the anomaly threshold from training reconstruction errors.

        Args:
            X_train: Training data (same as passed to ``fit()``).
            percentile: Percentile of training errors to use as threshold
                (default 95.0).
        """
        errors = self.compute_reconstruction_error(X_train)
        self.threshold_ = float(np.percentile(errors, percentile))

    # ------------------------------------------------------------------
    # flag_anomalies
    # ------------------------------------------------------------------

    def flag_anomalies(self, X: pd.DataFrame) -> np.ndarray:
        """Flag anomalous rows based on reconstruction error threshold.

        Args:
            X: Feature DataFrame.

        Returns:
            1-D boolean ndarray of shape ``(n_rows,)`` where ``True``
            indicates an anomalous row (reconstruction error exceeds
            threshold).
        """
        errors = self.compute_reconstruction_error(X)
        return errors > self.threshold_
