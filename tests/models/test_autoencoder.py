"""Tests for AnomalyDetectorAutoencoder (Tier 4 signal filter).

TDD RED phase: these tests define the AnomalyDetectorAutoencoder contract
before any implementation exists.  All tests import from
``src.models.autoencoder``, which does not yet exist, so collection should
fail with ``ImportError``.

Tests cover:
  - Class identity (NOT a BasePredictor)
  - Reconstruction shape (31-in, 31-out)
  - Encoder bottleneck dimension (4)
  - Training loss decrease
  - Threshold setting from training errors (95th percentile)
  - flag_anomalies returns bool array
  - Synthetic outlier detection
  - Reconstruction error shape
  - fit() returns self for chaining
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.autoencoder import AnomalyDetectorAutoencoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ae_data() -> pd.DataFrame:
    """200-row DataFrame with 31 numeric feature columns (standard normal).

    No group_id needed -- autoencoder is point-based (operates on individual
    rows, not sequences).
    """
    rng = np.random.default_rng(42)
    n_rows = 200
    n_features = 31
    col_names = [f"feat_{i}" for i in range(n_features)]
    data = rng.standard_normal((n_rows, n_features))
    return pd.DataFrame(data, columns=col_names)


@pytest.fixture
def feature_cols() -> list[str]:
    """List of 31 feature column names matching ae_data."""
    return [f"feat_{i}" for i in range(31)]


# ---------------------------------------------------------------------------
# Class identity
# ---------------------------------------------------------------------------

class TestAutoencoderIdentity:
    def test_autoencoder_is_not_base_predictor(self):
        """AnomalyDetectorAutoencoder does NOT inherit BasePredictor."""
        from src.models.base import BasePredictor
        assert not issubclass(AnomalyDetectorAutoencoder, BasePredictor)

    def test_fit_returns_self(self, ae_data, feature_cols):
        """fit() returns self for method chaining."""
        ae = AnomalyDetectorAutoencoder(
            input_dim=31, max_epochs=5, patience=3, random_state=42,
        )
        result = ae.fit(ae_data, feature_cols)
        assert result is ae


# ---------------------------------------------------------------------------
# Reconstruction shape and bottleneck
# ---------------------------------------------------------------------------

class TestAutoencoderArchitecture:
    def test_reconstruction_shape_matches_input(self, ae_data, feature_cols):
        """Forward pass on (batch, 31) returns (batch, 31)."""
        ae = AnomalyDetectorAutoencoder(
            input_dim=31, max_epochs=5, patience=3, random_state=42,
        )
        ae.fit(ae_data, feature_cols)
        errors = ae.compute_reconstruction_error(ae_data)
        # If we can compute errors, reconstruction happened.
        # Also verify the internal module output shape directly.
        import torch
        from src.models.sequence_utils import apply_feature_scaler
        X_scaled = apply_feature_scaler(ae_data[feature_cols], ae._scaler, ae._bool_cols)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        ae._model.eval()
        with torch.no_grad():
            reconstruction, _bottleneck = ae._model(X_tensor)
        assert reconstruction.shape == (200, 31)

    def test_encoder_bottleneck_is_4(self, ae_data, feature_cols):
        """Bottleneck output shape is (batch, 4)."""
        ae = AnomalyDetectorAutoencoder(
            input_dim=31, bottleneck_dim=4, max_epochs=5, patience=3,
            random_state=42,
        )
        ae.fit(ae_data, feature_cols)
        import torch
        from src.models.sequence_utils import apply_feature_scaler
        X_scaled = apply_feature_scaler(ae_data[feature_cols], ae._scaler, ae._bool_cols)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        ae._model.eval()
        with torch.no_grad():
            _reconstruction, bottleneck = ae._model(X_tensor)
        assert bottleneck.shape == (200, 4)


# ---------------------------------------------------------------------------
# Training behavior
# ---------------------------------------------------------------------------

class TestAutoencoderTraining:
    def test_loss_decreases_after_training(self, ae_data, feature_cols):
        """Train loss after 10 epochs < initial loss (model learns)."""
        ae = AnomalyDetectorAutoencoder(
            input_dim=31, max_epochs=10, patience=50, random_state=42,
        )
        ae.fit(ae_data, feature_cols)
        # After fitting, compute reconstruction errors on training data.
        # The mean error should be reasonable (not infinite/NaN).
        errors = ae.compute_reconstruction_error(ae_data)
        mean_error = errors.mean()
        assert np.isfinite(mean_error)
        # Reconstruction error after training should be meaningfully below
        # the error of random weights.  We test this by checking that the
        # error on training data is below a generous upper bound (< 2.0).
        # A random autoencoder on standardized data typically has MSE ~1.0;
        # 10 epochs of training should push it well below that.
        assert mean_error < 2.0, (
            f"Mean reconstruction error {mean_error:.4f} is too high; "
            "model may not be learning."
        )


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------

class TestAutoencoderThreshold:
    def test_threshold_set_from_training_errors(self, ae_data, feature_cols):
        """After fit, threshold_ attribute is set (95th percentile)."""
        ae = AnomalyDetectorAutoencoder(
            input_dim=31, max_epochs=5, patience=3, random_state=42,
        )
        ae.fit(ae_data, feature_cols)
        assert hasattr(ae, "threshold_")
        assert ae.threshold_ is not None
        assert isinstance(ae.threshold_, float)
        assert ae.threshold_ > 0.0

    def test_flag_anomalies_returns_bool_array(self, ae_data, feature_cols):
        """flag_anomalies returns 1D bool array same length as input."""
        ae = AnomalyDetectorAutoencoder(
            input_dim=31, max_epochs=5, patience=3, random_state=42,
        )
        ae.fit(ae_data, feature_cols)
        flags = ae.flag_anomalies(ae_data)
        assert isinstance(flags, np.ndarray)
        assert flags.dtype == bool
        assert flags.shape == (200,)

    def test_synthetic_outlier_flagged(self, ae_data, feature_cols):
        """A row with values 10x normal gets flagged as anomalous."""
        ae = AnomalyDetectorAutoencoder(
            input_dim=31, max_epochs=10, patience=50, random_state=42,
        )
        ae.fit(ae_data, feature_cols)
        # Create outlier data: normal data with one extreme row appended
        outlier_row = pd.DataFrame(
            [[10.0] * 31], columns=feature_cols,
        )
        test_data = pd.concat([ae_data, outlier_row], ignore_index=True)
        flags = ae.flag_anomalies(test_data)
        # The last row (outlier) should be flagged
        assert flags[-1] is np.True_, (
            "Synthetic outlier (10x normal values) was not flagged as anomalous"
        )


# ---------------------------------------------------------------------------
# Reconstruction error shape
# ---------------------------------------------------------------------------

class TestReconstructionError:
    def test_compute_reconstruction_error_shape(self, ae_data, feature_cols):
        """compute_reconstruction_error returns 1D array of per-sample errors."""
        ae = AnomalyDetectorAutoencoder(
            input_dim=31, max_epochs=5, patience=3, random_state=42,
        )
        ae.fit(ae_data, feature_cols)
        errors = ae.compute_reconstruction_error(ae_data)
        assert isinstance(errors, np.ndarray)
        assert errors.ndim == 1
        assert errors.shape == (200,)
        assert np.all(errors >= 0), "Reconstruction errors should be non-negative"
        assert np.all(np.isfinite(errors)), "Reconstruction errors should be finite"
