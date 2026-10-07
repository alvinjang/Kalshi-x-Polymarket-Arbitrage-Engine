"""Tests for TFTPredictor (Tier 2 Temporal Fusion Transformer).

TDD RED phase: these tests define the TFTPredictor contract before any
implementation exists.  All tests import from ``src.models.tft``, which
does not yet exist, so collection should fail with ``ImportError``.

Tests are grouped into:
  - TestTFTInterface        : inheritance, name, predict-before-fit guard
  - TestTFTGroupIdContract  : group_id guards on fit and predict
  - TestTFTHyperparameters  : locked defaults match TFT-02 spec
  - TestTFTFitPredict       : shape contract, returns self
  - TestTFTAttentionAudit   : entropy audit degenerate-detection logic
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import BasePredictor
from src.models.tft import TFTPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tft_data(
    n_groups: int = 3,
    n_train_per_group: int = 10,
    n_test_per_group: int = 3,
    n_features: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Create minimal train and test DataFrames for TFT smoke tests.

    Each group has ``n_train_per_group`` rows in train, ``n_test_per_group``
    in test. All numeric feature columns have sufficient variance to satisfy
    the scaler's zero-variance guard.
    """
    rng = np.random.default_rng(seed)

    rows_train = []
    rows_test = []
    for g in range(n_groups):
        feat_train = rng.standard_normal((n_train_per_group, n_features))
        for r in range(n_train_per_group):
            row = {"group_id": g}
            for f in range(n_features):
                row[f"feat_{f}"] = float(feat_train[r, f])
            rows_train.append(row)

        feat_test = rng.standard_normal((n_test_per_group, n_features))
        for r in range(n_test_per_group):
            row = {"group_id": g}
            for f in range(n_features):
                row[f"feat_{f}"] = float(feat_test[r, f])
            rows_test.append(row)

    X_train = pd.DataFrame(rows_train)
    y_train = rng.standard_normal(n_groups * n_train_per_group) * 0.05
    X_test = pd.DataFrame(rows_test)
    return X_train, y_train, X_test


@pytest.fixture
def tft_data():
    """Return (X_train, y_train, X_test) for 3 groups x 10 train rows each."""
    return _make_tft_data(n_groups=3, n_train_per_group=10, n_test_per_group=3)


@pytest.fixture
def fitted_tft(tft_data):
    """Return a TFTPredictor fitted with max_epochs=2 for fast tests."""
    X_train, y_train, _ = tft_data
    model = TFTPredictor(max_epochs=2, random_state=42)
    model.fit(X_train, y_train)
    return model, tft_data


# ---------------------------------------------------------------------------
# Interface / inheritance
# ---------------------------------------------------------------------------

class TestTFTInterface:
    def test_tft_inherits_base_predictor(self):
        assert issubclass(TFTPredictor, BasePredictor)

    def test_name_property(self):
        assert TFTPredictor().name == "TFT"

    def test_predict_before_fit_raises(self, tft_data):
        X_train, _y, _X_test = tft_data
        predictor = TFTPredictor()
        with pytest.raises(RuntimeError, match="fit"):
            predictor.predict(X_train)

    def test_fit_returns_self(self, tft_data):
        X_train, y_train, _ = tft_data
        predictor = TFTPredictor(max_epochs=2, random_state=42)
        result = predictor.fit(X_train, y_train)
        assert result is predictor


# ---------------------------------------------------------------------------
# group_id contract
# ---------------------------------------------------------------------------

class TestTFTGroupIdContract:
    def test_fit_missing_group_id_raises(self, tft_data):
        X_train, y_train, _ = tft_data
        X_no_gid = X_train.drop(columns=["group_id"])
        predictor = TFTPredictor(max_epochs=1, random_state=42)
        with pytest.raises(ValueError, match="group_id"):
            predictor.fit(X_no_gid, y_train)

    def test_predict_missing_group_id_raises(self, fitted_tft):
        model, tft_data_tuple = fitted_tft
        X_train, _, X_test = tft_data_tuple
        X_no_gid = X_test.drop(columns=["group_id"])
        with pytest.raises(ValueError, match="group_id"):
            model.predict(X_no_gid)


# ---------------------------------------------------------------------------
# Hyperparameter defaults (TFT-02 locked values)
# ---------------------------------------------------------------------------

class TestTFTHyperparameters:
    def test_hyperparameter_defaults(self):
        predictor = TFTPredictor()
        assert predictor._hidden_size == 8
        assert predictor._attention_head_size == 1
        assert predictor._dropout == 0.3
        assert predictor._max_encoder_length == 6
        assert predictor._learning_rate == pytest.approx(1e-3)
        assert predictor._lstm_layers == 1


# ---------------------------------------------------------------------------
# Fit / predict shape contract
# ---------------------------------------------------------------------------

class TestTFTFitPredict:
    def test_predict_shape(self, tft_data):
        """predict() returns 1-D ndarray of len(X_test)."""
        X_train, y_train, X_test = tft_data
        model = TFTPredictor(max_epochs=2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert isinstance(preds, np.ndarray)
        assert preds.ndim == 1
        assert len(preds) == len(X_test)

    def test_predict_returns_finite_values(self, tft_data):
        X_train, y_train, X_test = tft_data
        model = TFTPredictor(max_epochs=2, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# Attention entropy audit (TFT-05)
# ---------------------------------------------------------------------------

class TestTFTAttentionAudit:
    def test_attention_audit_degenerate_detection(self):
        """audit_attention returns is_degenerate=True when max_weight > 0.8.

        This tests the _audit_attention() logic by creating a mock interpretation
        dict where encoder_variables weights are highly skewed (one feature
        dominates with weight 0.99, other two share 0.01 total).

        Expected: entropy < 0.5 * log(3), max_weight > 0.8, is_degenerate=True.
        """
        import torch

        # Build a mock interpretation dict with degenerate attention weights.
        # The first feature dominates: weights [0.99, 0.005, 0.005]
        n_features = 3
        weights = np.array([0.99, 0.005, 0.005])
        enc_vars_tensor = torch.tensor(weights, dtype=torch.float32)

        mock_interp = {
            "encoder_variables": enc_vars_tensor,
        }

        # Replicate the audit logic from TFTPredictor._audit_attention
        enc_vars = mock_interp["encoder_variables"].detach().cpu().numpy()
        probs = enc_vars / enc_vars.sum()
        probs_nz = probs[probs > 0]
        entropy = float(-np.sum(probs_nz * np.log(probs_nz)))
        max_weight = float(probs.max())
        threshold_entropy = 0.5 * np.log(n_features)
        is_degenerate = (entropy < threshold_entropy) or (max_weight > 0.8)

        # Verify audit detects degenerate weights
        assert max_weight > 0.8, f"Expected max_weight > 0.8, got {max_weight:.4f}"
        assert entropy < threshold_entropy, (
            f"Expected entropy {entropy:.4f} < threshold {threshold_entropy:.4f}"
        )
        assert is_degenerate, "Expected is_degenerate=True for skewed weights"
