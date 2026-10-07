"""Tests for PPOFilteredPredictor (Tier 4 RL + anomaly filter).

TDD RED phase: these tests define the PPOFilteredPredictor contract before
any implementation exists.  All tests import from ``src.models.ppo_filtered``,
which does not yet exist, so collection should fail with ``ImportError``.

Tests cover:
  - BasePredictor inheritance and name property
  - fit/predict contract (returns self, 1-D ndarray, action-mapped values)
  - Action-to-prediction mapping: hold(0)->0.0, long(1)->+0.03, short(2)->-0.03
  - evaluate() returns Tier 1/2 compatible metric keys
  - group_id guard on fit()
  - Constructor accepts pre-trained AnomalyDetectorAutoencoder
  - fit() trains an autoencoder internally when none provided
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import BasePredictor
from src.models.autoencoder import AnomalyDetectorAutoencoder
from src.models.ppo_filtered import PPOFilteredPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ppo_data() -> tuple[pd.DataFrame, np.ndarray]:
    """40-row DataFrame with group_id + 31 feature columns and targets.

    Two groups of 20 rows each (group_id 0 and 1).  31 numeric features
    to match the real dataset dimensionality.
    """
    rng_feat = np.random.default_rng(42)
    rng_tgt = np.random.default_rng(7)

    n = 40
    group_ids = np.array([0] * 20 + [1] * 20)

    # Build 31 feature columns
    feature_data = {"group_id": group_ids}
    for i in range(31):
        feature_data[f"feat_{i:02d}"] = rng_feat.standard_normal(n)

    X = pd.DataFrame(feature_data)
    y = rng_tgt.standard_normal(n) * 0.05
    return X, y


@pytest.fixture
def trained_autoencoder(ppo_data) -> AnomalyDetectorAutoencoder:
    """Pre-trained AnomalyDetectorAutoencoder on same data, fast (5 epochs)."""
    X, _y = ppo_data
    feature_cols = [c for c in X.columns if c != "group_id"]
    ae = AnomalyDetectorAutoencoder(
        input_dim=len(feature_cols),
        max_epochs=5,
        patience=3,
        random_state=42,
    )
    ae.fit(X[feature_cols], feature_cols)
    return ae


@pytest.fixture
def fitted_ppo_filtered(
    ppo_data, trained_autoencoder
) -> tuple[PPOFilteredPredictor, pd.DataFrame, np.ndarray]:
    """Return a fitted PPOFilteredPredictor with small timesteps for speed."""
    X, y = ppo_data
    predictor = PPOFilteredPredictor(
        anomaly_detector=trained_autoencoder,
        total_timesteps=512,
        n_steps=64,
        batch_size=32,
        lookback=6,
        random_state=42,
    )
    predictor.fit(X, y)
    return predictor, X, y


# ---------------------------------------------------------------------------
# Interface / inheritance
# ---------------------------------------------------------------------------

class TestPPOFilteredPredictor:
    def test_ppo_filtered_inherits_base_predictor(self):
        assert issubclass(PPOFilteredPredictor, BasePredictor)

    def test_name_property(self):
        predictor = PPOFilteredPredictor()
        assert predictor.name == "PPO-Filtered"

    def test_predict_before_fit_raises(self, ppo_data):
        X, _y = ppo_data
        predictor = PPOFilteredPredictor()
        with pytest.raises(RuntimeError, match="fit"):
            predictor.predict(X)

    def test_fit_returns_self(self, ppo_data, trained_autoencoder):
        X, y = ppo_data
        predictor = PPOFilteredPredictor(
            anomaly_detector=trained_autoencoder,
            total_timesteps=512,
            n_steps=64,
            batch_size=32,
            lookback=6,
        )
        result = predictor.fit(X, y)
        assert result is predictor


# ---------------------------------------------------------------------------
# Fit / predict shape and action mapping
# ---------------------------------------------------------------------------

class TestPPOFilteredFitPredict:
    def test_predict_returns_1d_ndarray(self, fitted_ppo_filtered):
        predictor, X, _y = fitted_ppo_filtered
        preds = predictor.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)

    def test_predictions_are_from_action_mapping(self, fitted_ppo_filtered):
        """All prediction values must be from the set {-0.03, 0.0, +0.03}."""
        predictor, X, _y = fitted_ppo_filtered
        preds = predictor.predict(X)
        valid_values = {-0.03, 0.0, 0.03}
        for val in preds:
            assert float(val) in valid_values, (
                f"Prediction {val} not in valid action-mapped set {valid_values}"
            )


# ---------------------------------------------------------------------------
# Evaluation integration
# ---------------------------------------------------------------------------

class TestPPOFilteredEvaluate:
    def test_evaluate_returns_tier1_compatible_keys(self, fitted_ppo_filtered):
        predictor, X, y = fitted_ppo_filtered
        results = predictor.evaluate(X, y)
        expected_keys = {
            "rmse", "mae", "directional_accuracy",
            "total_pnl", "num_trades", "win_rate",
            "sharpe_ratio", "sharpe_per_trade", "pnl_series",
        }
        assert set(results.keys()) == expected_keys


# ---------------------------------------------------------------------------
# group_id contract
# ---------------------------------------------------------------------------

class TestPPOFilteredGroupIdContract:
    def test_fit_requires_group_id(self, ppo_data, trained_autoencoder):
        X, y = ppo_data
        X_no_gid = X.drop(columns=["group_id"])
        predictor = PPOFilteredPredictor(
            anomaly_detector=trained_autoencoder,
            total_timesteps=512,
            n_steps=64,
            batch_size=32,
            lookback=6,
        )
        with pytest.raises(ValueError, match="group_id"):
            predictor.fit(X_no_gid, y)


# ---------------------------------------------------------------------------
# Autoencoder integration
# ---------------------------------------------------------------------------

class TestPPOFilteredAutoencoderIntegration:
    def test_constructor_accepts_autoencoder(self, trained_autoencoder):
        """PPOFilteredPredictor(anomaly_detector=trained_ae) does not raise."""
        predictor = PPOFilteredPredictor(
            anomaly_detector=trained_autoencoder,
        )
        assert predictor._anomaly_detector is trained_autoencoder

    def test_fit_trains_autoencoder_if_not_provided(self, ppo_data):
        """If anomaly_detector=None, fit trains one internally."""
        X, y = ppo_data
        predictor = PPOFilteredPredictor(
            anomaly_detector=None,
            total_timesteps=512,
            n_steps=64,
            batch_size=32,
            lookback=6,
            random_state=42,
        )
        predictor.fit(X, y)
        # After fit, an internal autoencoder should exist
        assert predictor._anomaly_detector is not None
        assert isinstance(predictor._anomaly_detector, AnomalyDetectorAutoencoder)
        assert predictor._anomaly_detector.threshold_ is not None
