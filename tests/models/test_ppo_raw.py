"""Tests for PPORawPredictor (Tier 3 RL baseline).

TDD RED phase: these tests define the PPORawPredictor contract before any
implementation exists.  All tests import from ``src.models.ppo_raw``, which
does not yet exist, so collection should fail with ``ImportError``.

Tests cover:
  - BasePredictor inheritance and name property
  - fit/predict contract (returns self, 1-D ndarray, action-mapped values)
  - Action-to-prediction mapping: hold(0)->0.0, long(1)->+0.03, short(2)->-0.03
  - evaluate() returns Tier 1/2 compatible metric keys
  - group_id guard on fit()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import BasePredictor
from src.models.ppo_raw import PPORawPredictor


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
def fitted_ppo(ppo_data) -> tuple[PPORawPredictor, pd.DataFrame, np.ndarray]:
    """Return a fitted PPORawPredictor with small timesteps for speed."""
    X, y = ppo_data
    predictor = PPORawPredictor(
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

class TestPPORawPredictor:
    def test_ppo_raw_inherits_base_predictor(self):
        assert issubclass(PPORawPredictor, BasePredictor)

    def test_name_property(self):
        predictor = PPORawPredictor()
        assert predictor.name == "PPO-Raw"

    def test_predict_before_fit_raises(self, ppo_data):
        X, _y = ppo_data
        predictor = PPORawPredictor()
        with pytest.raises(RuntimeError, match="fit"):
            predictor.predict(X)

    def test_fit_returns_self(self, ppo_data):
        X, y = ppo_data
        predictor = PPORawPredictor(
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

class TestPPORawFitPredict:
    def test_predict_returns_1d_ndarray(self, fitted_ppo):
        predictor, X, _y = fitted_ppo
        preds = predictor.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)

    def test_predictions_are_from_action_mapping(self, fitted_ppo):
        """All prediction values must be from the set {-0.03, 0.0, +0.03}."""
        predictor, X, _y = fitted_ppo
        preds = predictor.predict(X)
        valid_values = {-0.03, 0.0, 0.03}
        for val in preds:
            assert float(val) in valid_values, (
                f"Prediction {val} not in valid action-mapped set {valid_values}"
            )

    def test_action_to_prediction_mapping(self):
        """Verify hold(0)->0.0, long(1)->+0.03, short(2)->-0.03."""
        # This tests the class-level mapping directly
        predictor = PPORawPredictor()
        mapping = predictor._action_to_prediction
        assert mapping[0] == 0.0, "hold should map to 0.0"
        assert mapping[1] == pytest.approx(0.03), "long should map to +0.03"
        assert mapping[2] == pytest.approx(-0.03), "short should map to -0.03"


# ---------------------------------------------------------------------------
# Evaluation integration
# ---------------------------------------------------------------------------

class TestPPORawEvaluate:
    def test_evaluate_returns_tier1_compatible_keys(self, fitted_ppo):
        predictor, X, y = fitted_ppo
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

class TestPPORawGroupIdContract:
    def test_fit_requires_group_id(self, ppo_data):
        X, y = ppo_data
        X_no_gid = X.drop(columns=["group_id"])
        predictor = PPORawPredictor(
            total_timesteps=512,
            n_steps=64,
            batch_size=32,
            lookback=6,
        )
        with pytest.raises(ValueError, match="group_id"):
            predictor.fit(X_no_gid, y)
