"""Tests for XGBoostPredictor (xgboost.XGBRegressor wrapper)."""
import numpy as np
import pandas as pd
import pytest

from src.models.base import BasePredictor
from src.models.xgboost_model import XGBoostPredictor


class TestXGBoostPredictor:
    def test_inherits_base_predictor(self):
        assert issubclass(XGBoostPredictor, BasePredictor)

    def test_name_property(self):
        predictor = XGBoostPredictor()
        assert predictor.name == "XGBoost"

    def test_predict_before_fit_raises(self, sample_features):
        predictor = XGBoostPredictor()
        with pytest.raises(RuntimeError, match="fit"):
            predictor.predict(sample_features)

    def test_fit_returns_self(self, sample_features, sample_targets):
        predictor = XGBoostPredictor(n_estimators=10, max_depth=3)
        result = predictor.fit(sample_features, sample_targets)
        assert result is predictor

    def test_predict_returns_ndarray_of_correct_length(
        self, sample_features, sample_targets
    ):
        predictor = XGBoostPredictor(n_estimators=10, max_depth=3)
        predictor.fit(sample_features, sample_targets)
        predictions = predictor.predict(sample_features)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(sample_features),)

    def test_predict_produces_finite_values(
        self, sample_features, sample_targets
    ):
        predictor = XGBoostPredictor(n_estimators=10, max_depth=3)
        predictor.fit(sample_features, sample_targets)
        predictions = predictor.predict(sample_features)
        assert np.all(np.isfinite(predictions))

    def test_evaluate_produces_full_metrics_dict(
        self, sample_features, sample_targets
    ):
        predictor = XGBoostPredictor(n_estimators=10, max_depth=3)
        predictor.fit(sample_features, sample_targets)
        results = predictor.evaluate(sample_features, sample_targets)
        expected = {
            "rmse", "mae", "directional_accuracy",
            "total_pnl", "num_trades", "win_rate",
            "sharpe_ratio", "pnl_series",
        }
        assert set(results.keys()) == expected

    def test_accepts_hyperparameters_via_constructor(self):
        predictor = XGBoostPredictor(
            n_estimators=50, max_depth=4, learning_rate=0.05
        )
        # Verify hyperparameters stored on the underlying estimator
        assert predictor._model.n_estimators == 50
        assert predictor._model.max_depth == 4
        assert predictor._model.learning_rate == 0.05

    def test_default_hyperparameters(self):
        predictor = XGBoostPredictor()
        assert predictor._model.n_estimators == 100
        assert predictor._model.max_depth == 6
        assert predictor._model.learning_rate == 0.1

    def test_random_state_is_reproducible(
        self, sample_features, sample_targets
    ):
        predictor_a = XGBoostPredictor(n_estimators=10, random_state=42)
        predictor_b = XGBoostPredictor(n_estimators=10, random_state=42)
        predictor_a.fit(sample_features, sample_targets)
        predictor_b.fit(sample_features, sample_targets)
        preds_a = predictor_a.predict(sample_features)
        preds_b = predictor_b.predict(sample_features)
        np.testing.assert_array_almost_equal(preds_a, preds_b)
