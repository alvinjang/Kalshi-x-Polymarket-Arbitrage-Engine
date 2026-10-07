"""Tests for LinearRegressionPredictor (sklearn LinearRegression wrapper)."""
import numpy as np
import pandas as pd
import pytest

from src.models.base import BasePredictor
from src.models.linear_regression import LinearRegressionPredictor


class TestLinearRegressionPredictor:
    def test_inherits_base_predictor(self):
        assert issubclass(LinearRegressionPredictor, BasePredictor)

    def test_name_property(self):
        predictor = LinearRegressionPredictor()
        assert predictor.name == "Linear Regression"

    def test_predict_before_fit_raises(self, sample_features):
        predictor = LinearRegressionPredictor()
        with pytest.raises(RuntimeError, match="fit"):
            predictor.predict(sample_features)

    def test_fit_returns_self(self, sample_features, sample_targets):
        predictor = LinearRegressionPredictor()
        result = predictor.fit(sample_features, sample_targets)
        assert result is predictor

    def test_predict_returns_ndarray_of_correct_length(
        self, sample_features, sample_targets
    ):
        predictor = LinearRegressionPredictor()
        predictor.fit(sample_features, sample_targets)
        predictions = predictor.predict(sample_features)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(sample_features),)

    def test_predict_produces_finite_values(
        self, sample_features, sample_targets
    ):
        predictor = LinearRegressionPredictor()
        predictor.fit(sample_features, sample_targets)
        predictions = predictor.predict(sample_features)
        assert np.all(np.isfinite(predictions))

    def test_evaluate_produces_full_metrics_dict(
        self, sample_features, sample_targets
    ):
        predictor = LinearRegressionPredictor()
        predictor.fit(sample_features, sample_targets)
        results = predictor.evaluate(sample_features, sample_targets)
        expected = {
            "rmse", "mae", "directional_accuracy",
            "total_pnl", "num_trades", "win_rate",
            "sharpe_ratio", "pnl_series",
        }
        assert set(results.keys()) == expected

    def test_fit_learns_from_data(self, sample_features, sample_targets):
        # After fitting, predictions should be closer to targets than zeros
        predictor = LinearRegressionPredictor()
        predictor.fit(sample_features, sample_targets)
        predictions = predictor.predict(sample_features)
        naive_rmse = float(np.sqrt(np.mean(sample_targets ** 2)))
        model_rmse = float(
            np.sqrt(np.mean((sample_targets - predictions) ** 2))
        )
        # Model should fit training data at least as well as predicting zeros
        assert model_rmse <= naive_rmse
