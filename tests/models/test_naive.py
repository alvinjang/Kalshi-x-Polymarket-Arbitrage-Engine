"""Tests for NaivePredictor (spread always closes fully to zero)."""
import numpy as np
import pandas as pd
import pytest

from src.models.base import BasePredictor
from src.models.naive import NaivePredictor


class TestNaivePredictor:
    def test_inherits_base_predictor(self):
        assert issubclass(NaivePredictor, BasePredictor)

    def test_name_property(self):
        predictor = NaivePredictor()
        assert predictor.name == "Naive (Spread Closes)"

    def test_fit_is_noop_returns_self(self):
        predictor = NaivePredictor()
        X = pd.DataFrame({"spread": [0.1, 0.2]})
        y = np.array([0.0, 0.1])
        result = predictor.fit(X, y)
        assert result is predictor

    def test_predict_returns_negative_spread(self):
        predictor = NaivePredictor()
        X = pd.DataFrame({"spread": [0.05, -0.03, 0.10]})
        predictions = predictor.predict(X)
        np.testing.assert_array_almost_equal(
            predictions, np.array([-0.05, 0.03, -0.10])
        )

    def test_predict_returns_ndarray(self):
        predictor = NaivePredictor()
        X = pd.DataFrame({"spread": [0.01, 0.02, 0.03]})
        predictions = predictor.predict(X)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (3,)

    def test_predict_length_matches_input(self, sample_features):
        predictor = NaivePredictor()
        predictions = predictor.predict(sample_features)
        assert len(predictions) == len(sample_features)

    def test_predict_missing_spread_column_raises(self):
        predictor = NaivePredictor()
        X = pd.DataFrame({"kalshi_vwap": [0.5, 0.6]})
        with pytest.raises(ValueError, match="spread"):
            predictor.predict(X)

    def test_evaluate_produces_full_metrics_dict(
        self, sample_features, sample_targets
    ):
        predictor = NaivePredictor()
        results = predictor.evaluate(sample_features, sample_targets)
        expected = {
            "rmse", "mae", "directional_accuracy",
            "total_pnl", "num_trades", "win_rate",
            "sharpe_ratio", "pnl_series",
        }
        assert set(results.keys()) == expected

    def test_fit_then_predict_is_stateless(self, sample_features, sample_targets):
        predictor = NaivePredictor()
        predictor.fit(sample_features, sample_targets)
        predictions_after_fit = predictor.predict(sample_features)
        # predictions should only depend on spread column, not fit
        expected = -sample_features["spread"].values
        np.testing.assert_array_almost_equal(predictions_after_fit, expected)
