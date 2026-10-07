"""Tests for VolumePredictor (higher-volume platform is assumed correct)."""
import numpy as np
import pandas as pd
import pytest

from src.models.base import BasePredictor
from src.models.volume import VolumePredictor


class TestVolumePredictor:
    def test_inherits_base_predictor(self):
        assert issubclass(VolumePredictor, BasePredictor)

    def test_name_property(self):
        predictor = VolumePredictor()
        assert predictor.name == "Volume (Higher Volume Correct)"

    def test_fit_is_noop_returns_self(self):
        predictor = VolumePredictor()
        X = pd.DataFrame(
            {"spread": [0.1], "kalshi_volume": [1.0], "polymarket_volume": [1.0]}
        )
        y = np.array([0.0])
        result = predictor.fit(X, y)
        assert result is predictor

    def test_predict_returns_ndarray(self, sample_features):
        predictor = VolumePredictor()
        predictions = predictor.predict(sample_features)
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (len(sample_features),)

    def test_equal_volumes_half_reversion(self):
        # when volumes equal, volume_ratio = 0.5, prediction = -spread * 0.5
        predictor = VolumePredictor()
        X = pd.DataFrame(
            {
                "spread": [0.10, -0.20],
                "kalshi_volume": [100.0, 100.0],
                "polymarket_volume": [100.0, 100.0],
            }
        )
        predictions = predictor.predict(X)
        np.testing.assert_array_almost_equal(predictions, [-0.05, 0.10])

    def test_polymarket_dominates_predicts_near_full_reversion(self):
        # polymarket_volume >> kalshi_volume -> volume_ratio -> 1
        # prediction -> -spread
        predictor = VolumePredictor()
        X = pd.DataFrame(
            {
                "spread": [0.05],
                "kalshi_volume": [0.0],
                "polymarket_volume": [200.0],
            }
        )
        predictions = predictor.predict(X)
        # volume_ratio = 200 / 200 = 1.0 -> prediction = -0.05 * 1.0 = -0.05
        np.testing.assert_array_almost_equal(predictions, [-0.05])

    def test_kalshi_dominates_predicts_near_full_reversion(self):
        # kalshi_volume >> polymarket_volume -> volume_ratio -> 1
        predictor = VolumePredictor()
        X = pd.DataFrame(
            {
                "spread": [0.05],
                "kalshi_volume": [300.0],
                "polymarket_volume": [50.0],
            }
        )
        predictions = predictor.predict(X)
        # volume_ratio = 300 / 350 = 6/7 -> prediction = -0.05 * 6/7
        expected = -0.05 * (300.0 / 350.0)
        np.testing.assert_array_almost_equal(predictions, [expected])

    def test_zero_total_volume_predicts_zero(self):
        # both volumes zero -> no signal, predict 0 (avoid div by zero)
        predictor = VolumePredictor()
        X = pd.DataFrame(
            {
                "spread": [0.05],
                "kalshi_volume": [0.0],
                "polymarket_volume": [0.0],
            }
        )
        predictions = predictor.predict(X)
        np.testing.assert_array_almost_equal(predictions, [0.0])

    def test_predict_missing_spread_column_raises(self):
        predictor = VolumePredictor()
        X = pd.DataFrame(
            {"kalshi_volume": [100.0], "polymarket_volume": [100.0]}
        )
        with pytest.raises(ValueError, match="spread"):
            predictor.predict(X)

    def test_predict_missing_volume_columns_raises(self):
        predictor = VolumePredictor()
        X = pd.DataFrame({"spread": [0.05]})
        with pytest.raises(ValueError, match="volume"):
            predictor.predict(X)

    def test_evaluate_produces_full_metrics_dict(
        self, sample_features, sample_targets
    ):
        predictor = VolumePredictor()
        results = predictor.evaluate(sample_features, sample_targets)
        expected = {
            "rmse", "mae", "directional_accuracy",
            "total_pnl", "num_trades", "win_rate",
            "sharpe_ratio", "pnl_series",
        }
        assert set(results.keys()) == expected
