"""Tests for regression metrics used across all model tiers."""
import numpy as np
import pytest

from src.evaluation.metrics import compute_regression_metrics


class TestComputeRegressionMetrics:
    def test_returns_dict_with_expected_keys(self, sample_y_true, sample_y_pred):
        result = compute_regression_metrics(sample_y_true, sample_y_pred)
        assert set(result.keys()) == {"rmse", "mae", "directional_accuracy"}

    def test_rmse_is_zero_for_identical_arrays(self):
        y = np.array([1.0, 2.0, 3.0])
        result = compute_regression_metrics(y, y)
        assert result["rmse"] == 0.0

    def test_rmse_matches_manual_calculation(self):
        # [0,0,0] vs [1,1,1] -> RMSE = sqrt(mean(1,1,1)) = 1.0
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        result = compute_regression_metrics(y_true, y_pred)
        assert result["rmse"] == pytest.approx(1.0)

    def test_mae_equals_one_for_constant_offset(self):
        # [1,2,3] vs [2,3,4] -> MAE = mean(|-1|, |-1|, |-1|) = 1.0
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        result = compute_regression_metrics(y_true, y_pred)
        assert result["mae"] == pytest.approx(1.0)

    def test_mae_is_zero_for_identical_arrays(self):
        y = np.array([0.5, 0.3, -0.2])
        result = compute_regression_metrics(y, y)
        assert result["mae"] == 0.0

    def test_directional_accuracy_all_correct(self):
        # all predictions have same sign as truths
        y_true = np.array([0.1, -0.1, 0.2])
        y_pred = np.array([0.05, -0.2, 0.3])
        result = compute_regression_metrics(y_true, y_pred)
        assert result["directional_accuracy"] == pytest.approx(1.0)

    def test_directional_accuracy_all_wrong(self):
        # opposite signs
        y_true = np.array([0.1, -0.1])
        y_pred = np.array([-0.1, 0.1])
        result = compute_regression_metrics(y_true, y_pred)
        assert result["directional_accuracy"] == pytest.approx(0.0)

    def test_directional_accuracy_excludes_zero_truths(self):
        # y_true == 0 should be excluded (no direction to predict)
        # only 2 of 3 are directional; both are correct -> 1.0
        y_true = np.array([0.0, 0.1, -0.1])
        y_pred = np.array([0.5, 0.2, -0.3])
        result = compute_regression_metrics(y_true, y_pred)
        assert result["directional_accuracy"] == pytest.approx(1.0)

    def test_directional_accuracy_mixed(self):
        # 2 correct, 2 wrong -> 0.5
        y_true = np.array([0.1, -0.1, 0.2, -0.2])
        y_pred = np.array([0.3, 0.1, 0.1, 0.1])
        result = compute_regression_metrics(y_true, y_pred)
        assert result["directional_accuracy"] == pytest.approx(0.5)

    def test_works_on_fixture_arrays(self, sample_y_true, sample_y_pred):
        result = compute_regression_metrics(sample_y_true, sample_y_pred)
        assert isinstance(result["rmse"], float)
        assert isinstance(result["mae"], float)
        assert isinstance(result["directional_accuracy"], float)
        assert result["rmse"] >= 0.0
        assert result["mae"] >= 0.0
        assert 0.0 <= result["directional_accuracy"] <= 1.0

    def test_directional_accuracy_all_truths_zero_returns_zero(self):
        # edge case: no truths with direction -> 0.0 (no signal to compare)
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([0.1, -0.1, 0.0])
        result = compute_regression_metrics(y_true, y_pred)
        assert result["directional_accuracy"] == 0.0
