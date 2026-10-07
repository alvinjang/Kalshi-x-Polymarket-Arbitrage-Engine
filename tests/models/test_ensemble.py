"""Tests for EnsemblePredictor (weighted-average ensemble with concordance gate).

Covers the 11 behaviors specified in Phase 13 Plan 01 (ENSM-01, ENSM-05):
fit/predict contract, weight normalization, concordance modes, save/load
round-trip, single-member defaults, input validation, name property,
internal fit of members, and group_id smoke test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import BasePredictor
from src.models.ensemble import EnsemblePredictor
from src.models.linear_regression import LinearRegressionPredictor


# ---------------------------------------------------------------------------
# Fixtures local to this module
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_features() -> pd.DataFrame:
    """5-row, 3-numeric-feature DataFrame for fast synthetic tests."""
    return pd.DataFrame(
        {
            "f1": [0.10, -0.20, 0.30, -0.05, 0.15],
            "f2": [0.01, 0.02, -0.01, 0.00, -0.02],
            "f3": [1.00, 2.00, 3.00, 4.00, 5.00],
        }
    )


@pytest.fixture
def tiny_targets(tiny_features) -> np.ndarray:
    """Simple linear target so LR fits well on tiny_features."""
    return (-0.5 * tiny_features["f1"].values
            + 0.1 * tiny_features["f2"].values).astype(float)


class _FixedSignPredictor(BasePredictor):
    """Deterministic stub predictor that returns a constant-valued vector.

    Used to force known-sign predictions for concordance-gate tests.
    """

    def __init__(self, value: float, label: str = "FixedSign") -> None:
        self._value = float(value)
        self._label = label
        self._fitted = False

    @property
    def name(self) -> str:
        return self._label

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> "_FixedSignPredictor":
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._value, dtype=float)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEnsemblePredictor:
    # Test 1: fit/predict contract
    def test_fit_predict_returns_ndarray_no_nan(self, tiny_features, tiny_targets):
        members = [LinearRegressionPredictor(), LinearRegressionPredictor()]
        ens = EnsemblePredictor(members, weights=[0.5, 0.5])
        ens.fit(tiny_features, tiny_targets)
        preds = ens.predict(tiny_features)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(tiny_features),)
        assert np.all(np.isfinite(preds))

    # Test 2: weight normalization (internal, not at call sites)
    def test_weights_are_normalized_internally(self, tiny_features, tiny_targets):
        m1 = [LinearRegressionPredictor(), LinearRegressionPredictor()]
        m2 = [LinearRegressionPredictor(), LinearRegressionPredictor()]
        ens_big = EnsemblePredictor(m1, weights=[2.0, 2.0])
        ens_small = EnsemblePredictor(m2, weights=[1.0, 1.0])
        ens_big.fit(tiny_features, tiny_targets)
        ens_small.fit(tiny_features, tiny_targets)
        preds_big = ens_big.predict(tiny_features)
        preds_small = ens_small.predict(tiny_features)
        np.testing.assert_allclose(preds_big, preds_small)

    # Test 3: concordance strict — disagreement emits 0.0, agreement emits weighted avg
    def test_concordance_strict_gates_disagreement(self, tiny_features, tiny_targets):
        pos = _FixedSignPredictor(0.1, "pos")
        neg = _FixedSignPredictor(-0.1, "neg")
        ens = EnsemblePredictor(
            [pos, neg], weights=[0.5, 0.5], concordance_mode="strict"
        )
        ens.fit(tiny_features, tiny_targets)
        preds = ens.predict(tiny_features)
        # Signs disagree (+0.1 vs -0.1) -> all rows gated to 0.0
        np.testing.assert_array_equal(preds, np.zeros(len(tiny_features)))

        pos2 = _FixedSignPredictor(0.2, "pos2")
        pos3 = _FixedSignPredictor(0.4, "pos3")
        ens2 = EnsemblePredictor(
            [pos2, pos3], weights=[0.5, 0.5], concordance_mode="strict"
        )
        ens2.fit(tiny_features, tiny_targets)
        preds2 = ens2.predict(tiny_features)
        # Both +; weighted avg of 0.2 and 0.4 with equal weights = 0.3
        np.testing.assert_allclose(preds2, np.full(len(tiny_features), 0.3))

    # Test 4: concordance none — weighted avg regardless of sign
    def test_concordance_none_returns_weighted_avg(
        self, tiny_features, tiny_targets
    ):
        pos = _FixedSignPredictor(0.1, "pos")
        neg = _FixedSignPredictor(-0.1, "neg")
        ens = EnsemblePredictor(
            [pos, neg], weights=[0.5, 0.5], concordance_mode="none"
        )
        ens.fit(tiny_features, tiny_targets)
        preds = ens.predict(tiny_features)
        # 0.5 * 0.1 + 0.5 * (-0.1) = 0.0 regardless of disagreement
        np.testing.assert_allclose(preds, np.zeros(len(tiny_features)))

    # Test 5: save/load round-trip produces identical predictions
    def test_save_load_roundtrip_preserves_predictions(
        self, tmp_path, tiny_features, tiny_targets
    ):
        ens = EnsemblePredictor(
            [LinearRegressionPredictor(), LinearRegressionPredictor()],
            weights=[0.5, 0.5],
            concordance_mode="strict",
        )
        ens.fit(tiny_features, tiny_targets)
        preds_before = ens.predict(tiny_features)

        path = tmp_path / "ensemble.pkl"
        ens.save(path)
        loaded = BasePredictor.load(path)

        assert isinstance(loaded, BasePredictor)
        assert isinstance(loaded, EnsemblePredictor)
        preds_after = loaded.predict(tiny_features)
        np.testing.assert_allclose(preds_before, preds_after)

    # Test 6: single member with no weights defaults to [1.0]; matches member
    def test_single_member_matches_member_predictions(
        self, tiny_features, tiny_targets
    ):
        lr = LinearRegressionPredictor()
        ens = EnsemblePredictor([LinearRegressionPredictor()])
        lr.fit(tiny_features, tiny_targets)
        ens.fit(tiny_features, tiny_targets)
        np.testing.assert_allclose(
            ens.predict(tiny_features), lr.predict(tiny_features)
        )

    # Test 7: empty members raises ValueError
    def test_empty_members_raises(self):
        with pytest.raises(ValueError):
            EnsemblePredictor([])

    # Test 8: all-zero weights raises ValueError
    def test_all_zero_weights_raises(self):
        with pytest.raises(ValueError):
            EnsemblePredictor([LinearRegressionPredictor()], weights=[0.0])

    # Test 9: name property is non-empty and contains member names
    def test_name_property_contains_member_names(self):
        ens = EnsemblePredictor(
            [LinearRegressionPredictor(), LinearRegressionPredictor()],
            weights=[0.5, 0.5],
            concordance_mode="strict",
        )
        nm = ens.name
        assert isinstance(nm, str)
        assert len(nm) > 0
        assert "Linear Regression" in nm
        assert "strict" in nm

    # Test 10: members are NOT pre-fitted; ensemble.fit() trains them
    def test_fit_trains_members_internally(self, tiny_features, tiny_targets):
        lr1 = LinearRegressionPredictor()
        lr2 = LinearRegressionPredictor()
        assert lr1._fitted is False
        assert lr2._fitted is False
        ens = EnsemblePredictor([lr1, lr2], weights=[0.5, 0.5])
        ens.fit(tiny_features, tiny_targets)
        assert lr1._fitted is True
        assert lr2._fitted is True

    # Test 11: group_id column does not crash LR-only ensemble
    def test_group_id_column_does_not_crash(self, tiny_features, tiny_targets):
        X_with_gid = tiny_features.copy()
        X_with_gid["group_id"] = [1, 1, 2, 2, 3]
        # Strip group_id before fit, as the runner does; ensemble should
        # handle a DataFrame whose numeric columns include an int group_id
        # without exploding either.
        ens = EnsemblePredictor([LinearRegressionPredictor()])
        ens.fit(X_with_gid, tiny_targets)
        preds = ens.predict(X_with_gid)
        assert preds.shape == (len(X_with_gid),)
        assert np.all(np.isfinite(preds))

    # Extra: predict-before-fit raises
    def test_predict_before_fit_raises(self, tiny_features):
        ens = EnsemblePredictor([LinearRegressionPredictor()])
        with pytest.raises(RuntimeError):
            ens.predict(tiny_features)

    # Extra: inherits BasePredictor (sanity)
    def test_inherits_base_predictor(self):
        assert issubclass(EnsemblePredictor, BasePredictor)
