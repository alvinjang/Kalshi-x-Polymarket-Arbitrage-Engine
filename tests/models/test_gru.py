"""Tests for GRUPredictor (Tier 2 recurrent baseline).

TDD RED phase: these tests define the GRUPredictor contract before any
implementation exists.  All tests import from ``src.models.gru``, which
does not yet exist, so collection should fail with ``ImportError``.

Tests are grouped into four classes:
  - TestGRUPredictor       : interface / inheritance / hyperparameters
  - TestGRUFitPredict      : shape, reproducibility, evaluate integration
  - TestGRUGroupIdContract : group_id guards on fit and predict
  - TestGRUPadding         : warm-up padding logging for short groups
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import BasePredictor
from src.models.gru import GRUPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gru_data() -> tuple[pd.DataFrame, np.ndarray]:
    """40-row DataFrame with group_id + 3 feature columns and targets.

    Two groups of 20 rows each (group_id 0 and 1).
    """
    rng_feat = np.random.default_rng(42)
    rng_tgt = np.random.default_rng(7)

    n = 40
    group_ids = np.array([0] * 20 + [1] * 20)
    feat_a = rng_feat.standard_normal(n)
    feat_b = rng_feat.standard_normal(n)
    feat_c = rng_feat.standard_normal(n)

    X = pd.DataFrame({
        "group_id": group_ids,
        "feat_a": feat_a,
        "feat_b": feat_b,
        "feat_c": feat_c,
    })
    y = rng_tgt.standard_normal(n) * 0.05
    return X, y


# ---------------------------------------------------------------------------
# Interface / inheritance / hyperparameters
# ---------------------------------------------------------------------------

class TestGRUPredictor:
    def test_gru_inherits_base_predictor(self):
        assert issubclass(GRUPredictor, BasePredictor)

    def test_name_property(self):
        predictor = GRUPredictor()
        assert predictor.name == "GRU"

    def test_predict_before_fit_raises(self, gru_data):
        X, _y = gru_data
        predictor = GRUPredictor()
        with pytest.raises(RuntimeError, match="fit"):
            predictor.predict(X)

    def test_fit_returns_self(self, gru_data):
        X, y = gru_data
        predictor = GRUPredictor(max_epochs=2, hidden_size=8, batch_size=16)
        result = predictor.fit(X, y)
        assert result is predictor

    def test_default_hyperparameters_match_research(self):
        predictor = GRUPredictor()
        assert predictor._hidden_size == 64
        assert predictor._num_layers == 1
        assert predictor._dropout == 0.3
        assert predictor._lookback == 6
        assert predictor._learning_rate == 1e-3
        assert predictor._weight_decay == 1e-4
        assert predictor._batch_size == 64
        assert predictor._max_epochs == 100
        assert predictor._patience == 10


# ---------------------------------------------------------------------------
# Fit / predict shape contracts, reproducibility, evaluate integration
# ---------------------------------------------------------------------------

class TestGRUFitPredict:
    def test_predict_returns_1d_ndarray(self, gru_data):
        X, y = gru_data
        predictor = GRUPredictor(max_epochs=2, hidden_size=8, batch_size=16)
        predictor.fit(X, y)
        preds = predictor.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)
        assert preds.dtype == np.float64 or np.issubdtype(preds.dtype, np.floating)

    def test_predict_produces_finite_values(self, gru_data):
        X, y = gru_data
        predictor = GRUPredictor(max_epochs=2, hidden_size=8, batch_size=16)
        predictor.fit(X, y)
        preds = predictor.predict(X)
        assert np.all(np.isfinite(preds))

    def test_seed_reproducibility(self, gru_data):
        X, y = gru_data
        kwargs = dict(max_epochs=3, hidden_size=8, batch_size=16, random_state=42)

        pred_a = GRUPredictor(**kwargs)
        pred_a.fit(X, y)
        preds_a = pred_a.predict(X)

        pred_b = GRUPredictor(**kwargs)
        pred_b.fit(X, y)
        preds_b = pred_b.predict(X)

        np.testing.assert_array_almost_equal(preds_a, preds_b, decimal=4)

    def test_evaluate_returns_tier1_compatible_keys(self, gru_data):
        X, y = gru_data
        predictor = GRUPredictor(max_epochs=2, hidden_size=8, batch_size=16)
        predictor.fit(X, y)
        results = predictor.evaluate(X, y)
        expected_keys = {
            "rmse", "mae", "directional_accuracy",
            "total_pnl", "num_trades", "win_rate",
            "sharpe_ratio", "sharpe_per_trade", "pnl_series",
        }
        assert set(results.keys()) == expected_keys

    def test_predict_respects_group_id(self, gru_data):
        X, y = gru_data
        predictor = GRUPredictor(max_epochs=2, hidden_size=8, batch_size=16)
        predictor.fit(X, y)
        # Input with 2 groups, 10 rows each -- still returns 20 predictions
        X_two = X.head(20).copy()  # first 20 rows = group 0 only (but 20)
        preds = predictor.predict(X_two)
        assert preds.shape == (len(X_two),)

    def test_fit_with_short_group_sequences_does_not_crash(self):
        """A group with fewer rows than lookback is skipped gracefully."""
        rng = np.random.default_rng(99)
        n_short = 3  # less than default lookback=6
        n_long = 20
        group_ids = np.array([0] * n_short + [1] * n_long)
        X = pd.DataFrame({
            "group_id": group_ids,
            "feat_a": rng.standard_normal(n_short + n_long),
            "feat_b": rng.standard_normal(n_short + n_long),
            "feat_c": rng.standard_normal(n_short + n_long),
        })
        y = rng.standard_normal(n_short + n_long) * 0.05
        predictor = GRUPredictor(max_epochs=2, hidden_size=8, batch_size=16)
        # Should not raise -- short group is simply skipped during training
        predictor.fit(X, y)


# ---------------------------------------------------------------------------
# group_id contract
# ---------------------------------------------------------------------------

class TestGRUGroupIdContract:
    def test_fit_raises_without_group_id(self, gru_data):
        X, y = gru_data
        X_no_gid = X.drop(columns=["group_id"])
        predictor = GRUPredictor(max_epochs=2, hidden_size=8, batch_size=16)
        with pytest.raises(ValueError, match="group_id"):
            predictor.fit(X_no_gid, y)

    def test_predict_raises_without_group_id(self, gru_data):
        X, y = gru_data
        predictor = GRUPredictor(max_epochs=2, hidden_size=8, batch_size=16)
        predictor.fit(X, y)
        X_no_gid = X.drop(columns=["group_id"])
        with pytest.raises(ValueError, match="group_id"):
            predictor.predict(X_no_gid)


# ---------------------------------------------------------------------------
# Warm-up padding logging
# ---------------------------------------------------------------------------

class TestGRUPadding:
    def test_padding_logged_for_short_groups(self, gru_data):
        """Predict on data with a novel group_id (no cached train data)
        that has fewer rows than lookback -- predictor must log it."""
        X, y = gru_data
        predictor = GRUPredictor(max_epochs=2, hidden_size=8, batch_size=16)
        predictor.fit(X, y)

        # Build a small DataFrame for a NEW group_id=99 with 3 rows (<6 lookback)
        rng = np.random.default_rng(123)
        n_short = 3
        X_short = pd.DataFrame({
            "group_id": [99] * n_short,
            "feat_a": rng.standard_normal(n_short),
            "feat_b": rng.standard_normal(n_short),
            "feat_c": rng.standard_normal(n_short),
        })
        predictor._padded_pairs = []  # reset tracking
        predictor.predict(X_short)
        assert 99 in predictor._padded_pairs
