"""Tests for LSTMPredictor (Tier 2 recurrent alternative).

TDD RED phase: these tests are written BEFORE the implementation.
Mirrors test_gru.py structure, adapted for LSTM with hidden_size=32.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.base import BasePredictor
from src.models.lstm import LSTMPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lstm_data():
    """40-row DataFrame with group_id + 3 feature columns.

    2 groups x 20 rows each.  Columns: group_id, feat_a, feat_b, feat_c.
    """
    rng = np.random.default_rng(42)
    n = 40
    group_ids = np.array([0] * 20 + [1] * 20)
    X = pd.DataFrame(
        {
            "group_id": group_ids,
            "feat_a": rng.standard_normal(n),
            "feat_b": rng.standard_normal(n),
            "feat_c": rng.standard_normal(n),
        }
    )
    y = np.random.default_rng(7).standard_normal(n) * 0.05
    return X, y


# ---------------------------------------------------------------------------
# Interface / inheritance
# ---------------------------------------------------------------------------


class TestLSTMPredictor:
    def test_lstm_inherits_base_predictor(self):
        assert issubclass(LSTMPredictor, BasePredictor)

    def test_name_property(self):
        predictor = LSTMPredictor()
        assert predictor.name == "LSTM"

    def test_predict_before_fit_raises(self, lstm_data):
        X, _ = lstm_data
        predictor = LSTMPredictor()
        with pytest.raises(RuntimeError, match="fit"):
            predictor.predict(X)

    def test_fit_returns_self(self, lstm_data):
        X, y = lstm_data
        predictor = LSTMPredictor(
            max_epochs=2, batch_size=16, hidden_size=8
        )
        result = predictor.fit(X, y)
        assert result is predictor

    def test_default_hyperparameters_match_research(self):
        predictor = LSTMPredictor()
        assert predictor._hidden_size == 32
        assert predictor._num_layers == 1
        assert predictor._dropout == 0.3
        assert predictor._lookback == 6
        assert predictor._learning_rate == 1e-3
        assert predictor._weight_decay == 1e-4
        assert predictor._batch_size == 64
        assert predictor._max_epochs == 100
        assert predictor._patience == 10


# ---------------------------------------------------------------------------
# Shape contracts
# ---------------------------------------------------------------------------


class TestLSTMFitPredict:
    def test_predict_returns_1d_ndarray(self, lstm_data):
        X, y = lstm_data
        predictor = LSTMPredictor(
            max_epochs=3, batch_size=16, hidden_size=8
        )
        predictor.fit(X, y)
        preds = predictor.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)
        assert preds.dtype == np.float64 or np.issubdtype(preds.dtype, np.floating)

    def test_predict_produces_finite_values(self, lstm_data):
        X, y = lstm_data
        predictor = LSTMPredictor(
            max_epochs=2, batch_size=16, hidden_size=8
        )
        predictor.fit(X, y)
        preds = predictor.predict(X)
        assert np.all(np.isfinite(preds))

    def test_seed_reproducibility(self, lstm_data):
        X, y = lstm_data
        predictor_a = LSTMPredictor(
            max_epochs=3, batch_size=16, hidden_size=8, random_state=42
        )
        predictor_b = LSTMPredictor(
            max_epochs=3, batch_size=16, hidden_size=8, random_state=42
        )
        predictor_a.fit(X, y)
        predictor_b.fit(X, y)
        preds_a = predictor_a.predict(X)
        preds_b = predictor_b.predict(X)
        np.testing.assert_array_almost_equal(preds_a, preds_b, decimal=4)

    def test_evaluate_returns_tier1_compatible_keys(self, lstm_data):
        X, y = lstm_data
        predictor = LSTMPredictor(
            max_epochs=2, batch_size=16, hidden_size=8
        )
        predictor.fit(X, y)
        results = predictor.evaluate(X, y)
        expected_keys = {
            "rmse",
            "mae",
            "directional_accuracy",
            "total_pnl",
            "num_trades",
            "win_rate",
            "sharpe_ratio",
            "sharpe_per_trade",
            "pnl_series",
        }
        assert set(results.keys()) == expected_keys

    def test_predict_respects_group_id(self, lstm_data):
        """2 groups x 20 rows each -> returns 40 predictions."""
        X, y = lstm_data
        predictor = LSTMPredictor(
            max_epochs=2, batch_size=16, hidden_size=8
        )
        predictor.fit(X, y)
        preds = predictor.predict(X)
        assert preds.shape == (40,)


# ---------------------------------------------------------------------------
# group_id contract
# ---------------------------------------------------------------------------


class TestLSTMGroupIdContract:
    def test_fit_raises_without_group_id(self, lstm_data):
        X, y = lstm_data
        X_no_gid = X.drop(columns=["group_id"])
        predictor = LSTMPredictor(
            max_epochs=2, batch_size=16, hidden_size=8
        )
        with pytest.raises(ValueError, match="group_id"):
            predictor.fit(X_no_gid, y)

    def test_predict_raises_without_group_id(self, lstm_data):
        X, y = lstm_data
        predictor = LSTMPredictor(
            max_epochs=2, batch_size=16, hidden_size=8
        )
        predictor.fit(X, y)
        X_no_gid = X.drop(columns=["group_id"])
        with pytest.raises(ValueError, match="group_id"):
            predictor.predict(X_no_gid)


# ---------------------------------------------------------------------------
# Short sequences + padding logging
# ---------------------------------------------------------------------------


class TestLSTMPadding:
    def test_fit_with_short_group_sequences_does_not_crash(self):
        """A group shorter than lookback should not cause a crash in fit."""
        rng = np.random.default_rng(99)
        # Group 0: 20 rows (long enough), Group 1: 3 rows (shorter than lookback=6)
        n_long, n_short = 20, 3
        X = pd.DataFrame(
            {
                "group_id": np.array([0] * n_long + [1] * n_short),
                "feat_a": rng.standard_normal(n_long + n_short),
                "feat_b": rng.standard_normal(n_long + n_short),
                "feat_c": rng.standard_normal(n_long + n_short),
            }
        )
        y = rng.standard_normal(n_long + n_short) * 0.05
        predictor = LSTMPredictor(
            max_epochs=2, batch_size=16, hidden_size=8
        )
        # Should not raise
        predictor.fit(X, y)

    def test_padding_logged_for_short_groups(self, lstm_data):
        """predict() on a group shorter than lookback logs it in _padded_pairs."""
        X_train, y_train = lstm_data
        predictor = LSTMPredictor(
            max_epochs=2, batch_size=16, hidden_size=8
        )
        predictor.fit(X_train, y_train)

        # Build a test set where group_id=99 has only 2 rows (< lookback=6)
        # and group_id=99 does NOT exist in the training data
        rng = np.random.default_rng(55)
        short_rows = 2
        X_test = pd.DataFrame(
            {
                "group_id": np.array([0] * 10 + [99] * short_rows),
                "feat_a": rng.standard_normal(10 + short_rows),
                "feat_b": rng.standard_normal(10 + short_rows),
                "feat_c": rng.standard_normal(10 + short_rows),
            }
        )
        predictor.predict(X_test)
        assert 99 in predictor._padded_pairs
