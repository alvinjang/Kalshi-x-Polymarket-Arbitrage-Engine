"""Tests for shared sequence model utilities.

TDD RED phase: all tests import from src.models.sequence_utils which
does not yet exist — test collection must fail with ImportError.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.models.sequence_utils import (
    EarlyStopping,
    apply_feature_scaler,
    create_sequences,
    fit_feature_scaler,
    get_device,
    set_seed,
)


# ---------------------------------------------------------------------------
# create_sequences
# ---------------------------------------------------------------------------


class TestCreateSequences:
    """Windowing utility respects pair_id boundaries and aligns targets."""

    def test_create_sequences_shape(self):
        """30-row input, lookback=6, single group -> (25, 6, 3) X, (25,) y."""
        rng = np.random.default_rng(42)
        n_rows, n_features, lookback = 30, 3, 6
        X = rng.standard_normal((n_rows, n_features))
        y = rng.standard_normal(n_rows)
        group_ids = np.zeros(n_rows, dtype=int)

        X_seq, y_seq = create_sequences(X, y, lookback, group_ids)

        assert X_seq.shape == (n_rows - lookback + 1, lookback, n_features)
        assert y_seq.shape == (n_rows - lookback + 1,)

    def test_create_sequences_respects_group_boundaries(self):
        """2 pairs x 20 rows, lookback=6 -> 2*(20-6+1)=30 windows, no cross."""
        rng = np.random.default_rng(42)
        n_per_pair, n_features, lookback = 20, 3, 6
        # Build X with pair_id encoded as the first feature column for verification
        X_pair0 = np.column_stack([
            np.zeros(n_per_pair),          # pair_id marker = 0
            rng.standard_normal((n_per_pair, n_features - 1)),
        ])
        X_pair1 = np.column_stack([
            np.ones(n_per_pair),           # pair_id marker = 1
            rng.standard_normal((n_per_pair, n_features - 1)),
        ])
        X = np.vstack([X_pair0, X_pair1])
        y = rng.standard_normal(2 * n_per_pair)
        group_ids = np.array([0] * n_per_pair + [1] * n_per_pair)

        X_seq, y_seq = create_sequences(X, y, lookback, group_ids)

        expected_windows = 2 * (n_per_pair - lookback + 1)
        assert X_seq.shape[0] == expected_windows
        assert y_seq.shape[0] == expected_windows

        # Verify no window crosses a group boundary: within each window,
        # the pair_id marker (column 0) must be constant.
        for i in range(X_seq.shape[0]):
            markers = X_seq[i, :, 0]  # all lookback rows, feature 0
            assert np.all(markers == markers[0]), (
                f"Window {i} crosses pair boundary: markers={markers}"
            )

    def test_create_sequences_too_short_pair_skipped(self):
        """Pair with 3 rows < lookback=6 produces 0 windows."""
        rng = np.random.default_rng(42)
        n_features, lookback = 3, 6
        short_rows = 3  # < lookback
        full_rows = 20

        X_short = rng.standard_normal((short_rows, n_features))
        X_full = rng.standard_normal((full_rows, n_features))
        X = np.vstack([X_short, X_full])
        y = rng.standard_normal(short_rows + full_rows)
        group_ids = np.array(
            [0] * short_rows + [1] * full_rows
        )

        X_seq, y_seq = create_sequences(X, y, lookback, group_ids)

        # Only the full pair contributes windows
        expected = full_rows - lookback + 1
        assert X_seq.shape[0] == expected
        assert y_seq.shape[0] == expected

    def test_create_sequences_target_alignment(self):
        """y_seq[i] equals input y at the last row of window i."""
        rng = np.random.default_rng(42)
        n_rows, n_features, lookback = 15, 2, 4
        X = rng.standard_normal((n_rows, n_features))
        # Use recognizable y values: y[i] = i * 10
        y = np.arange(n_rows, dtype=float) * 10.0
        group_ids = np.zeros(n_rows, dtype=int)

        X_seq, y_seq = create_sequences(X, y, lookback, group_ids)

        # Window i uses rows [i, i+lookback-1] inclusive, target = y[i+lookback-1]
        for i in range(len(y_seq)):
            expected_target = y[i + lookback - 1]
            assert y_seq[i] == pytest.approx(expected_target), (
                f"Window {i}: y_seq={y_seq[i]}, expected y[{i + lookback - 1}]={expected_target}"
            )


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    """Patience-based early stopping tracks validation loss."""

    def test_early_stopping_triggers_after_patience(self):
        """patience=3, min_delta=0.01: no step improves > min_delta from best."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        # All losses after step 0 are within min_delta of best (0.5):
        # none satisfy val_loss < 0.5 - 0.01 = 0.49, so counter increments each time.
        losses = [0.5, 0.499, 0.498, 0.497]
        # Step 0: 0.5 < inf-0.01 → best=0.5, counter=0, False
        # Step 1: 0.499 < 0.49? No → counter=1, False
        # Step 2: 0.498 < 0.49? No → counter=2, False
        # Step 3: 0.497 < 0.49? No → counter=3 >= 3 → True
        results = [es.step(l) for l in losses]
        assert results[0] is False
        assert results[1] is False
        assert results[2] is False
        assert results[3] is True

    def test_early_stopping_resets_on_improvement(self):
        """patience=2: big improvement resets counter."""
        es = EarlyStopping(patience=2, min_delta=0.01)
        losses = [0.5, 0.3, 0.31, 0.1, 0.11, 0.12]
        # Step 0: 0.5 < inf-0.01 → best=0.5, counter=0, False
        # Step 1: 0.3 < 0.5-0.01=0.49 → best=0.3, counter=0, False
        # Step 2: 0.31 < 0.3-0.01=0.29? No → counter=1, False
        # Step 3: 0.1 < 0.29? Yes → best=0.1, counter=0, False (RESET!)
        # Step 4: 0.11 < 0.1-0.01=0.09? No → counter=1, False
        # Step 5: 0.12 < 0.09? No → counter=2 >= 2 → True
        results = [es.step(l) for l in losses]
        # Should NOT trigger early (steps 0-4 all False)
        assert all(r is False for r in results[:5]), (
            f"Should not trigger before step 5, got: {results}"
        )
        # Step 5 should trigger
        assert results[5] is True

    def test_early_stopping_initial_steps_return_false(self):
        """First call to .step() never returns True."""
        es = EarlyStopping(patience=1, min_delta=0.0)
        assert es.step(100.0) is False


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------


class TestSetSeed:
    """Reproducibility seeding for numpy and torch."""

    def test_set_seed_makes_numpy_reproducible(self):
        """Same seed -> same numpy random output."""
        set_seed(42)
        a1 = np.random.rand(5)
        set_seed(42)
        a2 = np.random.rand(5)
        np.testing.assert_array_equal(a1, a2)

    def test_set_seed_makes_torch_reproducible(self):
        """Same seed -> same torch random output."""
        set_seed(42)
        t1 = torch.randn(5)
        set_seed(42)
        t2 = torch.randn(5)
        assert torch.equal(t1, t2)


# ---------------------------------------------------------------------------
# get_device
# ---------------------------------------------------------------------------


class TestGetDevice:
    """Device selection helper."""

    def test_get_device_returns_torch_device(self):
        """Returns a torch.device instance."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_get_device_type_is_cpu_or_cuda(self):
        """Device type is one of the expected values."""
        device = get_device()
        assert device.type in {"cpu", "cuda"}


# ---------------------------------------------------------------------------
# fit_feature_scaler / apply_feature_scaler
# ---------------------------------------------------------------------------


class TestFeatureScaler:
    """StandardScaler helpers with bool casting and zero-variance guard."""

    def test_fit_feature_scaler_casts_bools_before_fit(self):
        """Bool column does not cause dtype errors during scaler fit."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "price": rng.standard_normal(50),
            "volume": rng.standard_normal(50) + 10,
            "has_trade": rng.choice([True, False], 50),
        })
        # Should not raise
        scaler = fit_feature_scaler(df, bool_cols=["has_trade"])
        assert scaler is not None

    def test_apply_scaler_produces_zero_mean_unit_std_on_train(self):
        """Fit+apply on same data -> mean ~0, std ~1 per column."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": rng.standard_normal(100) * 5 + 20,
            "b": rng.standard_normal(100) * 2 - 3,
            "flag": rng.choice([True, False], 100),
        })
        scaler = fit_feature_scaler(df, bool_cols=["flag"])
        result = apply_feature_scaler(df, scaler, bool_cols=["flag"])

        assert result.shape == (100, 3)
        np.testing.assert_allclose(result.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(result.std(axis=0, ddof=0), 1.0, atol=1e-6)

    def test_apply_scaler_does_not_refit_on_test(self):
        """Scaler uses train stats, not test stats."""
        rng = np.random.default_rng(42)
        df_train = pd.DataFrame({
            "x": rng.standard_normal(100) * 2 + 10,
            "y": rng.standard_normal(100) * 3 + 5,
        })
        df_test = pd.DataFrame({
            "x": rng.standard_normal(50) * 1 + 0,
            "y": rng.standard_normal(50) * 1 + 0,
        })
        scaler = fit_feature_scaler(df_train, bool_cols=[])
        result = apply_feature_scaler(df_test, scaler, bool_cols=[])

        # If scaler refitted on test, mean would be ~0. With train stats, it won't be.
        assert not np.allclose(result.mean(axis=0), 0.0, atol=0.5), (
            "Output should NOT be zero-mean when applied with train stats to different test dist"
        )

    def test_fit_feature_scaler_raises_on_zero_variance(self):
        """Zero-variance column -> ValueError mentioning the column name."""
        df = pd.DataFrame({
            "good_col": np.random.default_rng(42).standard_normal(20),
            "zero_var_col": np.full(20, 5.0),
        })
        with pytest.raises(ValueError, match="zero_var_col"):
            fit_feature_scaler(df, bool_cols=[])
