"""Tests for temporal split, timeseries columns, and TimeSeriesDataSet builder."""
import numpy as np
import pandas as pd
import pytest

from tests.features.conftest import BASE_TIMESTAMP, BAR_SECONDS


class TestTemporalTrainTestSplit:
    """Tests for src/features/dataset.temporal_train_test_split."""

    def test_max_train_timestamp_lt_min_test_per_pair(self, aligned_pairs_df):
        """For each pair_id, max(train.timestamp) < min(test.timestamp)."""
        from src.features.dataset import temporal_train_test_split
        from src.features.engineering import compute_derived_features

        df = compute_derived_features(aligned_pairs_df)
        train, test = temporal_train_test_split(df, split_ratio=0.8)

        for pid in train["pair_id"].unique():
            t_max = train[train["pair_id"] == pid]["timestamp"].max()
            if pid in test["pair_id"].unique():
                t_min = test[test["pair_id"] == pid]["timestamp"].min()
                assert t_max < t_min, (
                    f"Temporal leak in {pid}: train max {t_max} >= test min {t_min}"
                )

    def test_per_pair_split_not_global_cutoff(self, aligned_pairs_df):
        """Split uses per-pair chronological boundary, not a single global timestamp."""
        from src.features.dataset import temporal_train_test_split
        from src.features.engineering import compute_derived_features

        df = compute_derived_features(aligned_pairs_df)
        train, test = temporal_train_test_split(df, split_ratio=0.8)

        # Each pair should have its own split point
        # With 5 rows per pair and 0.8 ratio: 4 train + 1 test
        for pid in aligned_pairs_df["pair_id"].unique():
            pair_train = train[train["pair_id"] == pid]
            pair_test = test[test["pair_id"] == pid]
            assert len(pair_train) == 4, f"Pair {pid} should have 4 train rows"
            assert len(pair_test) == 1, f"Pair {pid} should have 1 test row"

    def test_split_ratio_controls_proportions(self, aligned_pairs_df):
        """Different split ratios produce different train/test sizes."""
        from src.features.dataset import temporal_train_test_split
        from src.features.engineering import compute_derived_features

        df = compute_derived_features(aligned_pairs_df)

        train_80, test_80 = temporal_train_test_split(df, split_ratio=0.8)
        train_60, test_60 = temporal_train_test_split(df, split_ratio=0.6)

        assert len(train_80) > len(train_60)
        assert len(test_80) < len(test_60)

    def test_all_rows_accounted_for(self, aligned_pairs_df):
        """Train + test row count equals input row count."""
        from src.features.dataset import temporal_train_test_split
        from src.features.engineering import compute_derived_features

        df = compute_derived_features(aligned_pairs_df)
        train, test = temporal_train_test_split(df, split_ratio=0.8)

        assert len(train) + len(test) == len(df)


class TestAddTimeseriesColumns:
    """Tests for src/features/dataset.add_timeseries_columns."""

    def test_adds_time_idx_column(self, aligned_pairs_df):
        """add_timeseries_columns adds a 'time_idx' column."""
        from src.features.dataset import add_timeseries_columns
        from src.features.engineering import compute_derived_features

        df = compute_derived_features(aligned_pairs_df)
        result = add_timeseries_columns(df)
        assert "time_idx" in result.columns

    def test_adds_group_id_column(self, aligned_pairs_df):
        """add_timeseries_columns adds a 'group_id' column."""
        from src.features.dataset import add_timeseries_columns
        from src.features.engineering import compute_derived_features

        df = compute_derived_features(aligned_pairs_df)
        result = add_timeseries_columns(df)
        assert "group_id" in result.columns

    def test_time_idx_is_0_based_contiguous_per_group(self, aligned_pairs_df):
        """time_idx is 0, 1, 2, ..., N-1 within each group_id."""
        from src.features.dataset import add_timeseries_columns
        from src.features.engineering import compute_derived_features

        df = compute_derived_features(aligned_pairs_df)
        result = add_timeseries_columns(df)

        for gid in result["group_id"].unique():
            group = result[result["group_id"] == gid]
            expected = list(range(len(group)))
            actual = list(group["time_idx"])
            assert actual == expected, f"group_id {gid}: expected {expected}, got {actual}"

    def test_group_id_is_int(self, aligned_pairs_df):
        """group_id is an integer type."""
        from src.features.dataset import add_timeseries_columns
        from src.features.engineering import compute_derived_features

        df = compute_derived_features(aligned_pairs_df)
        result = add_timeseries_columns(df)
        assert np.issubdtype(result["group_id"].dtype, np.integer)

    def test_time_idx_is_int(self, aligned_pairs_df):
        """time_idx is an integer type."""
        from src.features.dataset import add_timeseries_columns
        from src.features.engineering import compute_derived_features

        df = compute_derived_features(aligned_pairs_df)
        result = add_timeseries_columns(df)
        assert np.issubdtype(result["time_idx"].dtype, np.integer)


class TestBuildTimeseriesDataset:
    """Tests for src/features/dataset.build_timeseries_dataset."""

    def test_returns_timeseries_dataset(self, aligned_pairs_df):
        """build_timeseries_dataset returns a pytorch_forecasting.TimeSeriesDataSet."""
        from pytorch_forecasting import TimeSeriesDataSet
        from src.features.dataset import (
            add_timeseries_columns,
            build_timeseries_dataset,
        )
        from src.features.engineering import compute_derived_features

        df = compute_derived_features(aligned_pairs_df)
        df = add_timeseries_columns(df)
        # Drop NaN spread rows since TimeSeriesDataSet needs valid targets
        df = df.dropna(subset=["spread"]).reset_index(drop=True)
        # Recompute time_idx after dropping rows
        df = df.drop(columns=["time_idx", "group_id"])
        df = add_timeseries_columns(df)

        ds = build_timeseries_dataset(df, max_encoder_length=3, max_prediction_length=1)
        assert isinstance(ds, TimeSeriesDataSet)

    def test_dataset_target_is_spread(self, aligned_pairs_df):
        """TimeSeriesDataSet target is 'spread'."""
        from src.features.dataset import (
            add_timeseries_columns,
            build_timeseries_dataset,
        )
        from src.features.engineering import compute_derived_features

        df = compute_derived_features(aligned_pairs_df)
        df = add_timeseries_columns(df)
        df = df.dropna(subset=["spread"]).reset_index(drop=True)
        df = df.drop(columns=["time_idx", "group_id"])
        df = add_timeseries_columns(df)

        ds = build_timeseries_dataset(df, max_encoder_length=3, max_prediction_length=1)
        assert ds.target == "spread" or ds.target_names == ["spread"]
