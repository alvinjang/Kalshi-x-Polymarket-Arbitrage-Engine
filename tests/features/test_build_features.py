"""Integration tests for the feature engineering CLI pipeline."""
import numpy as np
import pandas as pd
import pytest


class TestBuildFeaturePipeline:
    """Tests for src/features/build_features.build_feature_pipeline."""

    def test_produces_train_and_test_parquets(self, aligned_pairs_df, tmp_path):
        """Pipeline saves train.parquet and test.parquet."""
        from src.features.build_features import build_feature_pipeline

        input_path = tmp_path / "aligned_pairs.parquet"
        aligned_pairs_df.to_parquet(input_path, index=False)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        build_feature_pipeline(
            input_path=str(input_path),
            output_dir=str(output_dir),
            split_ratio=0.8,
        )

        assert (output_dir / "train.parquet").exists()
        assert (output_dir / "test.parquet").exists()

    def test_output_has_39_columns(self, aligned_pairs_df, tmp_path):
        """train.parquet and test.parquet have 37 features + time_idx + group_id = 39 columns."""
        from src.features.build_features import build_feature_pipeline

        input_path = tmp_path / "aligned_pairs.parquet"
        aligned_pairs_df.to_parquet(input_path, index=False)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        train, test = build_feature_pipeline(
            input_path=str(input_path),
            output_dir=str(output_dir),
            split_ratio=0.8,
        )

        assert len(train.columns) == 39, f"Expected 39 columns, got {len(train.columns)}"
        assert len(test.columns) == 39, f"Expected 39 columns, got {len(test.columns)}"

    def test_has_time_idx_and_group_id(self, aligned_pairs_df, tmp_path):
        """Output DataFrames contain time_idx and group_id columns."""
        from src.features.build_features import build_feature_pipeline

        input_path = tmp_path / "aligned_pairs.parquet"
        aligned_pairs_df.to_parquet(input_path, index=False)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        train, test = build_feature_pipeline(
            input_path=str(input_path),
            output_dir=str(output_dir),
            split_ratio=0.8,
        )

        assert "time_idx" in train.columns
        assert "group_id" in train.columns
        assert "time_idx" in test.columns
        assert "group_id" in test.columns

    def test_has_derived_features(self, aligned_pairs_df, tmp_path):
        """Output DataFrames contain the 6 derived feature columns."""
        from src.features.build_features import build_feature_pipeline

        input_path = tmp_path / "aligned_pairs.parquet"
        aligned_pairs_df.to_parquet(input_path, index=False)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        train, test = build_feature_pipeline(
            input_path=str(input_path),
            output_dir=str(output_dir),
            split_ratio=0.8,
        )

        expected_features = [
            "spread_momentum", "spread_volatility",
            "price_velocity", "volume_ratio",
            "kalshi_order_flow_imbalance", "polymarket_order_flow_imbalance",
        ]
        for feat in expected_features:
            assert feat in train.columns, f"Missing derived feature: {feat}"

    def test_temporal_invariant(self, aligned_pairs_df, tmp_path):
        """No training timestamp exceeds earliest test timestamp within any pair."""
        from src.features.build_features import build_feature_pipeline

        input_path = tmp_path / "aligned_pairs.parquet"
        aligned_pairs_df.to_parquet(input_path, index=False)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        train, test = build_feature_pipeline(
            input_path=str(input_path),
            output_dir=str(output_dir),
            split_ratio=0.8,
        )

        for pid in train["pair_id"].unique():
            t_max = train[train["pair_id"] == pid]["timestamp"].max()
            if pid in test["pair_id"].unique():
                t_min = test[test["pair_id"] == pid]["timestamp"].min()
                assert t_max < t_min, (
                    f"Temporal leak in {pid}: train max {t_max} >= test min {t_min}"
                )

    def test_saved_parquets_match_returned_dfs(self, aligned_pairs_df, tmp_path):
        """Saved parquet files match the returned DataFrames."""
        from src.features.build_features import build_feature_pipeline

        input_path = tmp_path / "aligned_pairs.parquet"
        aligned_pairs_df.to_parquet(input_path, index=False)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        train, test = build_feature_pipeline(
            input_path=str(input_path),
            output_dir=str(output_dir),
            split_ratio=0.8,
        )

        saved_train = pd.read_parquet(output_dir / "train.parquet")
        saved_test = pd.read_parquet(output_dir / "test.parquet")

        pd.testing.assert_frame_equal(train.reset_index(drop=True), saved_train)
        pd.testing.assert_frame_equal(test.reset_index(drop=True), saved_test)
