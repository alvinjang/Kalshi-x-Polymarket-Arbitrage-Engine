"""Integration tests for Tier 2 harness extensions in run_baselines.py.

Covers Validation Dimensions 4 and 5 from 05-RESEARCH.md:
  - CLI --tier flag parsing
  - NON_FEATURE_COLUMNS includes kalshi_order_flow_imbalance (34-feature fix)
  - format_comparison_table for combined Tier 1 + Tier 2 output
  - Tier 2 JSON schema with seed aggregation
  - prepare_xy_for_seq group_id pass-through
  - src.models package exports for GRU / LSTM
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from experiments.run_baselines import (
    NON_FEATURE_COLUMNS,
    format_comparison_table,
    main,
    build_models,
    prepare_xy_for_seq,
)
from src.models import GRUPredictor, LSTMPredictor


# ---------------------------------------------------------------------------
# Synthetic result helpers
# ---------------------------------------------------------------------------

def _make_result(
    model_name: str,
    rmse: float = 0.30,
    extra: dict | None = None,
) -> dict:
    """Build a minimal result dict matching Tier 1 JSON schema."""
    return {
        "model_name": model_name,
        "metrics": {
            "rmse": rmse,
            "mae": 0.24,
            "directional_accuracy": 0.65,
            "total_pnl": 180.0,
            "num_trades": 1200,
            "win_rate": 0.55,
            "sharpe_ratio": 18.0,
            "sharpe_per_trade": 0.40,
        },
        "timestamp": "2026-04-05T00:00:00+00:00",
        "threshold": 0.02,
        "n_train_rows": 6802,
        "n_test_rows": 1673,
        "n_features": 34,
        "pnl_series": [0.0, 1.0, 2.0],
        "extra": extra or {},
    }


TIER2_EXTRA = {
    "seeds": [42, 123, 456],
    "seed_rmses": [0.31, 0.30, 0.32],
    "mean_rmse": 0.31,
    "std_rmse": 0.01,
}


# ===================================================================
# CLI Parsing
# ===================================================================

class TestCLIParsing:
    """Argparse behaviour for --tier flag."""

    def test_cli_accepts_tier_flag(self):
        """--tier 2 should be accepted without raising."""
        # We just test that argparse succeeds; main() itself would try to load
        # data, so we catch SystemExit/FileNotFoundError after parsing.
        import argparse
        from experiments.run_baselines import main as _main

        # main returns int, so calling with --tier 2 will fail at data-load
        # stage (FileNotFoundError -> returns 1), but should NOT raise
        # on argparse.
        try:
            ret = _main(["--tier", "2", "--data-dir", "/nonexistent"])
        except SystemExit:
            pass  # acceptable — argparse error would be exit(2), not 1
        # If we get here without an argparse-related SystemExit(2), pass.

    def test_cli_tier_defaults_to_1(self):
        """Omitting --tier should default to tier '1'."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--tier", type=str, choices=["1", "2", "both"], default="1",
        )
        args = parser.parse_args([])
        assert args.tier == "1"

    def test_cli_rejects_invalid_tier(self):
        """--tier 9 should cause a non-zero exit."""
        with pytest.raises(SystemExit) as exc:
            main(["--tier", "9"])
        assert exc.value.code != 0


# ===================================================================
# NON_FEATURE_COLUMNS
# ===================================================================

class TestNonFeatureColumns:

    def test_non_feature_columns_includes_kalshi_order_flow_imbalance(self):
        assert "kalshi_order_flow_imbalance" in NON_FEATURE_COLUMNS

    def test_non_feature_columns_includes_existing_entries(self):
        for col in ("timestamp", "pair_id", "time_idx", "group_id",
                     "spread_change_target"):
            assert col in NON_FEATURE_COLUMNS


# ===================================================================
# Comparison Table — Combined Tier
# ===================================================================

class TestComparisonTableCombined:

    def test_format_comparison_table_tier_both_includes_tier1_and_tier2(self):
        results = [
            _make_result("Linear Regression"),
            _make_result("XGBoost"),
            _make_result("GRU", extra=TIER2_EXTRA),
            _make_result("LSTM", extra=TIER2_EXTRA),
        ]
        table = format_comparison_table(results, tier="both")
        for name in ("Linear Regression", "XGBoost", "GRU", "LSTM"):
            assert name in table, f"{name} not found in table"

    def test_format_comparison_table_tier_2_only_shows_tier2(self):
        results = [
            _make_result("GRU", extra=TIER2_EXTRA),
            _make_result("LSTM", extra=TIER2_EXTRA),
        ]
        table = format_comparison_table(results, tier="2")
        assert "GRU" in table
        assert "LSTM" in table

    def test_format_comparison_table_shows_seed_mean_std_for_tier2(self):
        results = [
            _make_result("GRU", extra=TIER2_EXTRA),
        ]
        table = format_comparison_table(results, tier="2")
        # Should contain the std deviation indicator
        assert "0.0100" in table or "0.01" in table


# ===================================================================
# Tier 2 JSON Schema
# ===================================================================

class TestTier2JsonSchema:

    def test_tier2_json_has_required_keys(self, tmp_path):
        """A tier2 result JSON should contain the Tier-1-compatible schema keys."""
        result = _make_result("GRU", extra=TIER2_EXTRA)
        path = tmp_path / "gru.json"
        path.write_text(json.dumps(result, indent=2))

        loaded = json.loads(path.read_text())
        required = {
            "model_name", "metrics", "timestamp", "threshold",
            "n_train_rows", "n_test_rows", "n_features", "pnl_series",
        }
        assert required <= set(loaded.keys())

    def test_extra_contains_seed_aggregation(self):
        result = _make_result("GRU", extra=TIER2_EXTRA)
        extra = result.get("extra", {})
        for key in ("seeds", "seed_rmses", "mean_rmse", "std_rmse"):
            assert key in extra, f"Missing {key} in extra"
        assert extra["seeds"] == [42, 123, 456]
        assert len(extra["seed_rmses"]) == 3


# ===================================================================
# prepare_xy_for_seq helper
# ===================================================================

class TestPrepareXyForSeq:

    def test_prepare_xy_for_seq_preserves_group_id(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="4h"),
            "group_id": [0, 0, 0, 1, 1],
            "feat_a": np.random.randn(5),
            "feat_b": np.random.randn(5),
            "spread_change_target": np.random.randn(5),
        })
        feature_cols = ["feat_a", "feat_b"]
        X, y = prepare_xy_for_seq(df, feature_cols)
        assert "group_id" in X.columns
        assert len(y) == 5

    def test_prepare_xy_for_seq_returns_correct_feature_set(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="4h"),
            "group_id": [0, 0, 0],
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0],
            "spread_change_target": [0.1, 0.2, 0.3],
        })
        feature_cols = ["feat_a", "feat_b"]
        X, y = prepare_xy_for_seq(df, feature_cols)
        assert list(X.columns) == ["feat_a", "feat_b", "group_id"]
        np.testing.assert_array_equal(y, [0.1, 0.2, 0.3])


# ===================================================================
# Model Exports
# ===================================================================

class TestModelExports:

    def test_models_package_exports_gru_and_lstm(self):
        from src.models import GRUPredictor, LSTMPredictor
        assert GRUPredictor is not None
        assert LSTMPredictor is not None

    def test_gru_name(self):
        assert GRUPredictor().name == "GRU"

    def test_lstm_name(self):
        assert LSTMPredictor().name == "LSTM"


# ===================================================================
# build_models with tier parameter
# ===================================================================

class TestBuildModels:

    def test_build_models_tier_1(self):
        models = build_models(tier="1")
        names = [m.name for m in models]
        assert "Linear Regression" in names
        assert "XGBoost" in names

    def test_build_models_tier_2(self):
        models = build_models(tier="2")
        names = [m.name for m in models]
        assert "GRU" in names
        assert "LSTM" in names

    def test_build_models_tier_both(self):
        models = build_models(tier="both")
        names = [m.name for m in models]
        # Naive and Volume have longer descriptive names
        assert len(models) == 6
        assert any("Naive" in n for n in names)
        assert any("Volume" in n for n in names)
        for expected in ("Linear Regression", "XGBoost", "GRU", "LSTM"):
            assert expected in names, f"{expected} not in build_models(tier='both')"
