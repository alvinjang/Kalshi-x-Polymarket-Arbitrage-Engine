"""
TDD tests for experiments/run_feature_ablation.py — Phase 12 LOGO ablation.
RED phase: these tests define the expected behavior before implementation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Unit tests for module-level structures (importable after implementation)
# ---------------------------------------------------------------------------

class TestFeatureGroups:
    """ABLA-02: Feature groups sum to 51, non-overlapping."""

    def test_import_feature_groups(self):
        """Module must export FEATURE_GROUPS dict."""
        from experiments.run_feature_ablation import FEATURE_GROUPS
        assert isinstance(FEATURE_GROUPS, dict)

    def test_five_groups_present(self):
        from experiments.run_feature_ablation import FEATURE_GROUPS
        assert set(FEATURE_GROUPS.keys()) == {"A", "B", "C", "D", "E"}

    def test_group_sizes(self):
        from experiments.run_feature_ablation import FEATURE_GROUPS
        assert len(FEATURE_GROUPS["A"]) == 15
        assert len(FEATURE_GROUPS["B"]) == 10
        assert len(FEATURE_GROUPS["C"]) == 6
        assert len(FEATURE_GROUPS["D"]) == 13
        assert len(FEATURE_GROUPS["E"]) == 7

    def test_total_is_51(self):
        from experiments.run_feature_ablation import FEATURE_GROUPS
        total = sum(len(v) for v in FEATURE_GROUPS.values())
        assert total == 51, f"Expected 51 features, got {total}"

    def test_no_overlaps(self):
        from experiments.run_feature_ablation import FEATURE_GROUPS
        all_feats = [f for g in FEATURE_GROUPS.values() for f in g]
        assert len(all_feats) == len(set(all_feats)), "Feature overlap detected"

    def test_group_a_contains_kalshi_vwap(self):
        from experiments.run_feature_ablation import FEATURE_GROUPS
        assert "kalshi_vwap" in FEATURE_GROUPS["A"]

    def test_group_b_contains_spread(self):
        from experiments.run_feature_ablation import FEATURE_GROUPS
        assert "spread" in FEATURE_GROUPS["B"]

    def test_group_d_contains_amihud(self):
        from experiments.run_feature_ablation import FEATURE_GROUPS
        assert "kalshi_amihud" in FEATURE_GROUPS["D"]


class TestTemporalSplit:
    """Three-way split: train_proper 85%, ablation_holdout 15%."""

    def _make_df(self, n: int) -> pd.DataFrame:
        return pd.DataFrame({"time_idx": range(n), "x": range(n)})

    def test_split_sizes(self):
        from experiments.run_feature_ablation import temporal_split
        df = self._make_df(100)
        train_proper, holdout = temporal_split(df, train_frac=0.85)
        assert len(train_proper) == 85
        assert len(holdout) == 15

    def test_split_preserves_order(self):
        from experiments.run_feature_ablation import temporal_split
        df = self._make_df(100)
        train_proper, holdout = temporal_split(df, train_frac=0.85)
        # First row of holdout should follow last row of train_proper
        assert train_proper.iloc[-1]["time_idx"] < holdout.iloc[0]["time_idx"]

    def test_split_no_overlap(self):
        from experiments.run_feature_ablation import temporal_split
        df = self._make_df(1000)
        train_proper, holdout = temporal_split(df, train_frac=0.85)
        train_indices = set(train_proper["time_idx"].tolist())
        holdout_indices = set(holdout["time_idx"].tolist())
        assert train_indices.isdisjoint(holdout_indices)


class TestBootstrapDeltaPnl:
    """ABLA-04: Bootstrap CI for delta-P&L."""

    def test_returns_tuple_of_three(self):
        from experiments.run_feature_ablation import bootstrap_delta_pnl
        rng = np.random.default_rng(42)
        n = 50
        preds_config = np.random.randn(n) * 0.05
        preds_baseline = np.random.randn(n) * 0.05
        actuals = np.random.randn(n) * 0.05
        ci_lo, ci_hi, arr = bootstrap_delta_pnl(preds_config, preds_baseline, actuals, n_boot=100, rng=rng)
        assert isinstance(ci_lo, float)
        assert isinstance(ci_hi, float)
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 100

    def test_ci_lower_le_upper(self):
        from experiments.run_feature_ablation import bootstrap_delta_pnl
        rng = np.random.default_rng(0)
        n = 100
        preds_config = np.random.randn(n) * 0.05
        preds_baseline = np.zeros(n)
        actuals = np.random.randn(n) * 0.05
        ci_lo, ci_hi, _ = bootstrap_delta_pnl(preds_config, preds_baseline, actuals, n_boot=200, rng=rng)
        assert ci_lo <= ci_hi

    def test_zero_delta_baseline_straddles_zero(self):
        """When config == baseline, delta should be ~0 and CI should straddle zero."""
        from experiments.run_feature_ablation import bootstrap_delta_pnl
        rng = np.random.default_rng(42)
        n = 200
        preds = np.random.randn(n) * 0.05
        actuals = np.random.randn(n) * 0.05
        ci_lo, ci_hi, _ = bootstrap_delta_pnl(preds, preds, actuals, n_boot=500, rng=rng)
        assert ci_lo <= 0.0 <= ci_hi, "Identical configs should have CI straddling zero"


# ---------------------------------------------------------------------------
# Integration tests: results files from actual run
# ---------------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "ablation"


@pytest.mark.skipif(
    not (RESULTS_DIR / "summary.json").exists(),
    reason="summary.json not yet generated — run the experiment first"
)
class TestSummaryJson:
    """ABLA-02/03/04: summary.json structure and content."""

    def _load(self):
        return json.loads((RESULTS_DIR / "summary.json").read_text())

    def test_has_split_row_counts(self):
        data = self._load()
        assert "train_proper_rows" in data
        assert "ablation_holdout_rows" in data
        assert "final_test_rows" in data

    def test_train_proper_rows_in_range(self):
        data = self._load()
        assert 5600 <= data["train_proper_rows"] <= 5900

    def test_ablation_holdout_rows_in_range(self):
        data = self._load()
        assert 900 <= data["ablation_holdout_rows"] <= 1100

    def test_final_test_rows(self):
        data = self._load()
        assert data["final_test_rows"] == pytest.approx(1673, abs=100)  # allow small tolerance

    def test_exactly_12_configs(self):
        data = self._load()
        assert len(data["configs"]) == 12

    def test_both_models_present(self):
        data = self._load()
        models = {c["model"] for c in data["configs"]}
        assert "LR" in models
        assert "XGBoost" in models

    def test_six_configs_per_model(self):
        data = self._load()
        for model in ("LR", "XGBoost"):
            count = sum(1 for c in data["configs"] if c["model"] == model)
            assert count == 6, f"Expected 6 configs for {model}, got {count}"

    def test_all_dropped_groups_present(self):
        data = self._load()
        dropped = {c["dropped_group"] for c in data["configs"]}
        expected = {"none", "A", "B", "C", "D", "E"}
        assert dropped == expected

    def test_required_keys_per_config(self):
        data = self._load()
        required = {
            "model", "dropped_group", "feature_count", "pnl", "delta_pnl",
            "rmse", "directional_accuracy", "ci_lower", "ci_upper",
            "num_trades", "num_bootstrap"
        }
        for cfg in data["configs"]:
            missing = required - set(cfg.keys())
            assert not missing, f"Config {cfg.get('model')}/{cfg.get('dropped_group')} missing keys: {missing}"

    def test_baseline_delta_pnl_is_zero(self):
        data = self._load()
        baselines = [c for c in data["configs"] if c["dropped_group"] == "none"]
        for b in baselines:
            assert b["delta_pnl"] == 0.0

    def test_bootstrap_count_is_1000(self):
        data = self._load()
        for cfg in data["configs"]:
            if cfg["dropped_group"] != "none":
                assert cfg["num_bootstrap"] == 1000


@pytest.mark.skipif(
    not (RESULTS_DIR / "report.md").exists(),
    reason="report.md not yet generated"
)
class TestReportMd:
    """ABLA-05: report.md has exactly 12 data rows."""

    def test_12_data_rows(self):
        text = (RESULTS_DIR / "report.md").read_text()
        # Count lines starting with "| " that are not header/separator
        data_rows = [
            line for line in text.splitlines()
            if line.startswith("| ") and not line.startswith("| Model") and "---" not in line
        ]
        assert len(data_rows) == 12, f"Expected 12 data rows, got {len(data_rows)}"

    def test_contains_lr_and_xgboost(self):
        text = (RESULTS_DIR / "report.md").read_text()
        assert "LR" in text
        assert "XGBoost" in text


@pytest.mark.skipif(
    not (RESULTS_DIR / "per_config.csv").exists(),
    reason="per_config.csv not yet generated"
)
class TestPerConfigCsv:
    """per_config.csv has exactly 12 data rows (13 lines including header)."""

    def test_13_lines(self):
        text = (RESULTS_DIR / "per_config.csv").read_text()
        lines = [l for l in text.splitlines() if l.strip()]
        assert len(lines) == 13, f"Expected 13 lines (header + 12 rows), got {len(lines)}"

    def test_has_required_columns(self):
        df = pd.read_csv(RESULTS_DIR / "per_config.csv")
        required_cols = {"model", "dropped_group", "feature_count", "pnl", "delta_pnl",
                         "rmse", "directional_accuracy", "ci_lower", "ci_upper"}
        missing = required_cols - set(df.columns)
        assert not missing, f"CSV missing columns: {missing}"
