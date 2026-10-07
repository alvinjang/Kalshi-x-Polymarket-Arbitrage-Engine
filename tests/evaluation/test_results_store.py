"""Tests for results_store: save/load JSON evaluation results."""
import json
from pathlib import Path

import pytest

from src.evaluation.results_store import (
    load_all_results,
    load_results,
    save_results,
)


@pytest.fixture
def sample_metrics() -> dict:
    return {
        "rmse": 0.0432,
        "mae": 0.0312,
        "directional_accuracy": 0.621,
        "total_pnl": 1.234,
        "num_trades": 45,
        "win_rate": 0.578,
        "sharpe_ratio": 0.89,
        "pnl_series": [0.1, 0.3, 0.5, 0.7, 1.234],
    }


class TestSaveResults:
    def test_save_writes_json_file(self, tmp_path, sample_metrics):
        path = save_results("Linear Regression", sample_metrics, tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_creates_directory_if_missing(self, tmp_path, sample_metrics):
        nested = tmp_path / "nested" / "results" / "tier1"
        assert not nested.exists()
        path = save_results("Naive", sample_metrics, nested)
        assert path.exists()
        assert nested.exists()

    def test_save_includes_model_name_metrics_and_timestamp(
        self, tmp_path, sample_metrics
    ):
        path = save_results("XGBoost", sample_metrics, tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert data["model_name"] == "XGBoost"
        assert data["metrics"]["rmse"] == 0.0432
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)

    def test_save_slugifies_model_name_in_filename(
        self, tmp_path, sample_metrics
    ):
        path = save_results(
            "Naive (Spread Closes)", sample_metrics, tmp_path
        )
        # Parens and spaces become safe for filesystem
        assert " " not in path.name
        assert "(" not in path.name
        assert ")" not in path.name
        assert path.name.endswith(".json")

    def test_save_accepts_extra_fields(self, tmp_path, sample_metrics):
        extra = {"hyperparameters": {"n_estimators": 200, "max_depth": 4}}
        path = save_results(
            "XGBoost", sample_metrics, tmp_path, extra=extra
        )
        with open(path) as f:
            data = json.load(f)
        assert data["hyperparameters"]["n_estimators"] == 200
        assert data["hyperparameters"]["max_depth"] == 4


class TestLoadResults:
    def test_load_reads_saved_file(self, tmp_path, sample_metrics):
        path = save_results("Linear Regression", sample_metrics, tmp_path)
        loaded = load_results(path)
        assert loaded["model_name"] == "Linear Regression"
        assert loaded["metrics"] == sample_metrics

    def test_round_trip_preserves_data(self, tmp_path, sample_metrics):
        saved_path = save_results("Naive", sample_metrics, tmp_path)
        loaded = load_results(saved_path)
        assert loaded["metrics"]["rmse"] == sample_metrics["rmse"]
        assert loaded["metrics"]["num_trades"] == sample_metrics["num_trades"]
        assert loaded["metrics"]["pnl_series"] == sample_metrics["pnl_series"]


class TestLoadAllResults:
    def test_load_all_returns_list(self, tmp_path, sample_metrics):
        save_results("Naive", sample_metrics, tmp_path)
        save_results("XGBoost", sample_metrics, tmp_path)
        all_results = load_all_results(tmp_path)
        assert isinstance(all_results, list)
        assert len(all_results) == 2

    def test_load_all_sorts_by_model_name(self, tmp_path, sample_metrics):
        save_results("XGBoost", sample_metrics, tmp_path)
        save_results("Linear Regression", sample_metrics, tmp_path)
        save_results("Naive", sample_metrics, tmp_path)
        all_results = load_all_results(tmp_path)
        names = [r["model_name"] for r in all_results]
        assert names == sorted(names)

    def test_load_all_empty_directory_returns_empty_list(self, tmp_path):
        all_results = load_all_results(tmp_path)
        assert all_results == []

    def test_load_all_ignores_non_json_files(self, tmp_path, sample_metrics):
        save_results("Naive", sample_metrics, tmp_path)
        (tmp_path / "README.txt").write_text("not json")
        all_results = load_all_results(tmp_path)
        assert len(all_results) == 1
        assert all_results[0]["model_name"] == "Naive"
