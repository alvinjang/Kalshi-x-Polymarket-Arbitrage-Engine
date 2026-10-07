"""Tests for the retraining policy — pure function, no I/O."""
from __future__ import annotations

import pytest

from src.experiments.retraining_policy import (
    DataScalingCheckpoint,
    next_retraining_actions,
    should_run_scaling_experiment,
)


class TestNextRetrainingActions:
    """Each model tier has a minimum bars-per-pair + positions trigger.

    The policy function is PURE: given current data volume, return which
    tiers should retrain. No side effects, no I/O.
    """

    def test_no_data_no_training(self):
        actions = next_retraining_actions(bars_per_pair=0, completed_positions=0)
        assert actions["linear_regression"] is False
        assert actions["xgboost"] is False
        assert actions["gru"] is False
        assert actions["lstm"] is False
        assert actions["ppo_raw"] is False
        assert actions["ppo_filtered"] is False

    def test_tier1_unlocks_early(self):
        """LR + XGBoost are cheap; they should retrain with even small data."""
        actions = next_retraining_actions(bars_per_pair=20, completed_positions=0)
        assert actions["linear_regression"] is True
        assert actions["xgboost"] is True
        # Sequence models still locked
        assert actions["gru"] is False
        assert actions["lstm"] is False
        # PPO needs completed positions, not bars
        assert actions["ppo_raw"] is False

    def test_tier2_needs_100_bars(self):
        """GRU/LSTM need at least ~100 bars per pair for sequence learning."""
        locked = next_retraining_actions(bars_per_pair=50, completed_positions=0)
        assert locked["gru"] is False
        assert locked["lstm"] is False

        unlocked = next_retraining_actions(bars_per_pair=100, completed_positions=0)
        assert unlocked["gru"] is True
        assert unlocked["lstm"] is True

    def test_ppo_needs_completed_positions(self):
        """PPO policies need completed trajectories, not just bars."""
        # Lots of bars but no positions closed yet
        locked = next_retraining_actions(bars_per_pair=1000, completed_positions=0)
        assert locked["ppo_raw"] is False
        assert locked["ppo_filtered"] is False

        # Enough trajectories
        unlocked = next_retraining_actions(
            bars_per_pair=1000, completed_positions=500
        )
        assert unlocked["ppo_raw"] is True
        assert unlocked["ppo_filtered"] is True


class TestShouldRunScalingExperiment:
    """The scaling experiment should run at checkpoint boundaries only.

    Goal: plot performance at 50, 100, 250, 500, 1000, 2000 bars/pair.
    Don't re-run the same checkpoint twice (wastes compute).
    """

    def test_runs_at_checkpoint_boundary(self):
        # We've accumulated 105 bars/pair, last checkpoint run was at 50
        result = should_run_scaling_experiment(
            current_bars_per_pair=105,
            last_checkpoint_ran=50,
        )
        assert result is True
        # After running, last_checkpoint_ran advances to 100

    def test_skips_between_checkpoints(self):
        # We've accumulated 130 bars/pair, already ran at 100
        result = should_run_scaling_experiment(
            current_bars_per_pair=130,
            last_checkpoint_ran=100,
        )
        assert result is False

    def test_runs_on_next_threshold(self):
        # We were at 100, now we're at 255 — time for 250 checkpoint
        result = should_run_scaling_experiment(
            current_bars_per_pair=255,
            last_checkpoint_ran=100,
        )
        assert result is True

    def test_no_run_at_zero(self):
        result = should_run_scaling_experiment(
            current_bars_per_pair=10,
            last_checkpoint_ran=0,
        )
        assert result is False


class TestDataScalingCheckpoint:
    """Checkpoint is a persistable record of one experiment run."""

    def test_round_trip(self):
        cp = DataScalingCheckpoint(
            bars_per_pair=100,
            training_rows=5432,
            timestamp="2026-04-10T12:00:00Z",
            metrics_by_model={
                "linear_regression": {"rmse": 0.12, "dir_acc": 0.54, "pnl_at_2pp": 3.5},
                "xgboost": {"rmse": 0.11, "dir_acc": 0.55, "pnl_at_2pp": 4.1},
            },
        )
        d = cp.to_dict()
        assert d["bars_per_pair"] == 100
        assert d["training_rows"] == 5432
        assert "linear_regression" in d["metrics_by_model"]

        # Round-trip through dict
        cp2 = DataScalingCheckpoint.from_dict(d)
        assert cp2.bars_per_pair == cp.bars_per_pair
        assert cp2.metrics_by_model == cp.metrics_by_model
