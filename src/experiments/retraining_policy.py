"""Retraining policy and data-scaling experiment infrastructure.

Answers two related questions:

1. **When should each model tier retrain?**
   Each tier has a minimum data threshold below which training is pure
   noise. Linear models benefit from any new data. Sequence models
   (GRU/LSTM) need ~100 bars/pair before they can learn meaningful
   temporal patterns. PPO needs completed position trajectories, not
   just raw bars.

2. **How does model performance scale with data?**
   This is the central research question from CLAUDE.md: *does increasing
   model complexity improve arbitrage detection, and when is that
   complexity justified?* Empirically, the answer is "it depends on how
   much data you have." We run a scaling experiment at checkpoint
   boundaries (50, 100, 250, 500, 1000, 2000 bars/pair) and plot each
   tier's performance vs training data size. If GRU/LSTM never beat LR
   across the whole curve, complexity was never justified at this scale.

The policy logic is pure (no I/O) so it can be unit-tested and reused
from both the trading cycle and offline experiment runners.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# Minimum bars-per-pair before each model tier produces signal above noise.
# These are conservative empirical floors — below them, the model just
# overfits or memorizes.
MIN_BARS_TIER1 = 10      # LR / XGBoost / Naive / Volume
MIN_BARS_TIER2 = 100     # GRU / LSTM
MIN_POSITIONS_TIER3 = 500  # PPO needs closed trajectories

# Checkpoint boundaries for the data-scaling experiment.
SCALING_CHECKPOINTS: tuple[int, ...] = (50, 100, 250, 500, 1000, 2000)


def next_retraining_actions(
    pairs_at_threshold: dict[str, int],
    completed_positions: int,
    min_pairs: int = 20,
) -> dict[str, bool]:
    """Return per-tier retraining eligibility.

    Args:
        pairs_at_threshold: Dict with keys 'tier1' and 'tier2' mapping
            to the number of pairs that have enough bars for that tier.
            Replaces the old median-based check which was defeated by
            ongoing pair discovery diluting the median.
        completed_positions: number of fully closed position trajectories
            available in position_history.jsonl.
        min_pairs: minimum number of pairs at threshold to enable training.

    Returns:
        Dict mapping model tier name -> bool (True = retrain eligible).
    """
    t1_ready = pairs_at_threshold.get("tier1", 0) >= min_pairs
    t2_ready = pairs_at_threshold.get("tier2", 0) >= min_pairs
    return {
        "linear_regression": t1_ready,
        "xgboost": t1_ready,
        "gru": t2_ready,
        "lstm": t2_ready,
        "ppo_raw": completed_positions >= MIN_POSITIONS_TIER3,
        "ppo_filtered": completed_positions >= MIN_POSITIONS_TIER3,
    }


def should_run_scaling_experiment(
    current_bars_per_pair: int,
    last_checkpoint_ran: int,
    checkpoints: tuple[int, ...] = SCALING_CHECKPOINTS,
) -> bool:
    """Return True if we've crossed a new scaling checkpoint.

    The scaling experiment is expensive (trains 4+ models), so we only
    run it when ``current_bars_per_pair`` crosses the next boundary above
    ``last_checkpoint_ran``.

    Example:
        last run at 50 → we run again the moment we hit 100
        last run at 100 → we run again the moment we hit 250
        last run at 100, currently at 130 → skip (not yet at 250)
    """
    # Find the largest checkpoint we've crossed
    eligible = [c for c in checkpoints if c <= current_bars_per_pair]
    if not eligible:
        return False
    highest_crossed = max(eligible)
    return highest_crossed > last_checkpoint_ran


@dataclass
class DataScalingCheckpoint:
    """One row in the scaling-curve experiment log.

    Each entry represents training every model tier on a specific slice
    of the data and recording the resulting metrics. The log lives in
    ``experiments/results/data_scaling/log.jsonl`` as append-only JSONL.
    """

    bars_per_pair: int
    training_rows: int
    timestamp: str
    metrics_by_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DataScalingCheckpoint":
        return cls(
            bars_per_pair=int(d["bars_per_pair"]),
            training_rows=int(d["training_rows"]),
            timestamp=str(d["timestamp"]),
            metrics_by_model=dict(d.get("metrics_by_model", {})),
        )
