"""Auto-retrain pipeline: combine original + live data, retrain models, track deltas.

Combines the original train.parquet with growing live bars.parquet, retrains
all 8 models (or a subset), evaluates on the fixed test.parquet, and logs
before/after metric deltas to data/live/retrain_log.json.

Usage:
    python -m src.live.retrain                  # full retrain (all 8 models)
    python -m src.live.retrain --skip-tier3     # skip PPO models (faster)
    python -m src.live.retrain --original-only  # baseline eval only (no combined)
    python -m src.live.retrain --help           # show all options

Tip: --skip-tier3 is recommended for development; PPO models take ~10 min each.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_baselines import (
    NON_FEATURE_COLUMNS,
    TARGET_COLUMN,
    _build_split,
    _feature_columns,
    build_models,
    prepare_xy,
    prepare_xy_for_seq,
)
from src.models.autoencoder import AnomalyDetectorAutoencoder
from src.models.ppo_filtered import PPOFilteredPredictor
from src.models.ppo_raw import PPORawPredictor

logger = logging.getLogger(__name__)


class RetrainPipeline:
    """Combine original + live data, retrain models, compare metrics.

    The pipeline:
      1. Loads original train.parquet + live bars.parquet (if exists)
      2. Evaluates all models on original data only ("before")
      3. Evaluates all models on combined data ("after")
      4. Computes per-model metric deltas
      5. Logs the retrain event to retrain_log.json
    """

    def __init__(
        self,
        data_dir: Path = Path("data/processed"),
        live_dir: Path = Path("data/live"),
        threshold: float = 0.02,
    ):
        self.data_dir = Path(data_dir)
        self.live_dir = Path(live_dir)
        self.threshold = threshold
        self.log_path = self.live_dir / "retrain_log.json"

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_combined_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load original train + live bars (if any), return (combined_train, test).

        If bars.parquet does not exist or is empty, returns original train only
        with a warning. The test set is always the fixed test.parquet.

        Returns:
            Tuple of (combined_train DataFrame, test DataFrame).
        """
        # Load original training data
        train_path = self.data_dir / "train.parquet"
        test_path = self.data_dir / "test.parquet"

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        original_train = pd.read_parquet(train_path)
        test = pd.read_parquet(test_path)

        n_original = len(original_train)

        # Try to load live bars
        bars_path = self.live_dir / "bars.parquet"
        if bars_path.exists():
            try:
                live_bars = pd.read_parquet(bars_path)
                n_live = len(live_bars)
            except Exception as e:
                logger.warning(f"Failed to read live bars: {e}. Using original only.")
                live_bars = pd.DataFrame()
                n_live = 0
        else:
            logger.warning(
                f"No live bars found at {bars_path}. Using original training data only."
            )
            live_bars = pd.DataFrame()
            n_live = 0

        # Combine
        if n_live > 0:
            combined_train = pd.concat(
                [original_train, live_bars], ignore_index=True
            )
            # Sort by (pair_id, time_idx)
            combined_train = combined_train.sort_values(
                ["pair_id", "time_idx"]
            ).reset_index(drop=True)
            # Deduplicate on (pair_id, timestamp) keeping first
            combined_train = combined_train.drop_duplicates(
                subset=["pair_id", "timestamp"], keep="first"
            ).reset_index(drop=True)
        else:
            combined_train = original_train.copy()

        n_combined = len(combined_train)
        logger.info(
            f"Combined training data: {n_original} original + {n_live} live "
            f"= {n_combined} total"
        )
        print(
            f"Data: {n_original} original + {n_live} live = {n_combined} combined "
            f"(test: {len(test)})"
        )

        # Store counts for later logging
        self._n_original = n_original
        self._n_live = n_live
        self._n_combined = n_combined
        self._n_test = len(test)

        return combined_train, test

    # ------------------------------------------------------------------
    # Model training + evaluation
    # ------------------------------------------------------------------

    def _train_and_evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        skip_tier3: bool = False,
        label: str = "baseline",
    ) -> dict[str, dict]:
        """Train all models on train_df, evaluate on test_df.

        Args:
            train_df: Training data (original or combined).
            test_df: Fixed test data.
            skip_tier3: If True, skip PPO models.
            label: Label for logging ("baseline" or "combined").

        Returns:
            Dict mapping model_name -> {rmse, mae, directional_accuracy}.
        """
        train_split = _build_split(train_df)
        test_split = _build_split(test_df)
        feature_cols = _feature_columns(train_split)

        # Timestamps for panel-aware Sharpe
        test_timestamps = test_split["timestamp"].to_numpy()

        results: dict[str, dict] = {}

        # ---- Tier 1 ----
        X_train, y_train = prepare_xy(train_split, feature_cols)
        X_test, y_test = prepare_xy(test_split, feature_cols)

        for model in build_models(tier="1"):
            print(f"  [{label}] Training {model.name}...")
            model.fit(X_train, y_train)
            metrics = model.evaluate(
                X_test, y_test,
                threshold=self.threshold,
                timestamps=test_timestamps,
            )
            results[model.name] = {
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "directional_accuracy": metrics["directional_accuracy"],
            }

        # ---- Tier 2 ----
        X_train_seq, y_train_seq = prepare_xy_for_seq(train_split, feature_cols)
        X_test_seq, y_test_seq = prepare_xy_for_seq(test_split, feature_cols)

        for model in build_models(tier="2"):
            print(f"  [{label}] Training {model.name}...")
            model.fit(X_train_seq, y_train_seq)
            metrics = model.evaluate(
                X_test_seq, y_test_seq,
                threshold=self.threshold,
                timestamps=test_timestamps,
            )
            results[model.name] = {
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "directional_accuracy": metrics["directional_accuracy"],
            }

        # ---- Tier 3 ----
        if not skip_tier3:
            n_features = len(feature_cols)

            # PPO-Raw
            print(f"  [{label}] Training PPO-Raw...")
            ppo_raw = PPORawPredictor(random_state=42, total_timesteps=100_000)
            ppo_raw.fit(X_train_seq, y_train_seq)
            metrics = ppo_raw.evaluate(
                X_test_seq, y_test_seq,
                threshold=self.threshold,
                timestamps=test_timestamps,
            )
            results[ppo_raw.name] = {
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "directional_accuracy": metrics["directional_accuracy"],
            }

            # PPO-Filtered
            print(f"  [{label}] Training autoencoder + PPO-Filtered...")
            autoencoder = AnomalyDetectorAutoencoder(
                input_dim=n_features, random_state=42
            )
            autoencoder.fit(train_split[feature_cols], feature_cols)

            ppo_filt = PPOFilteredPredictor(
                anomaly_detector=autoencoder,
                random_state=42,
                total_timesteps=100_000,
            )
            ppo_filt.fit(X_train_seq, y_train_seq)
            metrics = ppo_filt.evaluate(
                X_test_seq, y_test_seq,
                threshold=self.threshold,
                timestamps=test_timestamps,
            )
            results[ppo_filt.name] = {
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "directional_accuracy": metrics["directional_accuracy"],
            }

        return results

    def evaluate_baseline(
        self,
        test: pd.DataFrame,
        skip_tier3: bool = False,
    ) -> dict[str, dict]:
        """Train all models on ORIGINAL train.parquet only, evaluate on test.

        This is the "before" measurement.

        Args:
            test: Fixed test DataFrame.
            skip_tier3: If True, skip PPO models.

        Returns:
            Dict mapping model_name -> {rmse, mae, directional_accuracy}.
        """
        original_train = pd.read_parquet(self.data_dir / "train.parquet")
        print("\n--- Baseline evaluation (original data only) ---")
        return self._train_and_evaluate(
            original_train, test, skip_tier3=skip_tier3, label="baseline"
        )

    def train_and_evaluate_combined(
        self,
        combined_train: pd.DataFrame,
        test: pd.DataFrame,
        skip_tier3: bool = False,
    ) -> dict[str, dict]:
        """Train all models on combined (original + live) data, evaluate on test.

        This is the "after" measurement.

        Args:
            combined_train: Original + live training data.
            test: Fixed test DataFrame.
            skip_tier3: If True, skip PPO models.

        Returns:
            Dict mapping model_name -> {rmse, mae, directional_accuracy}.
        """
        print("\n--- Combined evaluation (original + live data) ---")
        return self._train_and_evaluate(
            combined_train, test, skip_tier3=skip_tier3, label="combined"
        )

    # ------------------------------------------------------------------
    # Delta computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_deltas(
        before: dict[str, dict], after: dict[str, dict]
    ) -> dict[str, dict]:
        """Compute per-model metric deltas (after - before).

        For RMSE/MAE: negative delta = improvement (lower is better).
        For directional_accuracy: positive delta = improvement (higher is better).
        Primary improvement criterion: delta_rmse < 0.

        Args:
            before: Baseline metrics per model.
            after: Combined metrics per model.

        Returns:
            Dict mapping model_name -> {before, after, delta, improved}.
        """
        deltas: dict[str, dict] = {}

        for model_name in before:
            if model_name not in after:
                continue

            b = before[model_name]
            a = after[model_name]

            delta = {
                "rmse": a["rmse"] - b["rmse"],
                "mae": a["mae"] - b["mae"],
                "directional_accuracy": (
                    a["directional_accuracy"] - b["directional_accuracy"]
                ),
            }

            # Improved if RMSE decreased (primary metric)
            improved = delta["rmse"] < -1e-8  # small tolerance for float noise

            deltas[model_name] = {
                "before": b,
                "after": a,
                "delta": delta,
                "improved": improved,
            }

        return deltas

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_retrain_event(
        self,
        deltas: dict[str, dict],
        n_original: int,
        n_live: int,
        n_combined: int,
        n_test: int,
    ) -> None:
        """Append a retrain event to retrain_log.json.

        Creates the file if it does not exist. Accumulates events over time
        so retrain history can be reviewed.

        Args:
            deltas: Per-model delta dict from compute_deltas().
            n_original: Number of original training rows.
            n_live: Number of live bar rows.
            n_combined: Number of combined training rows.
            n_test: Number of test rows.
        """
        self.live_dir.mkdir(parents=True, exist_ok=True)

        # Load existing log
        if self.log_path.exists():
            try:
                with open(self.log_path) as f:
                    log_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                log_data = {"retrain_events": []}
        else:
            log_data = {"retrain_events": []}

        # Build event entry
        event = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "original_train_rows": n_original,
            "live_rows": n_live,
            "combined_rows": n_combined,
            "test_rows": n_test,
            "models": {},
        }

        for model_name, d in deltas.items():
            event["models"][model_name] = {
                "before": {k: round(v, 6) for k, v in d["before"].items()},
                "after": {k: round(v, 6) for k, v in d["after"].items()},
                "delta": {k: round(v, 6) for k, v in d["delta"].items()},
                "improved": d["improved"],
            }

        log_data["retrain_events"].append(event)

        with open(self.log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Retrain event logged to {self.log_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    @staticmethod
    def _format_summary_table(
        deltas: dict[str, dict],
        n_original: int,
        n_live: int,
        n_combined: int,
        n_test: int,
    ) -> str:
        """Format a human-readable summary table of retrain results.

        Args:
            deltas: Per-model delta dict.
            n_original: Original training row count.
            n_live: Live bar row count.
            n_combined: Combined row count.
            n_test: Test row count.

        Returns:
            Multi-line formatted string.
        """
        # Model display order
        model_order = [
            "Naive (Spread Closes)",
            "Volume (Higher Volume Correct)",
            "Linear Regression",
            "XGBoost",
            "GRU",
            "LSTM",
            "PPO-Raw",
            "PPO-Filtered",
        ]

        # Sort models: ordered first, then any extras
        ordered = [m for m in model_order if m in deltas]
        extras = [m for m in deltas if m not in model_order]
        all_models = ordered + extras

        lines = [
            "",
            "====== Retrain Results ======",
            "",
            f"Data: {n_original} original + {n_live} live = {n_combined} combined "
            f"(test: {n_test})",
            "",
            f"{'Model':<30} | {'Before RMSE':>11} | {'After RMSE':>10} "
            f"| {'Delta':>8} | Improved?",
            f"{'-'*30}-+-{'-'*11}-+-{'-'*10}-+-{'-'*8}-+-{'-'*9}",
        ]

        n_improved = 0
        n_trainable = 0

        for model_name in all_models:
            d = deltas[model_name]
            b_rmse = d["before"]["rmse"]
            a_rmse = d["after"]["rmse"]
            delta_rmse = d["delta"]["rmse"]
            improved = d["improved"]

            # Naive/Volume are deterministic baselines -- mark as "--"
            is_baseline = model_name in (
                "Naive (Spread Closes)",
                "Volume (Higher Volume Correct)",
            )

            if is_baseline:
                status = "--"
            elif improved:
                status = "YES"
                n_improved += 1
                n_trainable += 1
            else:
                status = "no"
                n_trainable += 1

            lines.append(
                f"{model_name:<30} | {b_rmse:>11.4f} | {a_rmse:>10.4f} "
                f"| {delta_rmse:>+8.4f} | {status}"
            )

        # Key insight
        lines.append("")
        if n_live == 0:
            lines.append(
                "Key insight: No live data yet -- baseline metrics established. "
                "Before == After expected."
            )
        elif n_trainable > 0 and n_improved > 0:
            pct = n_improved / n_trainable * 100
            lines.append(
                f"Key insight: {n_improved}/{n_trainable} trainable models improved "
                f"({pct:.0f}%) with {n_live} additional live bars."
            )
        elif n_trainable > 0:
            lines.append(
                f"Key insight: No trainable models improved with {n_live} live bars. "
                "More data may be needed."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        skip_tier3: bool = False,
        original_only: bool = False,
    ) -> dict[str, dict]:
        """Execute the full retrain pipeline.

        Steps:
          1. Load combined data (original + live)
          2. Evaluate baseline (original only)
          3. Train and evaluate combined (original + live)
          4. Compute deltas
          5. Log retrain event
          6. Print summary table

        Args:
            skip_tier3: Skip PPO models for faster execution.
            original_only: Only evaluate baseline (no combined training).

        Returns:
            Deltas dict (model_name -> {before, after, delta, improved}).
        """
        combined_train, test = self.load_combined_data()

        # Step 1: Baseline evaluation
        t0 = time.time()
        before = self.evaluate_baseline(test, skip_tier3=skip_tier3)
        t_baseline = time.time() - t0

        if original_only:
            # Use before as both before and after (deltas will be zero)
            after = before
            t_combined = 0.0
        else:
            # Step 2: Combined evaluation
            t1 = time.time()
            after = self.train_and_evaluate_combined(
                combined_train, test, skip_tier3=skip_tier3
            )
            t_combined = time.time() - t1

        # Step 3: Compute deltas
        deltas = self.compute_deltas(before, after)

        # Step 4: Log retrain event
        self.log_retrain_event(
            deltas,
            n_original=self._n_original,
            n_live=self._n_live if not original_only else 0,
            n_combined=self._n_combined if not original_only else self._n_original,
            n_test=self._n_test,
        )

        # Step 5: Print summary
        table = self._format_summary_table(
            deltas,
            n_original=self._n_original,
            n_live=self._n_live if not original_only else 0,
            n_combined=self._n_combined if not original_only else self._n_original,
            n_test=self._n_test,
        )
        print(table)

        t_total = t_baseline + t_combined
        print(
            f"\nTiming: baseline={t_baseline:.1f}s, "
            f"combined={t_combined:.1f}s, total={t_total:.1f}s"
        )

        return deltas


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the retrain pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Auto-retrain pipeline: combine original training data with live bars, "
            "retrain all 8 models, and track whether more data improves performance. "
            "Use --skip-tier3 for faster runs (recommended for development)."
        ),
        prog="python -m src.live.retrain",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing train.parquet and test.parquet (default: data/processed)",
    )
    parser.add_argument(
        "--live-dir",
        type=Path,
        default=Path("data/live"),
        help="Directory containing bars.parquet (default: data/live)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Trading threshold for profit simulation (default: 0.02)",
    )
    parser.add_argument(
        "--skip-tier3",
        action="store_true",
        help="Skip PPO models (they take ~10 min each). Recommended for development.",
    )
    parser.add_argument(
        "--original-only",
        action="store_true",
        help="Evaluate baseline only (no combined training). Useful for validation.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipeline = RetrainPipeline(
        data_dir=args.data_dir,
        live_dir=args.live_dir,
        threshold=args.threshold,
    )

    pipeline.run(
        skip_tier3=args.skip_tier3,
        original_only=args.original_only,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
