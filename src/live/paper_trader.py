"""Paper trading engine for live prediction market arbitrage.

Loads all 8 trained models (Tier 1-3), runs inference on live bars,
and logs paper trades to a JSONL file. Models are trained from scratch
on the historical dataset at startup (no saved weights).

Usage:
    python -m src.live.paper_trader              # train + infer once
    python -m src.live.paper_trader --skip-tier3 # skip PPO models (faster)
    python -m src.live.paper_trader --collect-and-trade  # collect then infer
    python -m src.live.paper_trader --help
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# NOTE: torch-dependent imports (src.models.gru, src.models.lstm, ppo_raw,
# ppo_filtered, autoencoder, experiments.run_baselines) are LAZY — they only
# load when run_cycle()/train_all_models() is actually called. This lets
# ``append_trades`` and the daily-rotation helpers be imported and unit-tested
# without needing torch installed in the caller's environment.
from src.models.base import BasePredictor  # no torch dependency

logger = logging.getLogger(__name__)

# Models that need group_id + sequence context for prediction.
_SEQUENCE_MODEL_NAMES = {"GRU", "LSTM", "PPO-Raw", "PPO-Filtered"}

# Models in Tier 1 (flat feature matrix, no group_id).
_TIER1_MODEL_NAMES = {
    "Naive (Spread Closes)",
    "Volume (Higher Volume Correct)",
    "Linear Regression",
    "XGBoost",
}


class PaperTrader:
    """Paper trading engine that runs all models on live bars.

    Trains models lazily on first call to run_cycle(). Reads live bars
    from data/live/bars.parquet and logs paper trades to
    data/live/paper_trades.jsonl.
    """

    def __init__(
        self,
        data_dir: Path = Path("data/processed"),
        live_dir: Path = Path("data/live"),
        threshold: float = 0.02,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.live_dir = Path(live_dir)
        self.threshold = threshold

        # Lazy-loaded state
        self.models: list[BasePredictor] | None = None
        self.feature_cols: list[str] | None = None
        self._train_df: pd.DataFrame | None = None

    def _daily_trades_path(self) -> Path:
        """Path to today's per-UTC-day trades file.

        The pre-rotation ``paper_trades.jsonl`` archive (~81MB at the
        time of rotation) stays frozen at the repo root; new writes
        go to ``paper_trades_YYYY-MM-DD.jsonl`` so no single file ever
        crosses GitHub's 100MB hard limit. Dashboard and any reader
        must glob ``paper_trades*.jsonl`` to pick up both.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.live_dir / f"paper_trades_{today}.jsonl"

    @property
    def trades_path(self) -> Path:
        """Back-compat alias for the current day's trades file."""
        return self._daily_trades_path()

    def train_all_models(
        self, skip_tier3: bool = False
    ) -> list[BasePredictor]:
        """Train all 8 models (or 6 if skip_tier3) on historical data.

        Follows the exact same instantiation and training pattern as
        experiments/run_baselines.py to ensure consistency.

        Args:
            skip_tier3: If True, skip PPO-Raw and PPO-Filtered (faster).

        Returns:
            List of trained BasePredictor instances.
        """
        # Lazy imports — torch only required when actually training.
        from experiments.run_baselines import (
            _build_split,
            _feature_columns,
            build_models,
            load_train_test,
            prepare_xy,
            prepare_xy_for_seq,
        )
        from src.models.autoencoder import AnomalyDetectorAutoencoder
        from src.models.ppo_filtered import PPOFilteredPredictor
        from src.models.ppo_raw import PPORawPredictor

        logger.info("Loading training data from %s", self.data_dir)
        train_raw, _test_raw = load_train_test(self.data_dir)
        train = _build_split(train_raw)

        feature_cols = _feature_columns(train)
        self.feature_cols = feature_cols
        self._train_df = train

        n_features = len(feature_cols)
        logger.info(
            "Training data: %d rows, %d features", len(train), n_features
        )

        all_models: list[BasePredictor] = []

        # --- Tier 1: flat feature matrix ---
        logger.info("Training Tier 1 models...")
        t1_start = time.time()
        models_t1 = build_models(tier="1")
        X_train, y_train = prepare_xy(train, feature_cols)
        for model in models_t1:
            model.fit(X_train, y_train)
            logger.info("  Trained: %s", model.name)
        all_models.extend(models_t1)
        logger.info("Tier 1 done in %.1fs", time.time() - t1_start)

        # --- Tier 2: sequence models (need group_id) ---
        logger.info("Training Tier 2 models...")
        t2_start = time.time()
        models_t2 = build_models(tier="2")
        X_train_seq, y_train_seq = prepare_xy_for_seq(train, feature_cols)
        for model in models_t2:
            model.fit(X_train_seq, y_train_seq)
            logger.info("  Trained: %s", model.name)
        all_models.extend(models_t2)
        logger.info("Tier 2 done in %.1fs", time.time() - t2_start)

        # --- Tier 3: RL models (need group_id + autoencoder for filtered) ---
        if not skip_tier3:
            logger.info("Training Tier 3 models...")
            t3_start = time.time()

            # PPO-Raw
            ppo_raw = PPORawPredictor(random_state=42, total_timesteps=100_000)
            ppo_raw.fit(X_train_seq, y_train_seq)
            logger.info("  Trained: %s", ppo_raw.name)
            all_models.append(ppo_raw)

            # PPO-Filtered: needs a pre-trained autoencoder
            autoencoder = AnomalyDetectorAutoencoder(
                input_dim=n_features, random_state=42
            )
            autoencoder.fit(train[feature_cols], feature_cols)
            ppo_filt = PPOFilteredPredictor(
                anomaly_detector=autoencoder,
                random_state=42,
                total_timesteps=100_000,
            )
            ppo_filt.fit(X_train_seq, y_train_seq)
            logger.info("  Trained: %s", ppo_filt.name)
            all_models.append(ppo_filt)

            logger.info("Tier 3 done in %.1fs", time.time() - t3_start)
        else:
            logger.info("Skipping Tier 3 (PPO models) per --skip-tier3")

        self.models = all_models
        logger.info("All %d models trained", len(all_models))
        return all_models

    def load_latest_bars(self) -> pd.DataFrame:
        """Load the most recent bar for each pair from live bars.parquet.

        Returns:
            DataFrame with one row per pair_id (the latest bar).

        Raises:
            FileNotFoundError: If bars.parquet does not exist. Run the
                collector first: ``python -m src.live.collector --demo``
        """
        bars_path = self.live_dir / "bars.parquet"
        if not bars_path.exists():
            raise FileNotFoundError(
                f"Live bars not found at {bars_path}. "
                "Run the collector first: python -m src.live.collector --demo"
            )

        df = pd.read_parquet(bars_path)
        if df.empty:
            raise ValueError("bars.parquet exists but is empty.")

        # For each pair_id, take the row with the highest time_idx
        idx = df.groupby("pair_id")["time_idx"].idxmax()
        latest = df.loc[idx].reset_index(drop=True)

        logger.info(
            "Loaded %d latest bars for %d pairs",
            len(latest),
            latest["pair_id"].nunique(),
        )
        return latest

    def _load_all_live_bars(self) -> pd.DataFrame:
        """Load ALL live bars (not just latest) for sequence model context.

        Sequence models (Tier 2/3) need historical context for their
        lookback windows. Returns the full live bars DataFrame.
        """
        bars_path = self.live_dir / "bars.parquet"
        if not bars_path.exists():
            raise FileNotFoundError(
                f"Live bars not found at {bars_path}. "
                "Run the collector first: python -m src.live.collector --demo"
            )
        return pd.read_parquet(bars_path)

    def run_inference(self) -> list[dict]:
        """Run all trained models on the latest live bars.

        For Tier 1 models, uses only the latest bar per pair.
        For Tier 2/3 models, uses full bar history for sequence context.

        Returns:
            List of trade log dicts (one per model per pair).
        """
        if self.models is None or self.feature_cols is None:
            raise RuntimeError(
                "Models not trained. Call train_all_models() first."
            )

        latest_bars = self.load_latest_bars()
        all_live_bars = self._load_all_live_bars()

        # Fill NaN with 0 (same as _build_split does for training)
        latest_bars = latest_bars.fillna(0.0)
        all_live_bars = all_live_bars.fillna(0.0)

        feature_cols = self.feature_cols
        collection_time = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        all_trades: list[dict] = []

        for model in self.models:
            model_name = model.name

            try:
                if model_name in _TIER1_MODEL_NAMES:
                    # Tier 1: flat feature matrix on latest bars only
                    X = latest_bars[feature_cols].copy()
                    predictions = model.predict(X)
                    pair_ids = latest_bars["pair_id"].values
                    bar_data = latest_bars
                else:
                    # Tier 2/3: sequence models need full history + group_id
                    X_seq = all_live_bars[
                        feature_cols + ["group_id"]
                    ].copy()

                    # Warn if too few bars for lookback
                    n_bars = len(all_live_bars)
                    if n_bars < 6:
                        logger.warning(
                            "%s: only %d live bars available "
                            "(lookback=6). Models will pad.",
                            model_name,
                            n_bars,
                        )

                    predictions = model.predict(X_seq)

                    # The prediction for the LAST bar of each pair is current
                    # Get indices of last bar per pair in the full DataFrame
                    last_idx = all_live_bars.groupby("pair_id")[
                        "time_idx"
                    ].idxmax()
                    last_positions = [
                        all_live_bars.index.get_loc(i) for i in last_idx
                    ]
                    predictions = predictions[last_positions]
                    pair_ids = all_live_bars.loc[last_idx, "pair_id"].values
                    bar_data = all_live_bars.loc[last_idx].reset_index(
                        drop=True
                    )

            except Exception as e:
                logger.error(
                    "Inference failed for %s: %s", model_name, e
                )
                continue

            # Build trade log entries
            for i, pred in enumerate(predictions):
                pred_f = float(pred)
                row = bar_data.iloc[i] if i < len(bar_data) else None
                if row is None:
                    continue

                trade_entry = {
                    "timestamp": int(row.get("timestamp", 0)),
                    "collection_time": collection_time,
                    "pair_id": str(pair_ids[i]),
                    "model": model_name,
                    "prediction": round(pred_f, 6),
                    "direction": (
                        "long_spread" if pred_f > 0 else "short_spread"
                    ),
                    "threshold": self.threshold,
                    "kalshi_price": float(row.get("kalshi_close", 0.0)),
                    "polymarket_price": float(
                        row.get("polymarket_close", 0.0)
                    ),
                    "spread": float(row.get("spread", 0.0)),
                    "trade": bool(abs(pred_f) > self.threshold),
                }
                all_trades.append(trade_entry)

        logger.info(
            "Generated %d trade entries across %d models",
            len(all_trades),
            len(self.models),
        )
        return all_trades

    def append_trades(self, trades: list[dict]) -> None:
        """Append trade log entries to the current day's trades file.

        Writes to ``paper_trades_YYYY-MM-DD.jsonl`` (UTC date). Never
        touches the legacy ``paper_trades.jsonl`` archive — that file
        stays frozen. Creates the day's file on first write.

        Args:
            trades: List of trade dicts to log.
        """
        self.live_dir.mkdir(parents=True, exist_ok=True)

        daily_path = self._daily_trades_path()
        with open(daily_path, "a") as f:
            for trade in trades:
                f.write(json.dumps(trade) + "\n")

        logger.info(
            "Appended %d trades to %s", len(trades), daily_path
        )

    def run_cycle(self, skip_tier3: bool = False) -> int:
        """Run one full paper trading cycle.

        Trains models if not already trained, runs inference on the
        latest live bars, and appends trades to the log.

        Args:
            skip_tier3: If True, skip PPO models during training.

        Returns:
            Number of trades logged.
        """
        # Lazy model training
        if self.models is None:
            t_train = time.time()
            self.train_all_models(skip_tier3=skip_tier3)
            train_dur = time.time() - t_train
            print(f"Model training took {train_dur:.1f}s")

        # Run inference
        t_infer = time.time()
        trades = self.run_inference()
        infer_dur = time.time() - t_infer
        print(f"Inference took {infer_dur:.1f}s")

        # Log trades
        self.append_trades(trades)

        n_models = len(self.models) if self.models else 0
        n_trades = sum(1 for t in trades if t["trade"])
        n_skips = sum(1 for t in trades if not t["trade"])
        n_pairs = len(set(t["pair_id"] for t in trades)) if trades else 0

        print(
            f"{len(trades)} entries logged across {n_models} models "
            f"for {n_pairs} pairs ({n_trades} trades, {n_skips} skips)"
        )
        return len(trades)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the paper trader."""
    parser = argparse.ArgumentParser(
        description="Paper trading engine for prediction market arbitrage",
        prog="python -m src.live.paper_trader",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Use adaptive multi-bar position management (Phase 7.3)",
    )
    parser.add_argument(
        "--collect-and-trade",
        action="store_true",
        help="Run collector first (demo mode), then trade",
    )
    parser.add_argument(
        "--skip-tier3",
        action="store_true",
        help="Skip Tier 3 PPO models (faster for testing)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Trade threshold (default: 0.02)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with train.parquet (default: data/processed)",
    )
    parser.add_argument(
        "--live-dir",
        type=str,
        default="data/live",
        help="Live data directory (default: data/live)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Adaptive mode: delegate to TradingStrategy (Phase 7.3)
    if args.adaptive:
        from src.live.strategy import TradingStrategy

        print("=== Adaptive Trading (Phase 7.3) ===")
        strategy = TradingStrategy(live_dir=Path(args.live_dir))
        summary = strategy.run_cycle()
        print(
            f"\nAdaptive cycle: {summary['entries']} entries, "
            f"{summary['exits']} exits, "
            f"{summary['open_positions']} open positions"
        )
        return 0

    # Optionally collect first
    if args.collect_and_trade:
        from src.live.collector import LiveCollector

        print("=== Collection Phase ===")
        collector = LiveCollector(live_dir=Path(args.live_dir))
        n = collector.collect_demo(n_pairs=5)
        print(f"Collected {n} demo bars\n")

    # Run paper trading
    print("=== Paper Trading Phase ===")
    trader = PaperTrader(
        data_dir=Path(args.data_dir),
        live_dir=Path(args.live_dir),
        threshold=args.threshold,
    )
    n_logged = trader.run_cycle(skip_tier3=args.skip_tier3)

    if n_logged == 0:
        print("No trades logged. Ensure live bars exist.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
