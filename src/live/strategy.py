"""Adaptive trading strategy combining classifier, position manager, and models.

Integrates:
  - ContractClassifier (tier assignment by resolution proximity)
  - PositionManager (SQLite-backed position tracking + 5 exit rules)
  - Pre-trained LR + XGBoost models (loaded from pickle)
  - LiveCollector (price fetching + bar construction)

One call to ``run_cycle()`` executes the full trading loop:
  classify -> collect prices -> update positions -> check exits ->
  check entries -> save state.

Entry filters:
  - price >= $0.10 (max of kalshi, poly price)
  - abs(spread) >= 30pp (0.30)
  - model |prediction| > 0.02
  - tier is not UNKNOWN
  - no existing position on this pair

Usage:
    from src.live.strategy import TradingStrategy
    strategy = TradingStrategy()
    summary = strategy.run_cycle()
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.engineering import compute_derived_features
from src.live.collector import LiveCollector
from src.live.contract_classifier import ContractClassifier
from src.live.order_executor import execute_arb
from src.live.position_manager import ExitReason, PositionManager
from src.live.risk_manager import is_live_trading_armed
from src.models.base import BasePredictor

logger = logging.getLogger(__name__)


class TradingStrategy:
    """Adaptive multi-bar trading strategy for prediction market arbitrage.

    Combines contract classification (time-to-resolution tiers), position
    management (persistent state + exit rules), and model inference
    (pre-trained LR + XGBoost) into a single trading cycle.

    Args:
        live_dir: Directory for live data (positions.db, bars.parquet, etc.).
        model_dir: Directory containing pickle files and feature_columns.json.
        min_price: Minimum price filter (max of kalshi, poly price).
        min_spread: Minimum absolute spread filter (in probability points).
        prediction_threshold: Minimum |prediction| to trigger entry.
    """

    def __init__(
        self,
        live_dir: Path = Path("data/live"),
        model_dir: Path = Path("models/deployed"),
        min_price: float = 0.10,
        min_spread: float = 0.30,
        max_spread: float = 0.85,
        prediction_threshold: float = 0.02,
    ) -> None:
        self.live_dir = Path(live_dir)
        self.model_dir = Path(model_dir)
        self.min_price = min_price
        self.min_spread = min_spread
        # No convergence room above ~85pp — the max possible spread is
        # 1.00 (one side near 0, the other near 1), so a spread of
        # e.g. 0.96 has only 4pp to go before hitting the boundary.
        # Filed as task #28 after live_0237 (KXFEDCOMBO-26APR-0-0) opened
        # at |spread|=0.9645 and realized -$1.93 in a single bar due to
        # a sign-convention bug that couldn't occur if we hadn't entered.
        self.max_spread = max_spread
        self.prediction_threshold = prediction_threshold

        # Classifier
        self._classifier = ContractClassifier()

        # Position Manager
        self._pm = PositionManager(
            db_path=str(self.live_dir / "positions.db"),
            history_jsonl_path=str(self.live_dir / "position_history.jsonl"),
        )

        # Load models
        lr_path = self.model_dir / "linear_regression.pkl"
        xgb_path = self.model_dir / "xgboost.pkl"
        if not lr_path.exists() or not xgb_path.exists():
            raise FileNotFoundError(
                f"Model pickles not found in {self.model_dir}. "
                "Run: python -m src.live.trading_cycle --export-models"
            )
        self._lr_model = BasePredictor.load(lr_path)
        self._xgb_model = BasePredictor.load(xgb_path)
        logger.info(
            "Loaded models: %s, %s",
            self._lr_model.name,
            self._xgb_model.name,
        )

        # Load feature columns
        fc_path = self.model_dir / "feature_columns.json"
        if not fc_path.exists():
            raise FileNotFoundError(
                f"feature_columns.json not found at {fc_path}. "
                "Run: python -m src.live.trading_cycle --export-models"
            )
        with open(fc_path) as f:
            self._feature_columns: list[str] = json.load(f)
        logger.info("Feature columns: %d", len(self._feature_columns))

        # Load active matches (with structural quality filter applied).
        # The filter drops pairs like NBA season-wins vs champion markets,
        # Fed year mismatches, and cross-topic politics — patterns that
        # look semantically close but never converge.
        from src.matching.quality_filter import filter_active_matches

        matches_path = self.live_dir / "active_matches.json"
        if not matches_path.exists():
            logger.warning(
                "No active_matches.json found at %s. "
                "Trading will skip entry checks.",
                matches_path,
            )
            self._matches: list[dict] = []
        else:
            with open(matches_path) as f:
                raw_matches = json.load(f)
            self._matches, filter_stats = filter_active_matches(raw_matches)
            logger.info(
                "Active matches: %d pairs (filtered from %d; rejections=%s)",
                len(self._matches),
                filter_stats["total"],
                filter_stats["reasons"],
            )

        # Collector for price fetching + bar building
        self._collector = LiveCollector(
            live_dir=self.live_dir, use_live_pairs=True
        )

        # Real-money order clients. ONLY constructed when LIVE_TRADING
        # is armed — paper-mode runs should never instantiate live API
        # clients (avoids accidental config / credential loading).
        # Each client lazily defers heavy imports + network until used.
        self._kalshi_client = None
        self._poly_client = None
        if is_live_trading_armed():
            try:
                from src.live.kalshi_orders import KalshiOrderClient
                from src.live.polymarket_orders import PolymarketOrderClient
                self._kalshi_client = KalshiOrderClient()
                self._poly_client = PolymarketOrderClient()
                logger.warning(
                    "LIVE_TRADING ARMED — real-money order placement is ENABLED."
                )
            except Exception as e:
                logger.exception(
                    "LIVE_TRADING armed but failed to construct order clients: %s. "
                    "Falling back to paper-only.", e,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_last_collection_times(self) -> dict[str, int]:
        """Get the most recent collection timestamp per pair from bars.parquet.

        Returns:
            Dict mapping pair_id -> unix timestamp of last bar.
        """
        bars_path = self.live_dir / "bars.parquet"
        if not bars_path.exists():
            return {}
        try:
            df = pd.read_parquet(bars_path, columns=["pair_id", "timestamp"])
            return (
                df.groupby("pair_id")["timestamp"]
                .max()
                .to_dict()
            )
        except Exception as e:
            logger.warning("Failed to read bars.parquet: %s", e)
            return {}

    def _build_feature_vector(
        self,
        kalshi_price: float,
        poly_price: float,
        pair_id: str,
        timestamp: int,
    ) -> pd.DataFrame | None:
        """Build a single-row feature DataFrame from spot prices.

        Uses LiveCollector.build_snapshot_bar() then compute_derived_features().
        Returns None if feature extraction fails.
        """
        try:
            bar = LiveCollector.build_snapshot_bar(
                kalshi_price, poly_price, pair_id, timestamp
            )
            df = pd.DataFrame([bar])

            from src.features.schemas import ALIGNED_COLUMNS
            df = df[ALIGNED_COLUMNS]

            df = compute_derived_features(df)
            df = df.fillna(0.0)

            # Select only the feature columns the models expect
            available = [c for c in self._feature_columns if c in df.columns]
            if len(available) < len(self._feature_columns):
                missing = set(self._feature_columns) - set(available)
                # Add missing columns as zero
                for col in missing:
                    df[col] = 0.0
            features = df[self._feature_columns]
            return features
        except Exception as e:
            logger.warning("Feature build failed for %s: %s", pair_id, e)
            return None

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    def run_cycle(self, dry_run: bool = False) -> dict:
        """Execute one full adaptive trading cycle.

        Steps:
          1. Classify all pairs by resolution proximity
          2. Determine which pairs need price collection (bar interval elapsed)
          3. Collect prices from Kalshi + Polymarket
          4. Update open positions with current spreads
          5. Check exit rules for all open positions
          6. Check entry conditions for new positions
          7. Build and append bars for collected pairs
          8. Return summary

        Args:
            dry_run: If True, compute everything but do not save positions
                or append bars.

        Returns:
            Summary dict with counts and details.
        """
        now = datetime.now(timezone.utc)
        now_naive = now.replace(tzinfo=None)
        now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        ts = int(now.timestamp())

        logger.info("=== Trading Cycle @ %s ===", now_iso)

        # Step 1: Classify all pairs
        classifications = self._classifier.classify_all_pairs(
            self._matches, now=now_naive, use_api=False
        )
        logger.info("Classified %d pairs", len(classifications))

        # Step 2: Determine collection set (bar interval elapsed)
        last_times = self._get_last_collection_times()
        collection_set: set[str] = set()
        for pair_id, info in classifications.items():
            bar_interval = info["bar_interval_seconds"]
            last_ts = last_times.get(pair_id, 0)
            elapsed = ts - last_ts
            if elapsed >= bar_interval:
                collection_set.add(pair_id)

        logger.info(
            "Collection set: %d / %d pairs (bar interval elapsed)",
            len(collection_set),
            len(classifications),
        )

        # Step 3: Collect prices
        kalshi_prices = self._collector.fetch_kalshi_prices()
        poly_prices = self._collector.fetch_polymarket_prices()
        prices_fetched = len(kalshi_prices) + len(poly_prices)
        logger.info(
            "Prices fetched: %d kalshi, %d poly",
            len(kalshi_prices),
            len(poly_prices),
        )

        # Build ticker -> pair_id and poly_id -> pair_id mappings.
        # Pair_ids are content-addressed via make_pair_id so they stay
        # stable across filter/discovery runs — see task #32 fix and
        # memory/project_pair_id_schema_bug.md.
        from src.live.pair_ids import make_pair_id

        ticker_to_pair: dict[str, str] = {}
        poly_to_pair: dict[str, str] = {}
        pair_to_info: dict[str, dict] = {}
        for match in self._matches:
            pair_id = make_pair_id(
                match.get("kalshi_ticker", ""),
                match.get("poly_id", ""),
            )
            if not pair_id:
                continue
            ticker_to_pair[match["kalshi_ticker"]] = pair_id
            poly_to_pair[match["poly_id"]] = pair_id
            pair_to_info[pair_id] = match

        # Step 4: Update open positions
        open_positions = self._pm.get_open_positions()
        updated_count = 0
        for pair_id, pos in open_positions.items():
            # Find current prices for this position
            k_ticker = pos.kalshi_ticker
            k_price = kalshi_prices.get(k_ticker)
            # Find poly_id for this pair
            info = pair_to_info.get(pair_id, {})
            p_id = info.get("poly_id", "")
            p_price = poly_prices.get(p_id)

            if k_price is None or p_price is None:
                logger.debug(
                    "Skipping position update for %s (missing prices)", pair_id
                )
                continue

            current_spread = k_price - p_price
            if not dry_run:
                try:
                    self._pm.update_position(pair_id, current_spread)
                except KeyError:
                    # Position was closed by another runner (GHA) between
                    # when we loaded open_positions and now. Safe to skip.
                    logger.debug("Position %s already closed — skipping", pair_id)
                    continue
            updated_count += 1

        logger.info("Updated %d / %d open positions", updated_count, len(open_positions))

        # Step 5: Check exits
        exit_details: list[dict] = []
        if not dry_run:
            exits = self._pm.check_all_exits(now)
            for pair_id, reason in exits:
                pos = self._pm.get_open_positions().get(pair_id)
                if pos is None:
                    continue
                exit_spread = pos.current_spread
                record = self._pm.close_position(
                    pair_id, reason, exit_spread, now_iso
                )
                logger.info(
                    "EXIT: %s %s @ %.4f P&L=%+.4f held=%d bars",
                    pair_id,
                    reason.name,
                    exit_spread,
                    record["realized_pnl"],
                    record["bars_held"],
                )
                exit_details.append(record)
        else:
            # Dry run: report what would exit
            exits = self._pm.check_all_exits(now)
            for pair_id, reason in exits:
                logger.info(
                    "DRY-RUN EXIT: %s would exit via %s", pair_id, reason.name
                )
                exit_details.append(
                    {"pair_id": pair_id, "reason": reason.name, "dry_run": True}
                )

        logger.info("Exits: %d positions closed", len(exit_details))

        # Step 6: Check entries
        entry_details: list[dict] = []
        for pair_id in collection_set:
            # Skip if already in a position
            if self._pm.has_position(pair_id):
                continue

            # Get classification
            cl = classifications.get(pair_id)
            if cl is None or cl["tier"] == "UNKNOWN":
                continue

            # Get prices
            info = pair_to_info.get(pair_id, {})
            k_ticker = info.get("kalshi_ticker", "")
            p_id = info.get("poly_id", "")

            k_price = kalshi_prices.get(k_ticker)
            p_price = poly_prices.get(p_id)

            if k_price is None or p_price is None:
                continue

            # Price filter
            if max(k_price, p_price) < self.min_price:
                continue

            # Spread filter: reject both "too small to be worth trading"
            # and "too big to have convergence room". An entry at
            # |spread|=0.96 has effectively no upside because the max
            # possible spread is 1.0 — we'd need the disagreement to
            # grow AND then collapse, which is a losing setup.
            spread = k_price - p_price
            spread_abs = abs(spread)
            if spread_abs < self.min_spread:
                continue
            if spread_abs > self.max_spread:
                continue

            # Category-aware entry filter: oil/commodity pairs have a
            # historical +$0.41/trade edge (76.5% WR) while politics
            # and sports are near-zero or negative. We let commodity
            # entries through at the normal threshold but require a
            # higher bar for low-edge categories. This focuses capital
            # on the categories that actually make money.
            #
            # Universe tightened 2026-05-12 after live audit of 5000
            # exits: KXDOGED (crypto) ran n=1820 at 34% WR and net
            # flat — empirically not edge. Inflation similar. Restrict
            # the "commodity" universe to oil + non-oil commodities,
            # which are the only categories the paper actually validated.
            from src.features.category import derive_category_from_ticker
            category = derive_category_from_ticker(k_ticker)
            is_commodity = category in ("oil", "commodities")

            # Build features and predict
            features = self._build_feature_vector(k_price, p_price, pair_id, ts)
            if features is None:
                continue

            try:
                lr_pred = float(self._lr_model.predict(features)[0])
                xgb_pred = float(self._xgb_model.predict(features)[0])
            except Exception as e:
                logger.warning("Prediction failed for %s: %s", pair_id, e)
                continue

            avg_pred = (lr_pred + xgb_pred) / 2.0

            # Phase A canonical-oil shadow hook. Additive only — never
            # affects legacy trading decisions in shadow mode. If the
            # canonical bundle isn't installed or scoring fails, log a
            # warning and continue. See scripts/scc_hardened/phase_a_v3.md
            # and src/live/canonical_inference.py.
            from src.live import canonical_inference
            if (canonical_inference.is_shadow() or canonical_inference.is_enabled()) \
                    and canonical_inference.use_canonical(k_ticker):
                try:
                    canon_pred = canonical_inference.predict(features)
                    canonical_inference.log_shadow(
                        ts=ts,
                        pair_id=pair_id,
                        kalshi_ticker=k_ticker,
                        canonical_pred=canon_pred,
                        spread=spread,
                        legacy_avg_pred=avg_pred,
                        legacy_lr_pred=lr_pred,
                        legacy_xgb_pred=xgb_pred,
                    )
                except FileNotFoundError:
                    pass  # bundle not deployed yet; silent skip
                except Exception as e:
                    logger.warning("canonical shadow failed for %s: %s", pair_id, e)

            # Concordance check: only enter when both models agree on
            # direction. If LR says spread will widen and XGBoost says
            # it will narrow, the average might still cross threshold
            # but the signal is conflicted → skip. This eliminates
            # ambiguous entries where one model dominates the average.
            if np.sign(lr_pred) != np.sign(xgb_pred):
                continue

            # Prediction threshold: commodity pairs use the normal bar,
            # non-commodity pairs need 5x the prediction confidence.
            # This doesn't block non-commodity entirely — a strong
            # enough signal still triggers — but filters out marginal
            # entries that historically lost money.
            #
            # Multiplier raised 3x → 5x on 2026-05-12 after live audit:
            # KXPAYROLLS (n=489) and KXVOTESAVEAMERICA (n=132) were
            # still passing the 3x filter and bleeding capital. 5x
            # forces a stronger conviction for any non-edge category.
            effective_threshold = self.prediction_threshold
            if not is_commodity:
                effective_threshold = self.prediction_threshold * 5.0
            if abs(avg_pred) < effective_threshold:
                continue

            # Determine direction
            if spread > 0:
                direction = "short_spread"  # bet spread narrows
            else:
                direction = "long_spread"

            # Real-money execution layer.
            # When LIVE_TRADING env var is NOT 'true', execute_arb returns
            # a paper-mode no-op so the existing position_manager-only
            # codepath continues unchanged. When LIVE_TRADING IS 'true',
            # it places real orders on both Kalshi + Polymarket and runs
            # the risk gate, returning a permit/reject decision.
            exec_result = None
            if not dry_run:
                # Polymarket token IDs come from pair_mapping; the live
                # entries don't include them today so we attempt a best-
                # effort lookup. If the YES/NO tokens aren't known we
                # cannot place a real order — fall back to paper-only.
                poly_yes_token = info.get("polymarket_yes_token_id", "")
                poly_no_token = info.get("polymarket_no_token_id", "")
                if is_live_trading_armed() and (not poly_yes_token or not poly_no_token):
                    logger.warning(
                        "LIVE_TRADING armed but poly token ids missing for %s; "
                        "skipping entry (would be one-sided exposure).",
                        pair_id,
                    )
                    continue

                exec_result = execute_arb(
                    pair_id=pair_id,
                    kalshi_ticker=k_ticker,
                    poly_yes_token_id=poly_yes_token,
                    poly_no_token_id=poly_no_token,
                    kalshi_yes_price=k_price,
                    poly_yes_price=p_price,
                    spread=spread,
                    kalshi_client=getattr(self, "_kalshi_client", None),
                    poly_client=getattr(self, "_poly_client", None),
                )
                if not exec_result.permitted:
                    logger.info(
                        "ENTRY rejected by risk gate: %s — %s",
                        pair_id, exec_result.reject_reason,
                    )
                    continue
                # Skip writing to position_manager if a real-money entry
                # was attempted but only one leg filled. The order_executor
                # already cancelled the open leg; logging it as a position
                # would falsely mark capital as deployed.
                if is_live_trading_armed() and not (
                    exec_result.placed_kalshi and exec_result.placed_polymarket
                ):
                    continue

            # Open position.
            # Store entry_spread SIGNED (not abs) so it matches the
            # signed `current_spread` that update_position writes at
            # exit time. Storing one as abs and the other as signed
            # produced the live_0237 bug where realized_pnl came out
            # as 2x the spread magnitude. Task #28.
            if not dry_run:
                self._pm.open_position(
                    pair_id=pair_id,
                    kalshi_ticker=k_ticker,
                    direction=direction,
                    entry_spread=spread,
                    kalshi_price=k_price,
                    poly_price=p_price,
                    tier=cl["tier"],
                    bar_interval_seconds=cl["bar_interval_seconds"],
                    resolution_date=cl.get("resolution_date"),
                )

            entry_record = {
                "pair_id": pair_id,
                "direction": direction,
                "spread": round(spread, 4),
                "lr_pred": round(lr_pred, 4),
                "xgb_pred": round(xgb_pred, 4),
                "avg_pred": round(avg_pred, 4),
                "tier": cl["tier"],
                "dry_run": dry_run,
            }
            entry_details.append(entry_record)
            logger.info(
                "ENTRY: %s %s @ %.4f (LR=%.4f, XGB=%.4f)",
                pair_id,
                direction,
                spread,
                lr_pred,
                xgb_pred,
            )

        logger.info("Entries: %d new positions", len(entry_details))

        # Step 7: Build and append bars for collected pairs
        if not dry_run and collection_set:
            bars = []
            for pair_id in collection_set:
                info = pair_to_info.get(pair_id, {})
                k_ticker = info.get("kalshi_ticker", "")
                p_id = info.get("poly_id", "")
                k_price = kalshi_prices.get(k_ticker)
                p_price = poly_prices.get(p_id)
                if k_price is not None and p_price is not None:
                    bar = LiveCollector.build_snapshot_bar(
                        k_price, p_price, pair_id, ts
                    )
                    bars.append(bar)
            if bars:
                df = self._collector.assemble_bar_dataframe(bars)
                self._collector.append_to_parquet(df)
                logger.info("Appended %d bars", len(bars))

        # Phase A.E1 — resolve any shadow predictions whose next-bar
        # spread is now visible in bars.parquet. Runs AFTER bar-append
        # so a prediction from cycle N-1 can resolve against the bar
        # just appended (1-cycle resolution lag, ~15 min). Fail-safe.
        try:
            from src.live import canonical_inference as _ci
            if _ci.is_shadow() or _ci.is_enabled():
                _ci.resolve_pending_shadow()
        except Exception as e:
            logger.warning("resolve_pending_shadow hook failed: %s", e)

        # Step 8: Summary
        final_positions = self._pm.get_open_positions()
        summary = {
            "timestamp": now_iso,
            "entries": len(entry_details),
            "exits": len(exit_details),
            "open_positions": len(final_positions),
            "pairs_checked": len(collection_set),
            "pairs_classified": len(classifications),
            "prices_fetched": prices_fetched,
            "dry_run": dry_run,
            "exit_details": exit_details,
            "entry_details": entry_details,
        }

        logger.info(
            "Cycle complete: %d entries, %d exits, %d open, %d pairs checked",
            summary["entries"],
            summary["exits"],
            summary["open_positions"],
            summary["pairs_checked"],
        )
        return summary
