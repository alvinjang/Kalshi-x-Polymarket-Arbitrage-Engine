"""Phase A canonical-oil XGBoost inference + shadow logging.

Module-level singleton (lazy-loaded on first use):

    canonical_model    — fitted XGBoostPredictor from models/canonical_oil/xgboost.pkl
    canonical_columns  — 50-feature column order (matches training)
    canonical_zero_var — 25 column names that had zero variance on train

Public helpers:

    is_enabled()       — True if CANONICAL_OIL_ENABLED=true (A2 / narrow-replace mode)
    is_shadow()        — True if CANONICAL_OIL_SHADOW=true (A1 / shadow mode)
    use_canonical(t)   — True if Kalshi ticker is in oil family
    predict(row_df)    — score a single-row DataFrame, returning float prediction
    log_shadow(...)    — append a JSONL record to canonical_predictions.jsonl

Design constraints from phase_a_v3.md:
    - Live read path is data/live/bars.parquet (the canonical symlink),
      never hardcode /projectnb.
    - Force-zero columns from zero_variance_columns.json at inference
      so live data distribution shifts in those columns can't perturb
      the model away from its training-time effective view.
    - Threshold 0.001 (scale-equivalent adaptation, not a tuning lever).
    - Oil-family only: KXWTI*, KXBRENT*, KXCRUDE, KXDIESEL,
      KXHEATINGOIL, KXGASOLINE, KXMEXCUBOIL.
    - Independent of LIVE_TRADING — Phase A is paper-only in all modes.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = REPO_ROOT / "models" / "canonical_oil"
LIVE_BARS_PATH = REPO_ROOT / "data" / "live" / "bars.parquet"
SHADOW_LOG_PATH = REPO_ROOT / "data" / "live" / "canonical_predictions.jsonl"
REALIZED_LOG_PATH = REPO_ROOT / "data" / "live" / "canonical_realized.jsonl"

# Position size used to translate raw spread changes into per-trade
# dollar P&L. Matches the canonical training script's $100 convention
# (scripts/train_oil_canonical.py per_trade_outcomes() and the original
# paper's Table 1 reporting).
POSITION_SIZE_USD = 100.0

OIL_FAMILY_PREFIXES = (
    "KXWTI",
    "KXBRENT",
    "KXCRUDE",
    "KXDIESEL",
    "KXHEATINGOIL",
    "KXGASOLINE",
    "KXMEXCUBOIL",
)

PREDICTION_THRESHOLD = 0.001

# Module-level lazy state.
_lock = threading.Lock()
_loaded = False
_model = None
_columns: list[str] = []
_zero_var: frozenset[str] = frozenset()


def _read_bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def is_enabled() -> bool:
    """A2 narrow-replacement mode (canonical drives oil trade decisions).

    Defaults to False. Flip via CANONICAL_OIL_ENABLED=true at cutover.
    """
    return _read_bool_env("CANONICAL_OIL_ENABLED", False)


def is_shadow() -> bool:
    """A1 shadow mode (canonical scored + logged but not acted on).

    Defaults to True so the model produces parity data the moment the
    bundle is deployed. Turn off only after A2 stabilizes.
    """
    return _read_bool_env("CANONICAL_OIL_SHADOW", True)


def use_canonical(kalshi_ticker: str) -> bool:
    """Pair admission filter — canonical model serves the oil family only."""
    if not kalshi_ticker:
        return False
    return kalshi_ticker.startswith(OIL_FAMILY_PREFIXES)


def _ensure_loaded() -> None:
    global _loaded, _model, _columns, _zero_var
    if _loaded:
        return
    with _lock:
        if _loaded:
            return
        pkl_path = MODEL_DIR / "xgboost.pkl"
        cols_path = MODEL_DIR / "feature_columns.json"
        zv_path = MODEL_DIR / "zero_variance_columns.json"
        if not pkl_path.exists() or not cols_path.exists() or not zv_path.exists():
            raise FileNotFoundError(
                f"Canonical bundle incomplete in {MODEL_DIR}. Run scripts/export_canonical_oil.py."
            )
        with open(pkl_path, "rb") as f:
            _model = pickle.load(f)
        _columns = json.loads(cols_path.read_text())
        _zero_var = frozenset(json.loads(zv_path.read_text()))
        _loaded = True
        logger.info(
            "Canonical inference loaded: %d features, %d zero-variance, threshold=%.4f",
            len(_columns), len(_zero_var), PREDICTION_THRESHOLD,
        )


def build_row(features_df: pd.DataFrame) -> pd.DataFrame:
    """Project a feature DataFrame onto the canonical column schema.

    Accepts a single-row DataFrame produced by the existing strategy
    pipeline (compute_derived_features + fillna(0)). Returns a
    single-row DataFrame with EXACTLY the canonical 50 columns in fit
    order, with zero-variance columns force-zeroed regardless of
    incoming values.

    Missing canonical columns are filled with 0.0 (matching training-
    time fillna behavior on canonical_train).
    """
    _ensure_loaded()
    if len(features_df) != 1:
        raise ValueError(f"build_row expects 1 row, got {len(features_df)}")
    out_cols = {}
    for col in _columns:
        if col in _zero_var:
            out_cols[col] = 0.0
        elif col in features_df.columns:
            val = features_df[col].iloc[0]
            out_cols[col] = 0.0 if pd.isna(val) else float(val)
        else:
            out_cols[col] = 0.0
    return pd.DataFrame([out_cols], columns=_columns)


def predict(features_df: pd.DataFrame) -> float:
    """Score a single-row feature DataFrame, returning the canonical prediction."""
    _ensure_loaded()
    row = build_row(features_df)
    y = _model.predict(row)
    return float(np.asarray(y).reshape(-1)[0])


def would_trade(pred: float) -> bool:
    """Trade-rule: abs(pred) > threshold."""
    return abs(pred) > PREDICTION_THRESHOLD


def log_shadow(
    *,
    ts: int,
    pair_id: str,
    kalshi_ticker: str,
    canonical_pred: float,
    spread: float,
    legacy_avg_pred: Optional[float] = None,
    legacy_lr_pred: Optional[float] = None,
    legacy_xgb_pred: Optional[float] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Append a single shadow-prediction record to canonical_predictions.jsonl.

    Append-only, parent-dir auto-created. Designed to never raise — a
    logging failure here must not break the trading cycle.
    """
    try:
        SHADOW_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": int(ts),
            "ts_iso": datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat(),
            "pair_id": pair_id,
            "kalshi_ticker": kalshi_ticker,
            "canonical_pred": float(canonical_pred),
            "canonical_would_trade": bool(abs(canonical_pred) > PREDICTION_THRESHOLD),
            "canonical_threshold": PREDICTION_THRESHOLD,
            "spread": float(spread),
            "legacy_avg_pred": float(legacy_avg_pred) if legacy_avg_pred is not None else None,
            "legacy_lr_pred": float(legacy_lr_pred) if legacy_lr_pred is not None else None,
            "legacy_xgb_pred": float(legacy_xgb_pred) if legacy_xgb_pred is not None else None,
            "mode": "shadow" if is_shadow() and not is_enabled() else ("dual" if is_shadow() and is_enabled() else "live"),
        }
        if extra:
            rec["extra"] = extra
        with open(SHADOW_LOG_PATH, "a") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except Exception as e:
        logger.warning("canonical shadow log failed: %s", e)


def _load_resolved_keys() -> set[tuple[str, int]]:
    """Return the set of (pair_id, ts) pairs already present in the
    realized log so we never double-write a resolved prediction."""
    out: set[tuple[str, int]] = set()
    if not REALIZED_LOG_PATH.exists():
        return out
    try:
        with open(REALIZED_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    out.add((r["pair_id"], int(r["ts"])))
                except Exception:
                    continue
    except Exception as e:
        logger.warning("could not read %s: %s", REALIZED_LOG_PATH, e)
    return out


def _load_pending_shadow() -> list[dict]:
    """Read the shadow log and return records not yet in the realized log."""
    if not SHADOW_LOG_PATH.exists():
        return []
    resolved = _load_resolved_keys()
    pending: list[dict] = []
    try:
        with open(SHADOW_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                    key = (p["pair_id"], int(p["ts"]))
                    if key not in resolved:
                        pending.append(p)
                except Exception:
                    continue
    except Exception as e:
        logger.warning("could not read %s: %s", SHADOW_LOG_PATH, e)
    return pending


def resolve_pending_shadow() -> int:
    """Resolve every shadow prediction whose next-bar spread is now
    visible in bars.parquet.

    For each unresolved prediction we:
      - find the first bar for that pair_id with timestamp > prediction.ts
      - compute realized_spread_change = next_bar.spread - prediction.spread
        (matches the canonical training target ``spread_change_target``)
      - if would_trade==True, compute trade P&L =
          sign(canonical_pred) * realized_spread_change * POSITION_SIZE_USD
        (matches scripts/train_oil_canonical.py per_trade_outcomes())
      - append an enriched record to canonical_realized.jsonl

    Returns the number of newly-resolved records (0 if none ready,
    or if bars.parquet / shadow log missing). Fail-safe — never raises.
    """
    try:
        pending = _load_pending_shadow()
        if not pending:
            return 0
        if not LIVE_BARS_PATH.exists():
            return 0

        import numpy as np
        import pyarrow.parquet as pq

        # Read only what we need from bars.parquet (pair_id / ts / spread).
        # We can't easily filter to just the pending pair_ids without a
        # full pass, but the file is small enough (~half-million rows)
        # that a single read is fine.
        tbl = pq.read_table(
            LIVE_BARS_PATH, columns=["pair_id", "timestamp", "spread"]
        )
        bars = tbl.to_pandas()
        if bars.empty:
            return 0

        # Group bars by pair_id for fast lookup.
        bars_by_pair: dict[str, pd.DataFrame] = {}
        for pid, g in bars.groupby("pair_id"):
            bars_by_pair[pid] = g.sort_values("timestamp").reset_index(drop=True)

        REALIZED_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        n_new = 0
        with open(REALIZED_LOG_PATH, "a") as out:
            for p in pending:
                pid = p["pair_id"]
                pred_ts = int(p["ts"])
                pair_bars = bars_by_pair.get(pid)
                if pair_bars is None or pair_bars.empty:
                    continue
                next_bars = pair_bars[pair_bars["timestamp"] > pred_ts]
                if next_bars.empty:
                    continue
                next_row = next_bars.iloc[0]
                next_spread = float(next_row["spread"]) if pd.notna(next_row["spread"]) else None
                if next_spread is None:
                    continue
                pred_spread = float(p["spread"])
                realized_change = next_spread - pred_spread

                canonical_pred = float(p["canonical_pred"])
                would_trade = bool(p.get("canonical_would_trade", False))
                if would_trade and canonical_pred != 0.0:
                    direction = 1.0 if canonical_pred > 0 else -1.0
                    pnl_per_dollar = direction * realized_change
                    pnl_usd = pnl_per_dollar * POSITION_SIZE_USD
                    sign_match = (canonical_pred > 0) == (realized_change > 0)
                else:
                    pnl_per_dollar = 0.0
                    pnl_usd = 0.0
                    sign_match = None

                enriched = dict(p)
                enriched["realized_next_ts"] = int(next_row["timestamp"])
                enriched["realized_next_ts_iso"] = datetime.fromtimestamp(
                    int(next_row["timestamp"]), tz=timezone.utc
                ).isoformat()
                enriched["realized_next_spread"] = next_spread
                enriched["realized_spread_change"] = float(realized_change)
                enriched["realized_pnl_per_dollar"] = float(pnl_per_dollar)
                enriched["realized_pnl_usd"] = float(pnl_usd)
                enriched["realized_sign_match"] = sign_match
                enriched["realized_position_size_usd"] = POSITION_SIZE_USD
                enriched["realized_resolved_at_iso"] = datetime.now(
                    timezone.utc
                ).isoformat()

                out.write(json.dumps(enriched, separators=(",", ":")) + "\n")
                n_new += 1

        if n_new > 0:
            logger.info("canonical shadow: resolved %d predictions", n_new)
        return n_new
    except Exception as e:
        logger.warning("resolve_pending_shadow failed: %s", e)
        return 0
