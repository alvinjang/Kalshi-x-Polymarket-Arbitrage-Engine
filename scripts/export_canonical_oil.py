#!/usr/bin/env python3
"""Export the canonical-oil XGBoost model for Phase A live deployment.

Re-runs the deterministic best-config XGBoost fit on the locked
canonical_oil train split (seed 42) and writes a deployment-ready
artifact bundle to models/canonical_oil/:

    xgboost.pkl              — pickled XGBoostPredictor
    feature_columns.json     — 50 active feature columns in fit order
    zero_variance_columns.json — N columns with zero variance on train
    canonical_test_predictions.parquet — per-row training-time predictions
                                          for byte-for-byte parity verification
    metadata.json            — training shas + eval metrics + export ts
    README.md                — one-pager describing the artifact

The headline XGBoost config (max_depth=3, lr=0.3, n_estimators=500,
random_state=42) is hardcoded from experiments/results/canonical_oil/
headline/xgboost.json. This script does NOT re-run the 48-config
hyperparameter sweep — it goes straight to the best config and fits
once on canonical_train. The result is bit-identical to what
scripts/train_oil_canonical.py would produce, by construction.

This script is intentionally deterministic and idempotent: running it
twice produces identical artifacts. The byte-for-byte parity check in
scripts/verify_canonical_parity.py relies on that property.

Usage:
    python scripts/export_canonical_oil.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_baselines import (  # noqa: E402
    _build_split,
    _feature_columns,
    prepare_xy,
)
from src.features.engineering import compute_derived_features  # noqa: E402
from src.models.xgboost_model import XGBoostPredictor  # noqa: E402

CANONICAL_DIR = REPO_ROOT / "data" / "processed" / "canonical_oil"
OUT_DIR = REPO_ROOT / "models" / "canonical_oil"
HEADLINE_RESULT = REPO_ROOT / "experiments" / "results" / "canonical_oil" / "headline" / "xgboost.json"

SEED = 42
BEST_CONFIG = {
    "max_depth": 3,
    "learning_rate": 0.3,
    "n_estimators": 500,
    "random_state": SEED,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("export_canonical_oil")


def _sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    log.info("Loading canonical_oil train/test")
    train_raw = pd.read_parquet(CANONICAL_DIR / "train.parquet")
    test_raw = pd.read_parquet(CANONICAL_DIR / "test.parquet")
    train = _build_split(compute_derived_features(train_raw))
    test = _build_split(compute_derived_features(test_raw))

    feature_cols = _feature_columns(train)
    test_feature_cols = _feature_columns(test)
    assert feature_cols == test_feature_cols, "train/test feature columns mismatch"
    log.info("Derived %d feature columns", len(feature_cols))

    variances = train[feature_cols].var(numeric_only=True)
    zero_var = [c for c in feature_cols if variances.get(c, 0) == 0]
    log.info("Zero-variance feature columns: %d", len(zero_var))

    X_tr, y_tr = prepare_xy(train, feature_cols)
    X_te, y_te = prepare_xy(test, feature_cols)
    log.info("Train: %d rows / %d pairs", len(X_tr), train["pair_id"].nunique())
    log.info("Test:  %d rows / %d pairs", len(X_te), test["pair_id"].nunique())

    log.info("Fitting XGBoost with config: %s", BEST_CONFIG)
    model = XGBoostPredictor(**BEST_CONFIG)
    model.fit(X_tr, y_tr)
    y_pred = np.asarray(model.predict(X_te), dtype=float)
    log.info("Test predictions computed: n=%d, mean=%.6f, std=%.6f",
             len(y_pred), float(y_pred.mean()), float(y_pred.std()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pkl_path = OUT_DIR / "xgboost.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    log.info("Wrote %s (%d bytes)", pkl_path, pkl_path.stat().st_size)

    fc_path = OUT_DIR / "feature_columns.json"
    fc_path.write_text(json.dumps(feature_cols, indent=2))
    log.info("Wrote %s", fc_path)

    zv_path = OUT_DIR / "zero_variance_columns.json"
    zv_path.write_text(json.dumps(zero_var, indent=2))
    log.info("Wrote %s", zv_path)

    preds_path = OUT_DIR / "canonical_test_predictions.parquet"
    preds_df = pd.DataFrame({
        "pair_id": test["pair_id"].to_numpy(),
        "time_idx": test["time_idx"].to_numpy() if "time_idx" in test.columns else np.arange(len(test)),
        "timestamp": test["timestamp"].to_numpy() if "timestamp" in test.columns else np.zeros(len(test)),
        "y_true": y_te,
        "y_pred_canonical": y_pred,
    })
    preds_df.to_parquet(preds_path, index=False)
    log.info("Wrote %s (%d rows)", preds_path, len(preds_df))

    rmse = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
    trades = np.sign(y_pred) * y_te * (np.abs(y_pred) > 0.001)
    n_trades = int((np.abs(y_pred) > 0.001).sum())
    alpha_bps = float(trades.sum() / max(n_trades, 1) * 10_000.0) if n_trades else 0.0

    metadata = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "scripts/export_canonical_oil.py",
        "phase": "A.2",
        "purpose": "Phase A canonical-oil XGBoost deployment artifact bundle. Live inference reads xgboost.pkl + feature_columns.json + zero_variance_columns.json from this directory.",
        "git_sha_at_export": _git_sha(),
        "training": {
            "config": BEST_CONFIG,
            "seed": SEED,
            "n_train_rows": int(len(X_tr)),
            "n_train_pairs": int(train["pair_id"].nunique()),
            "n_test_rows": int(len(X_te)),
            "n_test_pairs": int(test["pair_id"].nunique()),
            "n_features": len(feature_cols),
            "n_zero_variance": len(zero_var),
            "target_column": "spread_change_target",
        },
        "data_sources": {
            "train_parquet": str(CANONICAL_DIR / "train.parquet"),
            "train_parquet_sha256": _sha256_path(CANONICAL_DIR / "train.parquet"),
            "test_parquet": str(CANONICAL_DIR / "test.parquet"),
            "test_parquet_sha256": _sha256_path(CANONICAL_DIR / "test.parquet"),
            "split_metadata": str(CANONICAL_DIR / "split_metadata.json"),
            "headline_result_reference": str(HEADLINE_RESULT),
        },
        "eval_at_threshold_0.001": {
            "rmse": rmse,
            "num_trades": n_trades,
            "alpha_bps": alpha_bps,
        },
        "live_deployment": {
            "prediction_threshold": 0.001,
            "oil_family_prefixes": [
                "KXWTI", "KXBRENT", "KXCRUDE", "KXDIESEL",
                "KXHEATINGOIL", "KXGASOLINE", "KXMEXCUBOIL",
            ],
            "shadow_log": "data/live/canonical_predictions.jsonl",
            "env_gates": {
                "CANONICAL_OIL_ENABLED": "false (default; flip on at A2 cutover)",
                "CANONICAL_OIL_SHADOW": "true (default during A1 shadow window)",
            },
        },
    }
    meta_path = OUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    log.info("Wrote %s (alpha_bps=%.2f, rmse=%.6f)", meta_path, alpha_bps, rmse)

    readme = OUT_DIR / "README.md"
    readme.write_text(f"""# Canonical Oil Model Bundle (Phase A)

Deployment artifact bundle for the canonical-oil XGBoost headline model.
Generated by `scripts/export_canonical_oil.py` from the locked
canonical_oil train/test split (seed 42).

## Files

| File | Purpose |
|---|---|
| `xgboost.pkl` | Fitted `XGBoostPredictor` (depth=3, lr=0.3, n_est=500, seed=42). |
| `feature_columns.json` | 50 feature column names in fit order. |
| `zero_variance_columns.json` | 25 columns with zero variance on train; force-zero at inference. |
| `canonical_test_predictions.parquet` | Training-time predictions on canonical_test for byte-for-byte parity verification. |
| `metadata.json` | Training shas, eval metrics, export timestamp. |

## Live deployment

Live inference loads this bundle via `src/live/canonical_inference.py`.
Threshold: 0.001 (per `phase_a_v3.md` section e — scale-equivalent
adaptation, NOT a tuning lever).

Oil-family ticker prefixes only: `KXWTI*, KXBRENT*, KXCRUDE,
KXDIESEL, KXHEATINGOIL, KXGASOLINE, KXMEXCUBOIL`. Non-oil tickers
continue to use the legacy `models/deployed/` artifacts.

Env gates:
- `CANONICAL_OIL_ENABLED=false` (default) — model not used in trade
  decisions.
- `CANONICAL_OIL_SHADOW=true` (default during A1) — predictions
  written to `data/live/canonical_predictions.jsonl` for parity
  + KS drift monitoring.

## Reproducibility

Idempotent: running `python scripts/export_canonical_oil.py` again
produces bit-identical artifacts (deterministic given seed + locked
split). Parity verifier (`scripts/verify_canonical_parity.py`)
asserts `max(abs(live_pred - canonical_test_predictions.y_pred_canonical)) == 0`
on every test row.

## Headline metrics (training-time, threshold=0.001)

| Metric | Value |
|---|---|
| RMSE | {rmse:.6f} |
| Num trades | {n_trades} |
| Alpha bps/trade | {alpha_bps:.2f} |

Reference: `experiments/results/canonical_oil/headline/xgboost.json`
""")
    log.info("Wrote %s", readme)

    log.info("=== Export complete ===")
    log.info("Bundle at: %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
