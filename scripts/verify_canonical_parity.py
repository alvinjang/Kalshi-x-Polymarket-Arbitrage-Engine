#!/usr/bin/env python3
"""Byte-for-byte parity check for the canonical-oil deployment.

Re-runs the live-mode inference helper (src.live.canonical_inference)
on every row of the canonical_test split and asserts that the
resulting predictions exactly match the training-time predictions
saved alongside xgboost.pkl by scripts/export_canonical_oil.py.

Exit code 0 on pass, 1 on fail.

Failure modes this catches:
    - Column order drift (e.g. canonical_inference's column list is
      out of sync with the pickled model's training column order)
    - Zero-variance enforcement gap (a forced-zero column has a
      nonzero value at inference and we're failing to mask it)
    - Numpy / XGBoost version skew between training environment
      and live inference environment
    - Stray transformation in the live feature path that the
      training pipeline didn't apply

On failure, dumps the first 5 mismatched rows with column-by-column
deltas so the root cause is immediately visible.

Phase A blocks on this passing before promotion to A2 (narrow-replace).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_baselines import _build_split, _feature_columns, prepare_xy  # noqa: E402
from src.features.engineering import compute_derived_features  # noqa: E402
from src.live import canonical_inference  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("verify_parity")

CANONICAL_DIR = REPO_ROOT / "data" / "processed" / "canonical_oil"
PREDS_PATH = REPO_ROOT / "models" / "canonical_oil" / "canonical_test_predictions.parquet"


def main() -> int:
    if not PREDS_PATH.exists():
        log.error(
            "Training-time predictions not found at %s. Run scripts/export_canonical_oil.py first.",
            PREDS_PATH,
        )
        return 1

    log.info("Loading canonical_test and training-time predictions")
    saved = pd.read_parquet(PREDS_PATH)
    test_raw = pd.read_parquet(CANONICAL_DIR / "test.parquet")
    test = _build_split(compute_derived_features(test_raw))

    if len(test) != len(saved):
        log.error("Row count mismatch: test=%d, saved=%d", len(test), len(saved))
        return 1

    # Verify the saved predictions match what we'd get by calling the
    # canonical model directly on the full feature matrix (the
    # "training-time path"). This first check is the canary — if it
    # fails, the saved predictions are stale (re-run export).
    feature_cols = _feature_columns(test)
    canonical_inference._ensure_loaded()
    if feature_cols != canonical_inference._columns:
        log.error(
            "Feature column ordering disagrees between test-side derivation and bundled feature_columns.json. "
            "test-side n=%d, bundled n=%d",
            len(feature_cols), len(canonical_inference._columns),
        )
        # Show first divergence
        for i, (a, b) in enumerate(zip(feature_cols, canonical_inference._columns)):
            if a != b:
                log.error("  index %d: test=%r bundled=%r", i, a, b)
                break
        return 1

    X_te, _ = prepare_xy(test, feature_cols)
    full_path_preds = np.asarray(canonical_inference._model.predict(X_te), dtype=float)
    diff_full = np.abs(full_path_preds - saved["y_pred_canonical"].to_numpy())
    if diff_full.max() != 0.0:
        log.error(
            "FATAL: bundled predictions do not match a fresh model.predict() pass. "
            "max abs diff = %.10g (n_mismatch=%d). Re-run export.",
            float(diff_full.max()), int((diff_full != 0).sum()),
        )
        return 1
    log.info("Canary OK: bundled predictions reproduce model.predict() bit-identically.")

    # Now the real test: per-row live-mode inference must produce the
    # same predictions as the training-time batch path.
    log.info("Running per-row live-mode inference on %d test rows", len(test))
    live_preds = np.empty(len(test), dtype=float)
    for i in range(len(test)):
        row_df = test.iloc[[i]].copy()
        live_preds[i] = canonical_inference.predict(row_df)
        if (i + 1) % 1000 == 0:
            log.info("  %d / %d rows", i + 1, len(test))

    diff = np.abs(live_preds - saved["y_pred_canonical"].to_numpy())
    max_diff = float(diff.max())
    n_mismatch = int((diff != 0).sum())

    if max_diff == 0.0:
        log.info("=== PARITY CHECK PASSED ===")
        log.info("All %d live-mode predictions are byte-for-byte equal to training-time predictions.", len(test))
        return 0

    log.error("=== PARITY CHECK FAILED ===")
    log.error("max abs diff = %.10g across %d rows (%d nonzero diffs)", max_diff, len(test), n_mismatch)

    bad_idx = np.argsort(-diff)[:5]
    for k, idx in enumerate(bad_idx):
        if diff[idx] == 0:
            break
        log.error(
            "  mismatch #%d: row %d pair_id=%s y_pred_saved=%.10g y_pred_live=%.10g diff=%.10g",
            k + 1, int(idx),
            saved["pair_id"].iloc[idx] if "pair_id" in saved.columns else "?",
            float(saved["y_pred_canonical"].iloc[idx]),
            float(live_preds[idx]),
            float(diff[idx]),
        )
        row_df = test.iloc[[idx]].copy()
        built = canonical_inference.build_row(row_df)
        for col in canonical_inference._columns:
            train_val = float(X_te.iloc[idx][col]) if col in X_te.columns else float("nan")
            live_val = float(built[col].iloc[0])
            if train_val != live_val:
                log.error("    col %s: training=%.10g live=%.10g", col, train_val, live_val)

    return 1


if __name__ == "__main__":
    sys.exit(main())
