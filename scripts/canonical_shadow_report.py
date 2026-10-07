#!/usr/bin/env python3
"""Phase A shadow-mode monitoring report.

Reads data/live/canonical_predictions.jsonl and produces a summary of:
    - count of shadow predictions per oil-family ticker prefix
    - count of predictions that "would have traded" (|pred| > 0.001)
    - KS statistic of live-pred distribution vs canonical_test pred
      distribution (per family prefix)
    - parity-drift counter (re-runs verify_canonical_parity equivalent
      and reports max abs diff — expected 0.0)

Designed for the 24h shadow-window checkpoint described in
phase_a_v3.md.

Usage:
    python scripts/canonical_shadow_report.py
    python scripts/canonical_shadow_report.py --since 2026-05-24T00:00:00Z
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_baselines import _build_split  # noqa: E402
from src.features.engineering import compute_derived_features  # noqa: E402
from src.live import canonical_inference  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("shadow_report")

SHADOW_LOG = REPO_ROOT / "data" / "live" / "canonical_predictions.jsonl"
PREDS_PATH = REPO_ROOT / "models" / "canonical_oil" / "canonical_test_predictions.parquet"
CANONICAL_TEST = REPO_ROOT / "data" / "processed" / "canonical_oil" / "test.parquet"

OIL_PREFIXES = canonical_inference.OIL_FAMILY_PREFIXES


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic. No scipy dep — manual ECDF."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a = np.sort(a)
    b = np.sort(b)
    grid = np.union1d(a, b)
    ecdf_a = np.searchsorted(a, grid, side="right") / len(a)
    ecdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(ecdf_a - ecdf_b)))


def _prefix(ticker: str) -> str:
    for p in OIL_PREFIXES:
        if ticker.startswith(p):
            return p
    return "?"


def shadow_section(records: list[dict]) -> dict:
    out: dict = {"total": len(records)}
    by_prefix: dict[str, list[float]] = defaultdict(list)
    would_trade = Counter()
    mode_counts = Counter()
    for r in records:
        pref = _prefix(r.get("kalshi_ticker", ""))
        by_prefix[pref].append(float(r["canonical_pred"]))
        if r.get("canonical_would_trade"):
            would_trade[pref] += 1
        mode_counts[r.get("mode", "?")] += 1
    out["by_prefix_count"] = {k: len(v) for k, v in by_prefix.items()}
    out["by_prefix_would_trade"] = dict(would_trade)
    out["mode_counts"] = dict(mode_counts)
    if records:
        first_ts = records[0].get("ts_iso", "?")
        last_ts = records[-1].get("ts_iso", "?")
        out["first_ts"] = first_ts
        out["last_ts"] = last_ts
    return out, by_prefix


def ks_drift_section(by_prefix: dict[str, list[float]]) -> dict:
    if not PREDS_PATH.exists():
        log.warning("Bundle predictions not found at %s — skipping KS section.", PREDS_PATH)
        return {"available": False}
    bundle = pd.read_parquet(PREDS_PATH)
    bundle_preds = bundle["y_pred_canonical"].to_numpy()
    out = {"available": True, "bundle_n": int(len(bundle_preds))}
    for prefix, live_preds in by_prefix.items():
        ks = _ks_statistic(np.asarray(live_preds), bundle_preds)
        out[prefix] = {
            "live_n": len(live_preds),
            "ks_stat": ks,
            "live_mean": float(np.mean(live_preds)) if live_preds else float("nan"),
            "live_std": float(np.std(live_preds)) if live_preds else float("nan"),
            "bundle_mean": float(np.mean(bundle_preds)),
            "bundle_std": float(np.std(bundle_preds)),
        }
    return out


def parity_drift_section() -> dict:
    """Re-run the parity check on the bundled canonical_test slice."""
    if not (PREDS_PATH.exists() and CANONICAL_TEST.exists()):
        return {"available": False}
    test = _build_split(compute_derived_features(pd.read_parquet(CANONICAL_TEST)))
    saved = pd.read_parquet(PREDS_PATH)
    if len(test) != len(saved):
        return {"available": True, "status": "ROW_COUNT_MISMATCH"}
    canonical_inference._ensure_loaded()
    live = np.empty(len(test), dtype=float)
    for i in range(len(test)):
        live[i] = canonical_inference.predict(test.iloc[[i]].copy())
    diff = np.abs(live - saved["y_pred_canonical"].to_numpy())
    return {
        "available": True,
        "status": "PASS" if diff.max() == 0.0 else "FAIL",
        "max_abs_diff": float(diff.max()),
        "n_mismatch": int((diff != 0).sum()),
        "n_total": int(len(test)),
    }


def load_records(since_iso: str | None) -> list[dict]:
    if not SHADOW_LOG.exists():
        return []
    cutoff = None
    if since_iso:
        try:
            cutoff = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
        except Exception:
            log.warning("Could not parse --since=%r; ignoring.", since_iso)
    records: list[dict] = []
    with open(SHADOW_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if cutoff is not None:
                ts_iso = r.get("ts_iso")
                if ts_iso:
                    try:
                        ts = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
                        if ts < cutoff:
                            continue
                    except Exception:
                        pass
            records.append(r)
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="ISO timestamp; drop records older than this")
    ap.add_argument("--json", action="store_true", help="Emit JSON to stdout instead of pretty text")
    args = ap.parse_args()

    records = load_records(args.since)
    shadow, by_prefix = shadow_section(records)
    ks = ks_drift_section(by_prefix)
    parity = parity_drift_section()
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "shadow_log_path": str(SHADOW_LOG),
        "since": args.since,
        "shadow": shadow,
        "ks_drift": ks,
        "parity_drift": parity,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if parity.get("status", "PASS") == "PASS" else 1

    print("=== Phase A shadow-mode report ===")
    print(f"Generated: {report['generated_at']}")
    print(f"Log: {SHADOW_LOG}")
    if args.since:
        print(f"Since:    {args.since}")
    print()
    print(f"Total shadow records: {shadow['total']}")
    print(f"  by prefix: {shadow.get('by_prefix_count', {})}")
    print(f"  would-trade by prefix: {shadow.get('by_prefix_would_trade', {})}")
    print(f"  modes seen: {shadow.get('mode_counts', {})}")
    if "first_ts" in shadow:
        print(f"  window: {shadow['first_ts']}  →  {shadow['last_ts']}")
    print()
    print("--- KS drift (live vs canonical_test predictions) ---")
    if ks.get("available"):
        for prefix, d in ks.items():
            if not isinstance(d, dict):
                continue
            print(f"  {prefix}: n_live={d['live_n']} ks={d['ks_stat']:.4f} "
                  f"mean_live={d['live_mean']:.5f} mean_bundle={d['bundle_mean']:.5f}")
    else:
        print("  (bundle predictions not found)")
    print()
    print("--- Parity drift (re-verify on bundled canonical_test) ---")
    if parity.get("available"):
        print(f"  status: {parity['status']}")
        print(f"  max_abs_diff: {parity['max_abs_diff']}")
        print(f"  n_mismatch / n_total: {parity['n_mismatch']} / {parity['n_total']}")
    else:
        print("  (not available)")

    return 0 if parity.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
