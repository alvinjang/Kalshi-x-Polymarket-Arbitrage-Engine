#!/usr/bin/env python3
"""Phase A.E1 realized-PNL report.

Consumes data/live/canonical_realized.jsonl (predictions enriched with
next-bar outcomes via src.live.canonical_inference.resolve_pending_shadow)
and computes per-trade alpha, Sharpe, and directional accuracy under
three time-window filter variants:

    Strict       — next bar within 20 min (~training cadence: 15 min)
    Moderate     — next bar within 30 min
    Unconstrained — any next bar (current resolver default)

Side-by-side comparison shows whether negative live alpha is driven by
the bar-cadence mismatch between training (15-min target) and live
(variable per pair) or by genuine distribution-driven model
degradation. Also surfaces the cadence distribution explicitly so
sample loss at each filter strictness is visible.

Trade rule (matches scripts/train_oil_canonical.py per_trade_outcomes()):
    Trade iff |canonical_pred| > 0.001 (PREDICTION_THRESHOLD)
    direction = sign(canonical_pred)
    per_trade_outcome = direction * realized_spread_change
Position-sizing: $100 per trade (matches the original paper).

Usage:
    python scripts/canonical_realized_report.py
    python scripts/canonical_realized_report.py --since 2026-05-26T00:00:00Z
    python scripts/canonical_realized_report.py --json
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

REPO_ROOT = Path(__file__).resolve().parent.parent
REALIZED_LOG = REPO_ROOT / "data" / "live" / "canonical_realized.jsonl"

PREDICTION_THRESHOLD = 0.001
POSITION_SIZE_USD = 100.0
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42

# Training-cadence reference: canonical_oil bars are 15-min by
# construction (the canonical_oil parquet was assembled from 15-min
# windowed snapshots). Strict apples-to-apples filter is ~training
# cadence plus a small slack for jitter.
STRICT_MAX_GAP_SEC = 20 * 60   # 20 min
MODERATE_MAX_GAP_SEC = 30 * 60  # 30 min

# Backtest reference (experiments/results/canonical_oil/headline/xgboost.json)
BACKTEST_ALPHA_BPS = 18.96
BACKTEST_SHARPE = 0.0644
BACKTEST_TRADE_RATE = 0.87

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("realized_report")

OIL_PREFIXES = (
    "KXWTIW", "KXWTI", "KXBRENTMON", "KXBRENTW", "KXBRENTD",
    "KXBRENT", "KXCRUDE", "KXDIESEL", "KXHEATINGOIL", "KXGASOLINE",
    "KXMEXCUBOIL",
)


def detailed_prefix(ticker: str) -> str:
    for p in OIL_PREFIXES:
        if ticker.startswith(p):
            return p
    return ticker.split("-")[0] if ticker else "?"


def gap_seconds(r: dict) -> float:
    """Time between prediction and the next-bar that resolved it."""
    return float(int(r["realized_next_ts"]) - int(r["ts"]))


def bootstrap_ci(values: np.ndarray, statistic, n_boot: int = N_BOOTSTRAP,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = values[idx]
    boot_stats = np.asarray([statistic(s) for s in samples], dtype=float)
    return (float(np.percentile(boot_stats, 2.5)),
            float(np.percentile(boot_stats, 97.5)))


def per_trade_sharpe(trades: np.ndarray) -> float:
    if len(trades) < 2:
        return 0.0
    std = float(np.std(trades, ddof=1))
    if std <= 0:
        return 0.0
    return float(np.mean(trades) / std)


def alpha_bps(trades: np.ndarray) -> float:
    if len(trades) == 0:
        return 0.0
    return float(np.mean(trades) * 10_000.0)


def load_records(since_iso: str | None) -> list[dict]:
    if not REALIZED_LOG.exists():
        return []
    cutoff = None
    if since_iso:
        try:
            cutoff = datetime.fromisoformat(since_iso.replace("Z", "+00:00"))
        except Exception:
            log.warning("could not parse --since %r; ignoring", since_iso)
    out: list[dict] = []
    with open(REALIZED_LOG) as f:
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
            out.append(r)
    return out


def cohort_stats(records: list[dict]) -> dict:
    n_total = len(records)
    if n_total == 0:
        return {"n_total": 0, "n_trades": 0, "trade_rate": 0.0,
                "alpha_bps": 0.0, "alpha_bps_ci": [float("nan")] * 2,
                "per_trade_sharpe": 0.0, "per_trade_sharpe_ci": [float("nan")] * 2,
                "directional_accuracy": float("nan"), "total_pnl_usd": 0.0}

    would = [r for r in records if r.get("canonical_would_trade")]
    n_trades = len(would)
    trade_rate = n_trades / n_total if n_total else 0.0

    if n_trades == 0:
        return {"n_total": n_total, "n_trades": 0, "trade_rate": 0.0,
                "alpha_bps": 0.0, "alpha_bps_ci": [float("nan")] * 2,
                "per_trade_sharpe": 0.0, "per_trade_sharpe_ci": [float("nan")] * 2,
                "directional_accuracy": float("nan"), "total_pnl_usd": 0.0}

    per_trade = np.asarray([r["realized_pnl_per_dollar"] for r in would], dtype=float)
    sign_matches = sum(1 for r in would if r.get("realized_sign_match"))

    return {
        "n_total": n_total,
        "n_trades": n_trades,
        "trade_rate": float(trade_rate),
        "alpha_bps": alpha_bps(per_trade),
        "alpha_bps_ci": list(bootstrap_ci(per_trade, alpha_bps)),
        "per_trade_sharpe": per_trade_sharpe(per_trade),
        "per_trade_sharpe_ci": list(bootstrap_ci(per_trade, per_trade_sharpe)),
        "directional_accuracy": float(sign_matches / n_trades) if n_trades else float("nan"),
        "total_pnl_usd": float(per_trade.sum() * POSITION_SIZE_USD),
    }


def cadence_buckets(records: list[dict]) -> dict:
    """Distribution of gap_seconds across the records (any-trade-eligibility)."""
    gaps = np.asarray([gap_seconds(r) for r in records], dtype=float)
    if len(gaps) == 0:
        return {"n": 0}
    buckets = {
        "le_15min": int((gaps <= 15 * 60).sum()),
        "le_20min": int((gaps <= 20 * 60).sum()),
        "le_30min": int((gaps <= 30 * 60).sum()),
        "30_60min": int(((gaps > 30 * 60) & (gaps <= 60 * 60)).sum()),
        "gt_60min": int((gaps > 60 * 60).sum()),
    }
    return {
        "n": int(len(gaps)),
        "median_sec": float(np.median(gaps)),
        "p25_sec": float(np.percentile(gaps, 25)),
        "p75_sec": float(np.percentile(gaps, 75)),
        "p95_sec": float(np.percentile(gaps, 95)),
        "max_sec": float(gaps.max()),
        "buckets": buckets,
    }


def filter_by_gap(records: list[dict], max_gap_sec: float | None) -> list[dict]:
    if max_gap_sec is None:
        return records
    return [r for r in records if gap_seconds(r) <= max_gap_sec]


def decision_route(strict: dict, moderate: dict, unconstrained: dict) -> str:
    """Routes A/B/C/D from the user's framework.

    A: strict alpha > 0 AND strict CI excludes zero (positive side)
    B: strict alpha > 0 AND strict CI straddles zero
    C: strict alpha < 0 AND strict CI excludes zero (negative side)
    D: strict trades < 100 — bar cadence is genuinely mostly slow,
       cannot deploy model at the cadence it was trained for
    """
    if strict["n_trades"] < 100:
        return "D"
    a = strict["alpha_bps"]
    lo, hi = strict["alpha_bps_ci"]
    if a > 0 and lo > 0:
        return "A"
    if a > 0 and lo <= 0 <= hi:
        return "B"
    if a < 0 and hi < 0:
        return "C"
    return "B"  # other edge cases default to "extend / inconclusive"


def fmt_ci(stat: float, ci: list[float]) -> str:
    if stat != stat or any(c != c for c in ci):
        return "n/a"
    return f"{stat:+8.2f}  [{ci[0]:+7.2f}, {ci[1]:+7.2f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="ISO timestamp; drop records older than this")
    ap.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    args = ap.parse_args()

    records = load_records(args.since)
    if not records:
        log.error("No records in %s%s", REALIZED_LOG, f" since {args.since}" if args.since else "")
        return 1

    cadence = cadence_buckets(records)

    strict_recs = filter_by_gap(records, STRICT_MAX_GAP_SEC)
    moderate_recs = filter_by_gap(records, MODERATE_MAX_GAP_SEC)
    unconstrained_recs = records

    strict_stats = cohort_stats(strict_recs)
    moderate_stats = cohort_stats(moderate_recs)
    uncon_stats = cohort_stats(unconstrained_recs)

    timestamps = sorted(r["ts_iso"] for r in records if "ts_iso" in r)

    # Per-prefix breakdowns at each filter level.
    def by_prefix(recs: list[dict]) -> dict[str, dict]:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in recs:
            groups[detailed_prefix(r["kalshi_ticker"])].append(r)
        return {k: cohort_stats(v) for k, v in groups.items()}

    strict_by_prefix = by_prefix(strict_recs)
    moderate_by_prefix = by_prefix(moderate_recs)
    uncon_by_prefix = by_prefix(unconstrained_recs)

    route = decision_route(strict_stats, moderate_stats, uncon_stats)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "realized_log_path": str(REALIZED_LOG),
        "since": args.since,
        "window": {
            "first_pred_iso": timestamps[0] if timestamps else None,
            "last_pred_iso": timestamps[-1] if timestamps else None,
            "n_records": len(records),
        },
        "cadence_distribution": cadence,
        "variants": {
            "strict_le_20min": {"stats": strict_stats, "by_prefix": strict_by_prefix},
            "moderate_le_30min": {"stats": moderate_stats, "by_prefix": moderate_by_prefix},
            "unconstrained": {"stats": uncon_stats, "by_prefix": uncon_by_prefix},
        },
        "backtest_reference": {
            "alpha_bps": BACKTEST_ALPHA_BPS,
            "per_trade_sharpe": BACKTEST_SHARPE,
            "trade_rate": BACKTEST_TRADE_RATE,
            "source": "experiments/results/canonical_oil/headline/xgboost.json",
        },
        "decision_route": route,
        "config": {
            "prediction_threshold": PREDICTION_THRESHOLD,
            "position_size_usd": POSITION_SIZE_USD,
            "strict_max_gap_sec": STRICT_MAX_GAP_SEC,
            "moderate_max_gap_sec": MODERATE_MAX_GAP_SEC,
            "bootstrap_resamples": N_BOOTSTRAP,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if route in ("A", "B") else 1

    print("=" * 78)
    print("Phase A.E1 — realized P&L report (canonical-oil shadow vs live)")
    print("=" * 78)
    print(f"Generated:   {report['generated_at']}")
    print(f"Log path:    {REALIZED_LOG}")
    if args.since:
        print(f"Since:       {args.since}")
    print(f"Window:      {report['window']['first_pred_iso']}  →  {report['window']['last_pred_iso']}")
    print(f"Records:     {report['window']['n_records']}")
    print()

    print("--- Bar cadence distribution (prediction → next-bar gap) ---")
    b = cadence["buckets"]
    n = cadence["n"]
    def pct(x): return f"{100*x/n:.1f}%" if n else "n/a"
    print(f"  <=15 min:   {b['le_15min']:>5}  ({pct(b['le_15min'])})")
    print(f"  <=20 min:   {b['le_20min']:>5}  ({pct(b['le_20min'])})  ← strict cutoff")
    print(f"  <=30 min:   {b['le_30min']:>5}  ({pct(b['le_30min'])})  ← moderate cutoff")
    print(f"  30-60 min:  {b['30_60min']:>5}  ({pct(b['30_60min'])})")
    print(f"  60+ min:    {b['gt_60min']:>5}  ({pct(b['gt_60min'])})")
    print(f"  median: {cadence['median_sec']/60:.1f} min  "
          f"p25: {cadence['p25_sec']/60:.1f} min  "
          f"p75: {cadence['p75_sec']/60:.1f} min  "
          f"p95: {cadence['p95_sec']/60:.1f} min  "
          f"max: {cadence['max_sec']/60:.1f} min")
    print()

    print("--- Side-by-side: three filter variants ---")
    print()
    print(f"  {'Variant':<20} {'n_tot':>6} {'n_tr':>6} {'trade%':>7}   {'alpha_bps  [95% CI]':<28}   {'sharpe  [95% CI]':<26}   {'dir_acc':>8}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*7}   {'-'*28}   {'-'*26}   {'-'*8}")
    for name, s in (
        ("Strict (<=20m)", strict_stats),
        ("Moderate (<=30m)", moderate_stats),
        ("Unconstrained", uncon_stats),
    ):
        dir_acc = s.get("directional_accuracy")
        dir_str = f"{dir_acc*100:.1f}%" if dir_acc is not None and not np.isnan(dir_acc) else "n/a"
        print(f"  {name:<20} {s['n_total']:>6} {s['n_trades']:>6} "
              f"{s['trade_rate']*100:>6.1f}%   "
              f"{fmt_ci(s['alpha_bps'], s['alpha_bps_ci']):<28}   "
              f"{fmt_ci(s['per_trade_sharpe'], s['per_trade_sharpe_ci']):<26}   "
              f"{dir_str:>8}")
    print()

    print("--- Backtest reference (experiments/results/canonical_oil/headline/xgboost.json) ---")
    print(f"  alpha (bps/trade):      +{BACKTEST_ALPHA_BPS:.2f}")
    print(f"  per-trade Sharpe:       +{BACKTEST_SHARPE:.4f}")
    print(f"  trade rate:             {BACKTEST_TRADE_RATE*100:.1f}%")
    print()

    print("--- Per-prefix breakdown ---")
    for label, prefix_dict in (
        ("STRICT (<=20m)", strict_by_prefix),
        ("MODERATE (<=30m)", moderate_by_prefix),
        ("UNCONSTRAINED", uncon_by_prefix),
    ):
        print(f"\n  ## {label}")
        print(f"  {'prefix':<13} {'n_tot':>6} {'n_tr':>5} {'trade%':>7} {'alpha_bps':>10} {'sharpe':>9} {'dir_acc':>8}")
        for prefix in sorted(prefix_dict, key=lambda k: -prefix_dict[k]["n_total"]):
            s = prefix_dict[prefix]
            dir_acc = s.get("directional_accuracy")
            dir_str = f"{dir_acc*100:.1f}%" if dir_acc is not None and not np.isnan(dir_acc) else "n/a"
            print(f"  {prefix:<13} {s['n_total']:>6} {s['n_trades']:>5} "
                  f"{s['trade_rate']*100:>6.1f}% {s['alpha_bps']:>+9.2f} "
                  f"{s['per_trade_sharpe']:>+9.4f} {dir_str:>8}")
    print()

    print("--- Sensitivity interpretation ---")
    sa = strict_stats["alpha_bps"]
    ma = moderate_stats["alpha_bps"]
    ua = uncon_stats["alpha_bps"]
    print(f"  Strict (<=20m):       alpha = {sa:+8.2f} bps  (n_trades = {strict_stats['n_trades']})")
    print(f"  Moderate (<=30m):     alpha = {ma:+8.2f} bps  (n_trades = {moderate_stats['n_trades']})")
    print(f"  Unconstrained:        alpha = {ua:+8.2f} bps  (n_trades = {uncon_stats['n_trades']})")
    if sa > 0 and ua < 0:
        interp = "Cadence mismatch is the binding constraint — model is fine at training cadence."
    elif sa < 0 and ua < sa:
        interp = "Strict is better than unconstrained (less negative), but still negative — partial drift + cadence both contribute."
    elif sa < 0 and ua < 0 and abs(sa - ua) < 2:
        interp = "Both variants negative and similar — cadence not the binding issue; genuine model drift."
    elif sa > 0 and ua > 0:
        interp = "Both positive — model is intact across cadences."
    else:
        interp = "Mixed signal — review per-prefix table to identify which cadences/prefixes drive the result."
    print(f"  → {interp}")
    print()

    print("--- Decision gate (per user's A/B/C/D framework) ---")
    routes = {
        "A": "Strict alpha positive, CI excludes zero → A2 narrow-replacement proposed (oil pairs with consistent 15-min cadence only)",
        "B": "Strict alpha positive but CI straddles zero → extend shadow 24-48h, gather more strict-cadence samples",
        "C": "Strict alpha negative with CI excluding zero → route C holds, document live degradation",
        "D": f"Strict filter yields {strict_stats['n_trades']} trades (<100) → model can't be deployed at training cadence; document as deployment-constraint finding",
    }
    print(f"  → ROUTE {route}: {routes[route]}")
    print()

    return 0 if route in ("A", "B") else 1


if __name__ == "__main__":
    sys.exit(main())
