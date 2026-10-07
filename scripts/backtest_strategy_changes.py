#!/usr/bin/env python3
"""Replay position_history.jsonl under the new vs old exit rules.

We don't have bar-by-bar `spread_history` in the closed records — only
the summary fields `min_spread` and `max_spread`. But that's enough
to know whether the magnitude of the spread ever crossed below the
new take-profit threshold during the position's life.

Per-trade outcome under each rule:
  - if min|spread| over life <= TP_RATIO * |entry|, TP would have fired
    and captured approximately (1 - TP_RATIO) * |entry| of P&L
  - else if max|spread| over life > STOP_RATIO * |entry|, SL fires and
    realizes approx -(STOP_RATIO - 1) * |entry|
  - else the trade closes as it actually did (use the recorded
    realized_pnl)

This is a counterfactual approximation, not a full replay — but the
question "would the new TP threshold fire on this trade?" is answered
exactly. The P&L magnitude is an estimate.

Entry-side filter changes (is_commodity narrowing, 5× multiplier) are
NOT backtested here — we'd need the original feature snapshots and
model predictions to replay those decisions. This script covers only
the take-profit threshold change.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


OLD_TP_RATIO = 0.5
NEW_TP_RATIO = 0.7
STOP_RATIO = 1.3


def min_abs_spread(min_s: float, max_s: float) -> float:
    """Lowest spread magnitude reached given signed [min_s, max_s] range."""
    if min_s <= 0 <= max_s:
        return 0.0
    return min(abs(min_s), abs(max_s))


def max_abs_spread(min_s: float, max_s: float) -> float:
    return max(abs(min_s), abs(max_s))


def simulate_pnl(entry: float, min_s: float, max_s: float, actual_pnl: float,
                 tp_ratio: float, direction: str) -> tuple[float, str]:
    """Return (estimated_pnl, exit_reason) under the given TP ratio."""
    entry_abs = abs(entry)
    if entry_abs == 0:
        return actual_pnl, "RESOLUTION_EXIT"

    tp_threshold = tp_ratio * entry_abs
    sl_threshold = STOP_RATIO * entry_abs
    min_abs = min_abs_spread(min_s, max_s)
    max_abs = max_abs_spread(min_s, max_s)

    if min_abs <= tp_threshold:
        # PnL ≈ |entry| - threshold = (1 - tp_ratio) * |entry|
        return (1.0 - tp_ratio) * entry_abs, "TAKE_PROFIT"
    if max_abs > sl_threshold:
        return -(STOP_RATIO - 1.0) * entry_abs, "STOP_LOSS"
    return actual_pnl, "HOLD_TO_REAL_EXIT"


def main() -> None:
    history_path = Path("data/live/position_history.jsonl")
    if not history_path.exists():
        print(f"Missing {history_path}")
        sys.exit(1)

    records = []
    with open(history_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "realized_pnl" not in r or "min_spread" not in r:
                continue
            records.append(r)

    n = len(records)
    if n == 0:
        print("No usable records.")
        sys.exit(0)

    print(f"Replaying {n:,} closed positions...")

    old_total, new_total, actual_total = 0.0, 0.0, 0.0
    old_counts, new_counts = defaultdict(int), defaultdict(int)
    deltas = []
    helped, hurt = [], []

    for r in records:
        entry = float(r["entry_spread"])
        min_s = float(r["min_spread"])
        max_s = float(r["max_spread"])
        actual_pnl = float(r["realized_pnl"])
        direction = r["direction"]

        old_pnl, old_reason = simulate_pnl(entry, min_s, max_s, actual_pnl,
                                           OLD_TP_RATIO, direction)
        new_pnl, new_reason = simulate_pnl(entry, min_s, max_s, actual_pnl,
                                           NEW_TP_RATIO, direction)

        old_total += old_pnl
        new_total += new_pnl
        actual_total += actual_pnl
        old_counts[old_reason] += 1
        new_counts[new_reason] += 1
        delta = new_pnl - old_pnl
        deltas.append(delta)
        if delta > 0:
            helped.append((r["pair_id"], delta))
        elif delta < 0:
            hurt.append((r["pair_id"], delta))

    print()
    print(f"=== Actual recorded PnL ===")
    print(f"  Total realized:        ${actual_total:+.2f}")
    print()
    print(f"=== Old rule (TP at 50% narrowing) ===")
    print(f"  Estimated total PnL:   ${old_total:+.2f}")
    print(f"  Exit counts: {dict(old_counts)}")
    print()
    print(f"=== New rule (TP at 30% narrowing) ===")
    print(f"  Estimated total PnL:   ${new_total:+.2f}")
    print(f"  Exit counts: {dict(new_counts)}")
    print()
    delta_total = new_total - old_total
    n_h, n_hh = len(helped), len(hurt)
    print(f"=== Delta ===")
    print(f"  PnL improvement:       ${delta_total:+.2f}")
    print(f"  Trades helped:         {n_h}")
    print(f"  Trades hurt:           {n_hh}")
    print(f"  Trades unchanged:      {n - n_h - n_hh}")
    if n > 0:
        print(f"  Avg per-trade delta:   ${delta_total/n:+.4f}")

    old_tp = old_counts.get("TAKE_PROFIT", 0)
    new_tp = new_counts.get("TAKE_PROFIT", 0)
    print()
    print(f"  TAKE_PROFIT fire rate: {old_tp}/{n} ({100*old_tp/n:.2f}%) "
          f"→ {new_tp}/{n} ({100*new_tp/n:.2f}%)")

    out_path = Path("experiments/results/full_retrain/strategy_backtest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "n_trades_replayed": n,
            "actual_total_pnl": actual_total,
            "old_rule": {
                "tp_ratio": OLD_TP_RATIO,
                "estimated_total_pnl": old_total,
                "exit_counts": dict(old_counts),
            },
            "new_rule": {
                "tp_ratio": NEW_TP_RATIO,
                "estimated_total_pnl": new_total,
                "exit_counts": dict(new_counts),
            },
            "delta_pnl": delta_total,
            "trades_helped": n_h,
            "trades_hurt": n_hh,
            "trades_unchanged": n - n_h - n_hh,
        }, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
