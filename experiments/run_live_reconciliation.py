"""Run live vs backtest reconciliation on closed paper-trading positions.

Replays each closed position's entry bar through the deployed models and
compares the shadow-simulation P&L against the actual live P&L.  All
reconciliation logic lives in src/analysis/reconciliation.py.

Run:
    python -m experiments.run_live_reconciliation
    python -m experiments.run_live_reconciliation --dry-run
    python -m experiments.run_live_reconciliation --db data/live/positions.db
    python -m experiments.run_live_reconciliation --window-start 2026-04-14
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.analysis.reconciliation import (
    acceptance_gate,
    build_summary,
    category_breakdown,
    exit_reason_attribution,
    filter_window,
    load_closed_positions,
    run_shadow_simulation,
)

OUTPUT_DIR = Path("experiments/results/reconciliation")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live vs backtest reconciliation for closed paper-trading positions."
    )
    parser.add_argument("--db", default="data/live/positions.db",
                        help="Path to positions.db (default: data/live/positions.db)")
    parser.add_argument("--bars", default="data/live/bars.parquet",
                        help="Path to bars.parquet (default: data/live/bars.parquet)")
    parser.add_argument("--models-dir", default="models/deployed",
                        help="Path to deployed models directory (default: models/deployed)")
    parser.add_argument("--window-start", default="2026-04-11",
                        help="Reconciliation window start ISO date (default: 2026-04-11)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load positions and print count, skip simulation")
    args = parser.parse_args()

    positions = load_closed_positions(args.db)
    filtered = filter_window(positions, start=args.window_start)

    if args.dry_run:
        print(f"Loaded {len(filtered)} positions in window. --dry-run: no simulation run.")
        sys.exit(0)

    # Run shadow simulation
    all_results = run_shadow_simulation(filtered, args.bars, args.models_dir)
    matched = [r for r in all_results if r["matched"]]
    unmatched_count = len(all_results) - len(matched)

    # Build summary and breakdowns (all logic delegated to module)
    summary = build_summary(matched)
    summary["unmatched_count"] = unmatched_count
    summary["gap_metric"] = unmatched_count / len(all_results) if all_results else 0.0
    summary["total_positions"] = len(all_results)

    cat_breakdown = category_breakdown(matched)
    exit_breakdown = exit_reason_attribution(matched)
    gate_passed = acceptance_gate(len(matched), len(all_results))

    # Print formatted comparison table
    print("\n=== Live vs Shadow-Simulation Reconciliation ===\n")
    print(f"  Positions in window:  {summary['total_positions']}")
    print(f"  Matched:              {summary['matched_count']}")
    print(f"  Unmatched:            {summary['unmatched_count']}")
    print(f"  Acceptance gate:      {'PASSED' if gate_passed else 'FAILED'}\n")
    print(f"  Live P&L:             ${summary['live_total_pnl']:.2f}")
    print(f"  Shadow-sim P&L:       ${summary['sim_total_pnl']:.2f}")
    print(f"  Tracking error:       ${summary['tracking_error']:.2f}")

    # Write artifacts (overwrite canonical artifacts on each run)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "summary": summary,
        "category_breakdown": cat_breakdown,
        "exit_reason_attribution": exit_breakdown,
        "acceptance_gate_passed": gate_passed,
        "window_start": args.window_start,
        "window_end": "current",
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(output, f, indent=2)

    import pandas as pd
    pd.DataFrame(all_results).to_csv(OUTPUT_DIR / "per_position.csv", index=False)

    print(f"\nArtifacts written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
