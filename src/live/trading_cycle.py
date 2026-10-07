"""Single-script trading cycle for cron deployment.

Main entrypoint for the adaptive trading system. Combines contract
classification, position management, and model inference into a
single command suitable for Oracle VM cron execution.

Usage:
    python -m src.live.trading_cycle              # one cycle
    python -m src.live.trading_cycle --loop        # continuous (5min interval)
    python -m src.live.trading_cycle --status       # show open positions
    python -m src.live.trading_cycle --dry-run      # cycle without saving
    python -m src.live.trading_cycle --export-models # train + save pickles
    python -m src.live.trading_cycle --help
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _run_export_models(model_dir: Path, data_dir: Path) -> int:
    """Train LR + XGBoost on train.parquet and export as pickle files."""
    from scripts.export_models import export_models
    export_models(data_dir=data_dir, model_dir=model_dir)
    return 0


def _show_status(live_dir: Path) -> int:
    """Print a table of open positions with unrealized P&L."""
    from src.live.position_manager import PositionManager

    pm = PositionManager(
        db_path=str(live_dir / "positions.db"),
        history_jsonl_path=str(live_dir / "position_history.jsonl"),
    )

    positions = pm.get_open_positions()
    if not positions:
        print("No open positions.")
        return 0

    # Header
    print(f"\n{'Pair ID':<12} {'Ticker':<30} {'Dir':<14} "
          f"{'Entry':>7} {'Current':>8} {'P&L':>8} {'Bars':>5} {'Tier':<10}")
    print("-" * 105)

    total_pnl = 0.0
    for pair_id, pos in sorted(positions.items()):
        ticker_short = pos.kalshi_ticker[:28]
        total_pnl += pos.unrealized_pnl
        print(
            f"{pair_id:<12} {ticker_short:<30} {pos.direction:<14} "
            f"{pos.entry_spread:>7.4f} {pos.current_spread:>8.4f} "
            f"{pos.unrealized_pnl:>+8.4f} {pos.bars_held:>5d} {pos.tier:<10}"
        )

    print("-" * 105)
    print(f"{'Total':>66} {total_pnl:>+8.4f} {len(positions):>5d} positions")

    # Recent closed positions
    closed = pm.get_closed_positions()
    if closed:
        print(f"\nRecent closed positions (last 5):")
        for rec in closed[:5]:
            print(
                f"  {rec['pair_id']:<12} {rec['exit_reason']:<16} "
                f"P&L={rec['realized_pnl']:>+.4f} "
                f"held={rec['bars_held']}bars "
                f"@ {rec['exit_time']}"
            )

    return 0


def _run_cycle(
    live_dir: Path,
    model_dir: Path,
    dry_run: bool = False,
    min_price: float = 0.10,
    min_spread: float = 0.30,
    threshold: float = 0.02,
) -> dict:
    """Run one trading cycle and print summary."""
    from src.live.strategy import TradingStrategy

    strategy = TradingStrategy(
        live_dir=live_dir,
        model_dir=model_dir,
        min_price=min_price,
        min_spread=min_spread,
        prediction_threshold=threshold,
    )
    summary = strategy.run_cycle(dry_run=dry_run)

    # Print human-readable summary
    prefix = "[DRY-RUN] " if dry_run else ""
    print(f"\n{prefix}Trading Cycle Summary")
    print(f"  Timestamp:      {summary['timestamp']}")
    print(f"  Pairs checked:  {summary['pairs_checked']}")
    print(f"  Prices fetched: {summary['prices_fetched']}")
    print(f"  Entries:        {summary['entries']}")
    print(f"  Exits:          {summary['exits']}")
    print(f"  Open positions: {summary['open_positions']}")

    if summary["entry_details"]:
        print("\n  New entries:")
        for e in summary["entry_details"]:
            print(
                f"    {e['pair_id']} {e['direction']} "
                f"spread={e['spread']:.4f} "
                f"pred={e['avg_pred']:.4f} ({e['tier']})"
            )

    if summary["exit_details"]:
        print("\n  Exits:")
        for x in summary["exit_details"]:
            if "realized_pnl" in x:
                print(
                    f"    {x['pair_id']} {x['exit_reason']} "
                    f"P&L={x['realized_pnl']:+.4f}"
                )
            else:
                print(f"    {x['pair_id']} {x.get('reason', 'unknown')}")

    return summary


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the adaptive trading cycle."""
    parser = argparse.ArgumentParser(
        description="Adaptive trading cycle for prediction market arbitrage",
        prog="python -m src.live.trading_cycle",
    )
    # Mode flags
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--cycle",
        action="store_true",
        default=True,
        help="Run one trading cycle (default)",
    )
    mode_group.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously every 300 seconds (5 minutes)",
    )
    mode_group.add_argument(
        "--status",
        action="store_true",
        help="Show current open positions and exit",
    )
    mode_group.add_argument(
        "--export-models",
        action="store_true",
        help="Train LR + XGBoost on train.parquet and save pickles",
    )

    # Options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run cycle without saving positions or bars",
    )
    parser.add_argument(
        "--live-dir",
        type=str,
        default="data/live",
        help="Live data directory (default: data/live)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/deployed",
        help="Model pickle directory (default: models/deployed)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Training data directory for --export-models (default: data/processed)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between cycles in --loop mode (default: 300)",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=0.10,
        help="Minimum price filter (default: 0.10)",
    )
    parser.add_argument(
        "--min-spread",
        type=float,
        default=0.30,
        help="Minimum absolute spread filter (default: 0.30)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Minimum |prediction| for entry (default: 0.02)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    live_dir = Path(args.live_dir)
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)

    # --- Export models ---
    if args.export_models:
        return _run_export_models(model_dir, data_dir)

    # --- Status ---
    if args.status:
        return _show_status(live_dir)

    # --- Loop mode ---
    if args.loop:
        logger.info("Starting continuous loop (interval=%ds)", args.interval)
        while True:
            try:
                _run_cycle(
                    live_dir=live_dir,
                    model_dir=model_dir,
                    dry_run=args.dry_run,
                    min_price=args.min_price,
                    min_spread=args.min_spread,
                    threshold=args.threshold,
                )
            except Exception as e:
                logger.error("Cycle failed: %s", e, exc_info=True)
            logger.info("Sleeping %ds until next cycle ...", args.interval)
            time.sleep(args.interval)
        return 0  # unreachable

    # --- Single cycle (default) ---
    try:
        _run_cycle(
            live_dir=live_dir,
            model_dir=model_dir,
            dry_run=args.dry_run,
            min_price=args.min_price,
            min_spread=args.min_spread,
            threshold=args.threshold,
        )
    except Exception as e:
        logger.error("Cycle failed: %s", e, exc_info=True)
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
