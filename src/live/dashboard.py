"""Paper trading dashboard for monitoring per-model performance.

Reads the paper_trades.jsonl log and live bars to compute and display
per-model cumulative P&L, win rate, trade counts, and signal stats.

Usage:
    python -m src.live.dashboard              # formatted table
    python -m src.live.dashboard --json       # JSON output
    python -m src.live.dashboard --help
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fixed model display order (same as run_baselines.py).
_MODEL_ORDER = [
    "Naive (Spread Closes)",
    "Volume (Higher Volume Correct)",
    "Linear Regression",
    "XGBoost",
    "GRU",
    "LSTM",
    "PPO-Raw",
    "PPO-Filtered",
]


def load_trade_log(path: Path) -> list[dict]:
    """Load paper trades from a JSONL file.

    Args:
        path: Path to paper_trades.jsonl.

    Returns:
        List of trade dicts. Empty list if file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        return []

    trades: list[dict] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                trades.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON at line %d: %s", line_num, e)
    return trades


def load_all_trade_logs(live_dir: Path) -> list[dict]:
    """Load trades from every ``paper_trades*.jsonl`` file in ``live_dir``.

    Covers both the legacy frozen archive (``paper_trades.jsonl``) and
    all per-UTC-day rotated files (``paper_trades_YYYY-MM-DD.jsonl``)
    created after task #27's size-fix rollout.

    Args:
        live_dir: Directory to scan.

    Returns:
        Concatenated list of trade dicts across all matching files.
        Empty list if the directory does not exist or has no matching
        files.
    """
    live_dir = Path(live_dir)
    if not live_dir.exists():
        return []

    trades: list[dict] = []
    # sorted() so tests and humans see deterministic ordering
    for p in sorted(live_dir.glob("paper_trades*.jsonl")):
        trades.extend(load_trade_log(p))
    return trades


def compute_paper_pnl(
    trades: list[dict],
    resolution_data: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Compute per-model paper P&L from the trade log.

    Groups trades by model name and computes statistics. When resolution
    data is available (future bars showing actual spread changes), actual
    P&L is computed. Otherwise, only signal/trade counts and prediction
    magnitudes are reported.

    Args:
        trades: List of trade dicts from load_trade_log().
        resolution_data: DataFrame with columns [pair_id, time_idx, spread]
            for matching trades to outcomes. If None, P&L is not computed.

    Returns:
        Dict mapping model_name -> {signals, trades, wins, losses,
        cumulative_pnl, win_rate, avg_prediction}.
    """
    if not trades:
        return {}

    # Group by model
    by_model: dict[str, list[dict]] = {}
    for t in trades:
        model = t.get("model", "Unknown")
        by_model.setdefault(model, []).append(t)

    # Build resolution lookup if available
    # Key: (pair_id, timestamp) -> next_spread
    resolution_map: dict[tuple[str, int], float] | None = None
    if resolution_data is not None and not resolution_data.empty:
        resolution_map = {}
        for pid, grp in resolution_data.groupby("pair_id"):
            grp_sorted = grp.sort_values("time_idx")
            timestamps = grp_sorted["timestamp"].values
            spreads = grp_sorted["spread"].values
            for i in range(len(timestamps) - 1):
                resolution_map[(str(pid), int(timestamps[i]))] = float(
                    spreads[i + 1]
                )

    results: dict[str, dict] = {}

    for model_name, model_trades in by_model.items():
        signals = len(model_trades)
        actual_trades = [t for t in model_trades if t.get("trade", False)]
        n_trades = len(actual_trades)

        # Average prediction magnitude (all signals)
        preds = [abs(t.get("prediction", 0.0)) for t in model_trades]
        avg_prediction = float(np.mean(preds)) if preds else 0.0

        # Try to compute P&L if resolution data available
        wins = 0
        losses = 0
        cumulative_pnl = 0.0

        if resolution_map and actual_trades:
            for t in actual_trades:
                key = (t["pair_id"], t["timestamp"])
                if key in resolution_map:
                    next_spread = resolution_map[key]
                    current_spread = t.get("spread", 0.0)
                    actual_change = next_spread - current_spread

                    # P&L based on direction
                    if t.get("direction") == "long_spread":
                        pnl = actual_change  # bet spread widens
                    else:
                        pnl = -actual_change  # bet spread narrows

                    cumulative_pnl += pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1

        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.0

        results[model_name] = {
            "signals": signals,
            "trades": n_trades,
            "wins": wins,
            "losses": losses,
            "cumulative_pnl": round(cumulative_pnl, 6),
            "win_rate": round(win_rate, 4),
            "avg_prediction": round(avg_prediction, 6),
        }

    return results


def _format_table(
    results: dict[str, dict],
    collection_stats: dict | None = None,
) -> str:
    """Format results as a fixed-width dashboard table.

    Args:
        results: Per-model stats from compute_paper_pnl().
        collection_stats: Optional dict with total_bars, date_range,
            active_pairs, last_collection.

    Returns:
        Formatted multi-line string.
    """
    lines: list[str] = []
    lines.append("====== Paper Trading Dashboard ======")
    lines.append("")

    # Collection stats
    if collection_stats:
        lines.append("Collection stats:")
        lines.append(
            f"  Total bars collected: {collection_stats.get('total_bars', 0)}"
        )
        dr = collection_stats.get("date_range", ("N/A", "N/A"))
        lines.append(f"  Date range: {dr[0]} to {dr[1]}")
        lines.append(
            f"  Active pairs: {collection_stats.get('active_pairs', 0)}"
        )
        lines.append("")

    if not results:
        lines.append("No trades yet. Run the paper trader first:")
        lines.append("  python -m src.live.paper_trader --skip-tier3")
        return "\n".join(lines)

    # Table header
    name_width = max(
        max((len(name) for name in results), default=5), len("Model")
    )

    header = (
        f"{'Model':<{name_width}} | {'Signals':>7} | {'Trades':>6} "
        f"| {'Win Rate':>8} | {'Cum P&L':>9} | {'Avg |Pred|':>10}"
    )
    separator = (
        f"{'-' * name_width}-+-{'-' * 7}-+-{'-' * 6}"
        f"-+-{'-' * 8}-+-{'-' * 9}-+-{'-' * 10}"
    )

    lines.append("Per-model performance:")
    lines.append(header)
    lines.append(separator)

    # Sort by model order
    def _sort_key(name: str) -> int:
        try:
            return _MODEL_ORDER.index(name)
        except ValueError:
            return len(_MODEL_ORDER)

    for model_name in sorted(results.keys(), key=_sort_key):
        stats = results[model_name]
        wr_str = (
            f"{stats['win_rate']:>7.1%}"
            if stats["wins"] + stats["losses"] > 0
            else "    N/A"
        )
        pnl_str = (
            f"{stats['cumulative_pnl']:>9.4f}"
            if stats["wins"] + stats["losses"] > 0
            else "      N/A"
        )

        lines.append(
            f"{model_name:<{name_width}} | "
            f"{stats['signals']:>7} | "
            f"{stats['trades']:>6} | "
            f"{wr_str} | "
            f"{pnl_str} | "
            f"{stats['avg_prediction']:>10.6f}"
        )

    # Footer
    lines.append("")
    if collection_stats and collection_stats.get("last_collection"):
        lines.append(
            f"Last collection: {collection_stats['last_collection']}"
        )

    return "\n".join(lines)


def _get_collection_stats(live_dir: Path) -> dict | None:
    """Extract collection stats from bars.parquet if it exists."""
    bars_path = live_dir / "bars.parquet"
    if not bars_path.exists():
        return None

    try:
        df = pd.read_parquet(bars_path)
        if df.empty:
            return None

        from datetime import datetime, timezone

        timestamps = df["timestamp"].values
        min_ts = int(timestamps.min())
        max_ts = int(timestamps.max())

        return {
            "total_bars": len(df),
            "date_range": (
                datetime.fromtimestamp(min_ts, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M"
                ),
                datetime.fromtimestamp(max_ts, tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M"
                ),
            ),
            "active_pairs": df["pair_id"].nunique(),
            "last_collection": datetime.fromtimestamp(
                max_ts, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    except Exception as e:
        logger.warning("Could not read bars.parquet: %s", e)
        return None


def print_dashboard(live_dir: Path = Path("data/live")) -> None:
    """Load trade log and print the dashboard.

    Args:
        live_dir: Directory containing paper_trades*.jsonl + bars.parquet.
    """
    live_dir = Path(live_dir)
    bars_path = live_dir / "bars.parquet"

    # Covers both the legacy archive and all daily-rotated files.
    trades = load_all_trade_logs(live_dir)

    # Load bars for resolution matching (if available)
    resolution_data = None
    if bars_path.exists():
        try:
            resolution_data = pd.read_parquet(bars_path)
        except Exception:
            pass

    results = compute_paper_pnl(trades, resolution_data=resolution_data)
    collection_stats = _get_collection_stats(live_dir)

    print(_format_table(results, collection_stats=collection_stats))


def _derive_pair_category(pair_id: str, active_matches: list[dict] | None = None) -> str:
    """Return category for a live_NNNN pair by looking up its Kalshi
    ticker in active_matches.json, then routing through
    derive_category_from_ticker. Returns 'other' on any miss."""
    from src.features.category import derive_category_from_ticker

    if active_matches is None:
        return "other"
    try:
        idx = int(pair_id.split("_", 1)[1])
    except (IndexError, ValueError):
        return "other"
    if idx >= len(active_matches):
        return "other"
    return derive_category_from_ticker(active_matches[idx].get("kalshi_ticker", ""))


def print_category_breakdown(
    live_dir: Path = Path("data/live"),
    min_price: float = 0.10,
    min_spread: float = 0.30,
    fees_pp: float = 0.02,
) -> None:
    """Print a per-category P&L table — the task #26 monitoring view.

    When a single category materially outperforms the pooled model, it
    is time to revisit Option B (per-category models). The trigger
    heuristic from task #26's design notes:

      * Category has at least 50 resolved trades (enough signal to
        not be noise)
      * Category fee-adjusted P&L > pooled-model fee-adjusted P&L
        by > 15% (material edge)

    This function does NOT train anything — it just prints the table
    plus a flag line when the trigger fires.
    """
    import json as _json

    live_dir = Path(live_dir)

    # Load matches for pair_id -> ticker -> category lookup
    matches_path = live_dir / "active_matches.json"
    active_matches: list[dict] | None = None
    if matches_path.exists():
        try:
            with open(matches_path) as f:
                active_matches = _json.load(f)
        except Exception:
            active_matches = None

    # Load trades + bars
    trades = load_all_trade_logs(live_dir)
    if not trades:
        print("No trades in data/live/paper_trades*.jsonl — nothing to break down.")
        return

    bars_path = live_dir / "bars.parquet"
    if not bars_path.exists():
        print("No bars.parquet — can't compute realized P&L.")
        return
    bars = pd.read_parquet(bars_path).sort_values(["pair_id", "timestamp"])

    # Build (pair_id, timestamp) -> next_spread lookup for resolution
    next_spread: dict[tuple[str, int], float] = {}
    for pid, g in bars.groupby("pair_id"):
        ts = g["timestamp"].values
        sp = g["spread"].values
        for i in range(len(ts) - 1):
            next_spread[(str(pid), int(ts[i]))] = float(sp[i + 1])

    # Resolve each trade and tag with category
    rows: list[dict] = []
    for t in trades:
        if not t.get("trade", False):
            continue
        k = (t["pair_id"], int(t["timestamp"]))
        if k not in next_spread:
            continue
        # Trade-level filters (same as the live strategy)
        if t.get("kalshi_price", 0) < min_price:
            continue
        if t.get("polymarket_price", 0) < min_price:
            continue
        if abs(t.get("spread", 0)) < min_spread:
            continue

        next_sp = next_spread[k]
        delta = next_sp - t["spread"]
        pnl = delta if t.get("direction") == "long_spread" else -delta

        category = _derive_pair_category(t["pair_id"], active_matches)
        rows.append({
            "pair_id": t["pair_id"],
            "model": t.get("model", "?"),
            "category": category,
            "pnl": pnl,
        })

    if not rows:
        print("No resolved trades passing filters — cannot break down by category.")
        return

    df = pd.DataFrame(rows)
    # Focus on Linear Regression (our current best fee-adjusted model)
    lr = df[df["model"] == "Linear Regression"]
    if lr.empty:
        lr = df  # fall back to everything if LR absent

    # Pooled stats
    pooled_n = len(lr)
    pooled_gross = float(lr["pnl"].sum())
    pooled_fees = fees_pp * pooled_n
    pooled_net = pooled_gross - pooled_fees
    pooled_wr = float((lr["pnl"] > 0).mean())
    pooled_per_trade_net = pooled_net / pooled_n if pooled_n > 0 else 0.0

    # Per-category stats
    by_cat = lr.groupby("category").agg(
        trades=("pnl", "size"),
        win_rate=("pnl", lambda s: (s > 0).mean()),
        gross=("pnl", "sum"),
    )
    by_cat["fees"] = fees_pp * by_cat["trades"]
    by_cat["net"] = by_cat["gross"] - by_cat["fees"]
    by_cat["net_per_trade"] = by_cat["net"] / by_cat["trades"]
    by_cat = by_cat.sort_values("net", ascending=False)

    # Print
    print("=" * 80)
    print(f"Per-Category P&L Breakdown (Linear Regression, {pooled_n} resolved trades)")
    print("=" * 80)
    print(
        f"{'category':<20} {'trades':>8} {'win%':>7} {'gross':>9} "
        f"{'net@2pp':>9} {'$/trade':>9}"
    )
    print("-" * 80)
    for cat, row in by_cat.iterrows():
        print(
            f"{cat:<20} {int(row['trades']):>8} {row['win_rate']:>6.1%} "
            f"{row['gross']:>+9.2f} {row['net']:>+9.2f} {row['net_per_trade']:>+9.4f}"
        )
    print("-" * 80)
    print(
        f"{'POOLED':<20} {pooled_n:>8} {pooled_wr:>6.1%} "
        f"{pooled_gross:>+9.2f} {pooled_net:>+9.2f} {pooled_per_trade_net:>+9.4f}"
    )
    print()

    # Trigger check (task #26)
    triggered = []
    for cat, row in by_cat.iterrows():
        if int(row["trades"]) < 50:
            continue
        if pooled_per_trade_net <= 0 and row["net_per_trade"] > 0:
            edge_pct = float("inf")
        elif pooled_per_trade_net <= 0:
            continue
        else:
            edge_pct = (row["net_per_trade"] / pooled_per_trade_net) - 1.0
        if edge_pct > 0.15:
            triggered.append((cat, row["trades"], row["net_per_trade"], edge_pct))

    if triggered:
        print(">>> TASK #26 TRIGGER: category-specific models worth trying <<<")
        for cat, n, ptr, edge in triggered:
            edge_str = "inf" if edge == float("inf") else f"{edge:+.1%}"
            print(
                f"    {cat}: {int(n)} trades, ${ptr:+.4f}/trade, "
                f"edge={edge_str} vs pooled"
            )
        print()
        print("Next step: train a category-specific XGBoost on oil data only,")
        print("A/B against the pooled model on the same subset, decide from delta.")
    else:
        print("Task #26 trigger: no category meets the criterion yet")
        print("  (need >= 50 trades AND > 15% per-trade edge vs pooled)")


def print_dashboard_json(live_dir: Path = Path("data/live")) -> None:
    """Load trade log and print the dashboard as JSON.

    Args:
        live_dir: Directory containing paper_trades*.jsonl + bars.parquet.
    """
    live_dir = Path(live_dir)
    bars_path = live_dir / "bars.parquet"

    trades = load_all_trade_logs(live_dir)

    resolution_data = None
    if bars_path.exists():
        try:
            resolution_data = pd.read_parquet(bars_path)
        except Exception:
            pass

    results = compute_paper_pnl(trades, resolution_data=resolution_data)
    collection_stats = _get_collection_stats(live_dir)

    output = {
        "collection_stats": collection_stats,
        "models": results,
        "total_trades": len(trades),
    }
    print(json.dumps(output, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the dashboard."""
    parser = argparse.ArgumentParser(
        description="Paper trading performance dashboard",
        prog="python -m src.live.dashboard",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (for programmatic consumption)",
    )
    parser.add_argument(
        "--by-category",
        action="store_true",
        help=(
            "Task #26 monitor: print per-category fee-adjusted P&L and "
            "flag any category that materially outperforms the pooled "
            "model (50+ trades, >15%% per-trade edge). Use to decide "
            "when to train category-specific models (Option B)."
        ),
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

    live_dir = Path(args.live_dir)

    if args.by_category:
        print_category_breakdown(live_dir)
    elif args.json:
        print_dashboard_json(live_dir)
    else:
        print_dashboard(live_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
