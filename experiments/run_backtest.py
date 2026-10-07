"""Walk-forward backtest runner for all 8 models.

Retrains each model from scratch on train.parquet, runs predictions on
test.parquet, and feeds them through the WalkForwardBacktester from
``src.evaluation.backtester`` to produce honest, capital-normalized,
fee-adjusted Sharpe ratios.

Outputs per-model JSON files to ``experiments/results/backtest/``,
equity-curve and drawdown PNGs to ``experiments/figures/``, and a
comparison table printed to stdout showing backtested Sharpe alongside
the original (inflated) Sharpe from tier1/tier2/tier3 results.

Run:
    python -m experiments.run_backtest                  # all 8 models
    python -m experiments.run_backtest --tier 1         # Tier 1 only
    python -m experiments.run_backtest --tier 2         # Tier 2 only
    python -m experiments.run_backtest --tier 3         # Tier 3 only
    python -m experiments.run_backtest --threshold 0.05 # custom threshold
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.run_baselines import (
    _build_split,
    _feature_columns,
    load_train_test,
    prepare_xy,
    prepare_xy_for_seq,
    NON_FEATURE_COLUMNS,
)
from src.evaluation.backtester import WalkForwardBacktester, compute_break_even_fee
from src.evaluation.results_store import load_all_results, save_results
from src.utils.seed import set_all_seeds
from src.models.autoencoder import AnomalyDetectorAutoencoder
from src.models.gru import GRUPredictor
from src.models.linear_regression import LinearRegressionPredictor
from src.models.lstm import LSTMPredictor
from src.models.naive import NaivePredictor
from src.models.ppo_filtered import PPOFilteredPredictor
from src.models.ppo_raw import PPORawPredictor
from src.models.volume import VolumePredictor
from src.models.xgboost_model import XGBoostPredictor

BACKTEST_RESULTS_DIR = Path("experiments/results/backtest")
FIGURES_DIR = Path("experiments/figures")

# Old (inflated) Sharpe result directories for comparison table.
OLD_RESULTS_DIRS = {
    "tier1": Path("experiments/results/tier1"),
    "tier2": Path("experiments/results/tier2"),
    "tier3": Path("experiments/results/tier3"),
}

# Fixed display order matching run_baselines.py _MODEL_ORDER.
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

_TIER_MAP = {
    "Naive (Spread Closes)": 1,
    "Volume (Higher Volume Correct)": 1,
    "Linear Regression": 1,
    "XGBoost": 1,
    "GRU": 2,
    "LSTM": 2,
    "PPO-Raw": 3,
    "PPO-Filtered": 3,
}

_TIER_COLORS = {
    1: {
        "Naive (Spread Closes)": "#8DA0CB",
        "Volume (Higher Volume Correct)": "#A6CEE3",
        "Linear Regression": "#4C72B0",
        "XGBoost": "#1F4E79",
    },
    2: {
        "GRU": "#DD8452",
        "LSTM": "#E5A17A",
    },
    3: {
        "PPO-Raw": "#C44E52",
        "PPO-Filtered": "#D98880",
    },
}

_TIER_LINE_STYLES = {1: "-", 2: "--", 3: ":"}


def _get_color(model_name: str) -> str:
    """Get tier-distinct color for a model."""
    tier = _TIER_MAP.get(model_name, 1)
    return _TIER_COLORS[tier].get(model_name, "#333333")


def run_single_model_backtest(
    model_name: str,
    df_test: pd.DataFrame,
    predictions: np.ndarray,
    backtester: WalkForwardBacktester,
) -> dict:
    """Run a single model through the backtester and return results."""
    result = backtester.run(df_test, predictions)
    result["model_name"] = model_name
    return result


def run_all_backtests(
    tier: str = "all",
    threshold: float = 0.02,
    data_dir: Path = Path("data/processed"),
) -> list[dict]:
    """Train all models, run backtests, save results, return list of dicts.

    Args:
        tier: Which tier(s) to run: "1", "2", "3", or "all".
        threshold: Minimum absolute prediction for entry.
        data_dir: Path to directory containing train.parquet / test.parquet.

    Returns:
        List of per-model result dicts with backtest metrics.
    """
    # Load and prepare data
    train_raw, test_raw = load_train_test(data_dir)
    train = _build_split(train_raw)
    test = _build_split(test_raw)
    feature_cols = _feature_columns(train)

    n_train = len(train)
    n_test = len(test)
    n_features = len(feature_cols)

    print(
        f"Loaded {n_train} train rows, {n_test} test rows, "
        f"{n_features} features."
    )

    # Instantiate backtester with CONTEXT.md defaults
    backtester = WalkForwardBacktester(
        initial_capital=10_000.0,
        position_size=100.0,
        entry_cost_pp=0.03,
        exit_cost_pp=0.02,
        threshold=threshold,
    )

    all_results: list[dict] = []

    # ---- Tier 1 ----
    if tier in ("1", "all"):
        print("\n" + "=" * 60)
        print("TIER 1: Regression Baselines")
        print("=" * 60)

        X_train, y_train = prepare_xy(train, feature_cols)
        X_test, y_test = prepare_xy(test, feature_cols)

        tier1_models = [
            NaivePredictor(),
            VolumePredictor(),
            LinearRegressionPredictor(),
            XGBoostPredictor(n_estimators=200, max_depth=4, learning_rate=0.05),
        ]

        for model in tier1_models:
            try:
                print(f"\n--- {model.name} ---")
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                result = run_single_model_backtest(
                    model.name, test, predictions, backtester
                )

                # Compute break-even fee
                be_fee = compute_break_even_fee(
                    test, predictions, threshold=threshold
                )
                result["break_even_fee_pp"] = be_fee

                # Save to JSON
                metrics_for_json = {
                    "annualized_sharpe": result["annualized_sharpe"],
                    "total_pnl": result["total_pnl"],
                    "max_drawdown": result["max_drawdown"],
                    "calmar_ratio": result["calmar_ratio"],
                    "win_rate": result["win_rate"],
                    "num_trades": result["num_trades"],
                    "total_fees": result["total_fees"],
                    "avg_trade_duration_hours": result["avg_trade_duration_hours"],
                }
                extra = {
                    "threshold": threshold,
                    "n_train_rows": n_train,
                    "n_test_rows": n_test,
                    "n_features": n_features,
                    "break_even_fee_pp": be_fee,
                    "equity_curve": result["equity_curve"],
                    "daily_returns": result["daily_returns"],
                }
                save_results(
                    model.name, metrics_for_json, BACKTEST_RESULTS_DIR, extra=extra
                )

                all_results.append(result)
                print(
                    f"  Sharpe(BT): {result['annualized_sharpe']:.4f}  "
                    f"P&L: ${result['total_pnl']:.2f}  "
                    f"MaxDD: {result['max_drawdown']:.4f}  "
                    f"Trades: {result['num_trades']}  "
                    f"BE Fee: {be_fee:.4f}"
                )
            except Exception:
                print(f"  ERROR on {model.name}:")
                traceback.print_exc()

    # ---- Tier 2 ----
    if tier in ("2", "all"):
        print("\n" + "=" * 60)
        print("TIER 2: Time Series Models (seed=42)")
        print("=" * 60)

        X_train_seq, y_train = prepare_xy_for_seq(train, feature_cols)
        X_test_seq, y_test = prepare_xy_for_seq(test, feature_cols)

        tier2_models = [
            GRUPredictor(random_state=42),
            LSTMPredictor(random_state=42),
        ]

        for model in tier2_models:
            try:
                print(f"\n--- {model.name} (seed=42) ---")
                model.fit(X_train_seq, y_train)
                predictions = model.predict(X_test_seq)
                result = run_single_model_backtest(
                    model.name, test, predictions, backtester
                )

                be_fee = compute_break_even_fee(
                    test, predictions, threshold=threshold
                )
                result["break_even_fee_pp"] = be_fee

                metrics_for_json = {
                    "annualized_sharpe": result["annualized_sharpe"],
                    "total_pnl": result["total_pnl"],
                    "max_drawdown": result["max_drawdown"],
                    "calmar_ratio": result["calmar_ratio"],
                    "win_rate": result["win_rate"],
                    "num_trades": result["num_trades"],
                    "total_fees": result["total_fees"],
                    "avg_trade_duration_hours": result["avg_trade_duration_hours"],
                }
                extra = {
                    "threshold": threshold,
                    "n_train_rows": n_train,
                    "n_test_rows": n_test,
                    "n_features": n_features,
                    "break_even_fee_pp": be_fee,
                    "seed": 42,
                    "equity_curve": result["equity_curve"],
                    "daily_returns": result["daily_returns"],
                }
                save_results(
                    model.name, metrics_for_json, BACKTEST_RESULTS_DIR, extra=extra
                )

                all_results.append(result)
                print(
                    f"  Sharpe(BT): {result['annualized_sharpe']:.4f}  "
                    f"P&L: ${result['total_pnl']:.2f}  "
                    f"MaxDD: {result['max_drawdown']:.4f}  "
                    f"Trades: {result['num_trades']}  "
                    f"BE Fee: {be_fee:.4f}"
                )
            except Exception:
                print(f"  ERROR on {model.name}:")
                traceback.print_exc()

    # ---- Tier 3 ----
    if tier in ("3", "all"):
        print("\n" + "=" * 60)
        print("TIER 3: RL Models (seed=42)")
        print("=" * 60)

        X_train_seq, y_train = prepare_xy_for_seq(train, feature_cols)
        X_test_seq, y_test = prepare_xy_for_seq(test, feature_cols)

        # PPO-Raw
        try:
            print("\n--- PPO-Raw (seed=42) ---")
            ppo_raw = PPORawPredictor(
                random_state=42, total_timesteps=50_000
            )
            ppo_raw.fit(X_train_seq, y_train)
            predictions = ppo_raw.predict(X_test_seq)
            result = run_single_model_backtest(
                ppo_raw.name, test, predictions, backtester
            )

            be_fee = compute_break_even_fee(
                test, predictions, threshold=threshold
            )
            result["break_even_fee_pp"] = be_fee

            metrics_for_json = {
                "annualized_sharpe": result["annualized_sharpe"],
                "total_pnl": result["total_pnl"],
                "max_drawdown": result["max_drawdown"],
                "calmar_ratio": result["calmar_ratio"],
                "win_rate": result["win_rate"],
                "num_trades": result["num_trades"],
                "total_fees": result["total_fees"],
                "avg_trade_duration_hours": result["avg_trade_duration_hours"],
            }
            extra = {
                "threshold": threshold,
                "n_train_rows": n_train,
                "n_test_rows": n_test,
                "n_features": n_features,
                "break_even_fee_pp": be_fee,
                "seed": 42,
                "total_timesteps": 50_000,
                "equity_curve": result["equity_curve"],
                "daily_returns": result["daily_returns"],
            }
            save_results(
                ppo_raw.name, metrics_for_json, BACKTEST_RESULTS_DIR, extra=extra
            )

            all_results.append(result)
            print(
                f"  Sharpe(BT): {result['annualized_sharpe']:.4f}  "
                f"P&L: ${result['total_pnl']:.2f}  "
                f"MaxDD: {result['max_drawdown']:.4f}  "
                f"Trades: {result['num_trades']}  "
                f"BE Fee: {be_fee:.4f}"
            )
        except Exception:
            print("  ERROR on PPO-Raw:")
            traceback.print_exc()

        # PPO-Filtered (train autoencoder first, then PPO)
        try:
            print("\n--- PPO-Filtered (seed=42) ---")
            print("  Training autoencoder (seed=42)...")
            autoencoder = AnomalyDetectorAutoencoder(
                input_dim=n_features, random_state=42
            )
            autoencoder.fit(train[feature_cols], feature_cols)
            flagging_rate = float(
                autoencoder.flag_anomalies(train[feature_cols]).mean()
            )
            print(f"  Autoencoder flagging rate on train: {flagging_rate:.1%}")

            print("  Training PPO-Filtered...")
            ppo_filtered = PPOFilteredPredictor(
                anomaly_detector=autoencoder,
                random_state=42,
                total_timesteps=50_000,
            )
            ppo_filtered.fit(X_train_seq, y_train)
            predictions = ppo_filtered.predict(X_test_seq)
            result = run_single_model_backtest(
                ppo_filtered.name, test, predictions, backtester
            )

            be_fee = compute_break_even_fee(
                test, predictions, threshold=threshold
            )
            result["break_even_fee_pp"] = be_fee

            metrics_for_json = {
                "annualized_sharpe": result["annualized_sharpe"],
                "total_pnl": result["total_pnl"],
                "max_drawdown": result["max_drawdown"],
                "calmar_ratio": result["calmar_ratio"],
                "win_rate": result["win_rate"],
                "num_trades": result["num_trades"],
                "total_fees": result["total_fees"],
                "avg_trade_duration_hours": result["avg_trade_duration_hours"],
            }
            extra = {
                "threshold": threshold,
                "n_train_rows": n_train,
                "n_test_rows": n_test,
                "n_features": n_features,
                "break_even_fee_pp": be_fee,
                "seed": 42,
                "total_timesteps": 50_000,
                "equity_curve": result["equity_curve"],
                "daily_returns": result["daily_returns"],
            }
            save_results(
                ppo_filtered.name, metrics_for_json, BACKTEST_RESULTS_DIR, extra=extra
            )

            all_results.append(result)
            print(
                f"  Sharpe(BT): {result['annualized_sharpe']:.4f}  "
                f"P&L: ${result['total_pnl']:.2f}  "
                f"MaxDD: {result['max_drawdown']:.4f}  "
                f"Trades: {result['num_trades']}  "
                f"BE Fee: {be_fee:.4f}"
            )
        except Exception:
            print("  ERROR on PPO-Filtered:")
            traceback.print_exc()

    return all_results


def plot_equity_curves(results: list[dict], output_path: Path) -> None:
    """Plot overlaid equity curves for all backtested models.

    Tier 1: solid lines (muted colors)
    Tier 2: dashed lines (brighter colors)
    Tier 3: dotted lines
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    for r in results:
        name = r["model_name"]
        tier = _TIER_MAP.get(name, 1)
        ec = r.get("equity_curve", [])
        if not ec:
            continue

        # X = trading day index, Y = portfolio value
        values = [v for _, v in ec]
        days = list(range(len(values)))
        final_val = values[-1] if values else 10_000
        label = f"{name} (${final_val:,.0f})"

        ax.plot(
            days, values,
            color=_get_color(name),
            linestyle=_TIER_LINE_STYLES[tier],
            linewidth=1.8,
            alpha=0.85,
            label=label,
        )

    # Initial capital reference line
    ax.axhline(
        y=10_000, color="gray", linestyle="-", linewidth=0.8, alpha=0.5,
        label="Initial Capital ($10,000)",
    )

    ax.set_xlabel("Trading Day", fontsize=12)
    ax.set_ylabel("Portfolio Value ($)", fontsize=12)
    ax.set_title(
        "Walk-Forward Backtest: Equity Curves ($10k initial, 5pp round-trip costs)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"\nEquity curves saved to {output_path}")


def plot_drawdown(results: list[dict], output_path: Path) -> None:
    """Plot drawdown series for all backtested models as filled areas."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    for r in results:
        name = r["model_name"]
        tier = _TIER_MAP.get(name, 1)
        ec = r.get("equity_curve", [])
        if not ec:
            continue

        values = [v for _, v in ec]
        # Compute drawdown series
        peak = values[0]
        dd_series = []
        for v in values:
            if v > peak:
                peak = v
            dd = (v - peak) / peak  # negative values
            dd_series.append(dd)

        days = list(range(len(dd_series)))
        max_dd = min(dd_series)
        label = f"{name} (max: {max_dd:.1%})"

        ax.fill_between(
            days, dd_series, 0,
            color=_get_color(name),
            alpha=0.25,
        )
        ax.plot(
            days, dd_series,
            color=_get_color(name),
            linestyle=_TIER_LINE_STYLES[tier],
            linewidth=1.2,
            alpha=0.85,
            label=label,
        )

    ax.set_xlabel("Trading Day", fontsize=12)
    ax.set_ylabel("Drawdown (%)", fontsize=12)
    ax.set_title(
        "Walk-Forward Backtest: Drawdown",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Drawdown chart saved to {output_path}")


def _load_old_sharpe() -> dict[str, float]:
    """Load original (inflated) Sharpe from tier1/tier2/tier3 result JSONs."""
    old_sharpe: dict[str, float] = {}
    for tier_key, dir_path in OLD_RESULTS_DIRS.items():
        results = load_all_results(dir_path)
        for r in results:
            name = r["model_name"]
            sharpe = r.get("metrics", {}).get("sharpe_ratio", 0.0)
            old_sharpe[name] = sharpe
    return old_sharpe


def format_backtest_table(results: list[dict]) -> str:
    """Format a comparison table: backtested Sharpe vs old (inflated) Sharpe.

    Columns: Model | Sharpe (BT) | Old Sharpe | Total P&L | Max DD |
             Calmar | Win Rate | Trades | Fees | Break-Even Fee
    """
    old_sharpe = _load_old_sharpe()

    # Sort by fixed order
    def _sort_key(r: dict) -> int:
        name = r["model_name"]
        try:
            return _MODEL_ORDER.index(name)
        except ValueError:
            return len(_MODEL_ORDER)

    results = sorted(results, key=_sort_key)

    model_width = max(
        (len(r["model_name"]) for r in results), default=0
    )
    model_width = max(model_width, len("Model"))

    header = (
        f"{'Model':<{model_width}} | {'Sharpe(BT)':>10} | {'Old Sharpe':>10} "
        f"| {'Total P&L':>10} | {'Max DD':>8} | {'Calmar':>8} "
        f"| {'Win Rate':>8} | {'Trades':>6} | {'Fees':>8} | {'BE Fee':>8}"
    )
    separator = (
        f"{'-' * model_width}-+-{'-' * 10}-+-{'-' * 10}"
        f"-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}"
        f"-+-{'-' * 8}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 8}"
    )

    title = "====== Walk-Forward Backtest: Cross-Tier Comparison ======"

    lines = [
        title,
        "",
        "  Sharpe(BT): Annualized on daily % returns, $10k capital, 5pp round-trip fees",
        "  Old Sharpe: Inflated per-trade Sharpe from profit_sim (panel data, no fees)",
        "",
        header,
        separator,
    ]

    for r in results:
        name = r["model_name"]
        bt_sharpe = r.get("annualized_sharpe", 0.0)
        old_s = old_sharpe.get(name, 0.0)
        total_pnl = r.get("total_pnl", 0.0)
        max_dd = r.get("max_drawdown", 0.0)
        calmar = r.get("calmar_ratio", 0.0)
        win_rate = r.get("win_rate", 0.0)
        trades = r.get("num_trades", 0)
        fees = r.get("total_fees", 0.0)
        be_fee = r.get("break_even_fee_pp", 0.0)

        lines.append(
            f"{name:<{model_width}} | "
            f"{bt_sharpe:>10.4f} | "
            f"{old_s:>10.4f} | "
            f"${total_pnl:>9.2f} | "
            f"{max_dd:>8.4f} | "
            f"{calmar:>8.4f} | "
            f"{win_rate:>8.4f} | "
            f"{trades:>6d} | "
            f"${fees:>7.2f} | "
            f"{be_fee:>8.4f}"
        )

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run backtests, generate figures, print comparison table."""
    import torch
    set_all_seeds(42)

    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward backtest for all models. Produces honest, "
            "capital-normalized, fee-adjusted Sharpe ratios."
        )
    )
    parser.add_argument(
        "--tier",
        type=str,
        choices=["1", "2", "3", "all"],
        default="all",
        help="Which tier(s) to backtest (default: all)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Minimum absolute prediction for trade entry (default: 0.02)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing train.parquet and test.parquet",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("Walk-Forward Backtest: All Models")
    print("=" * 60)
    print(f"  Capital: $10,000  |  Position: $100/trade")
    print(f"  Entry cost: 3pp  |  Exit cost: 2pp  |  Threshold: {args.threshold}")
    print()

    # Run backtests
    results = run_all_backtests(
        tier=args.tier,
        threshold=args.threshold,
        data_dir=args.data_dir,
    )

    if not results:
        print("ERROR: No backtest results produced.", file=sys.stderr)
        return 1

    # Sort results by model order for consistent plotting
    def _sort_key(r: dict) -> int:
        try:
            return _MODEL_ORDER.index(r["model_name"])
        except ValueError:
            return len(_MODEL_ORDER)

    results.sort(key=_sort_key)

    # Generate figures
    plot_equity_curves(
        results, FIGURES_DIR / "backtest_equity_curves.png"
    )
    plot_drawdown(
        results, FIGURES_DIR / "backtest_drawdown.png"
    )

    # Print comparison table
    print()
    table = format_backtest_table(results)
    print(table)
    print()

    print(f"\nBacktest complete. {len(results)} models evaluated.")
    print(f"  Results: {BACKTEST_RESULTS_DIR}/")
    print(f"  Figures: {FIGURES_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
