"""Transaction cost sensitivity analysis across all model tiers.

Varies per-trade transaction cost from 0 to 10 percentage points (in 0.5pp
steps) and computes net P&L for each model. Reports break-even cost per model
(the per-trade fee that zeroes out gross P&L).

Outputs:
  - Sensitivity line plot (experiments/figures/transaction_cost_sensitivity.png)
  - Results JSON (experiments/results/transaction_costs/sensitivity_results.json)

Run:
    python -m experiments.run_transaction_costs
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.evaluation.results_store import load_all_results
from experiments.run_baselines import _MODEL_ORDER


FIGURES_DIR = Path("experiments/figures")
TXCOST_RESULTS_DIR = Path("experiments/results/transaction_costs")

TIER1_DIR = Path("experiments/results/tier1")
TIER2_DIR = Path("experiments/results/tier2")
TIER3_DIR = Path("experiments/results/tier3")

# 0 to 10pp in 0.5pp steps (0.005 = 0.5 percentage points)
COST_LEVELS = np.arange(0.0, 0.105, 0.005)

# Tier color mapping for visual distinction in plot
TIER_COLORS = {
    "Naive (Spread Closes)": "#7f7f7f",         # gray
    "Volume (Higher Volume Correct)": "#bcbd22", # olive
    "Linear Regression": "#1f77b4",              # blue
    "XGBoost": "#ff7f0e",                        # orange
    "GRU": "#2ca02c",                            # green
    "LSTM": "#d62728",                           # red
    "PPO-Raw": "#9467bd",                        # purple
    "PPO-Filtered": "#e377c2",                   # pink
}

TIER_LINESTYLES = {
    "Naive (Spread Closes)": ":",
    "Volume (Higher Volume Correct)": ":",
    "Linear Regression": "-",
    "XGBoost": "-",
    "GRU": "--",
    "LSTM": "--",
    "PPO-Raw": "-.",
    "PPO-Filtered": "-.",
}


def main() -> int:
    """Run transaction cost sensitivity analysis."""
    # --- Load all results ---
    all_results = (
        load_all_results(TIER1_DIR)
        + load_all_results(TIER2_DIR)
        + load_all_results(TIER3_DIR)
    )

    if not all_results:
        print("Error: no results found. Run experiments first.")
        return 1

    # --- Build per-model sensitivity data ---
    models_data: dict[str, dict] = {}

    for r in all_results:
        # Tolerate alternate schema: TFT.json uses a flat layout with
        # "model" + top-level metric fields, while every other result JSON
        # uses "model_name" + a nested "metrics" dict.
        name = r.get("model_name") or r.get("model")
        if name is None:
            continue
        m = r.get("metrics", r)
        if "total_pnl" not in m or "num_trades" not in m:
            continue
        total_pnl = float(m["total_pnl"])
        num_trades = int(m["num_trades"])

        # Break-even: per-trade cost that makes net P&L = 0
        if num_trades > 0:
            break_even = total_pnl / num_trades
        else:
            break_even = 0.0

        # Net P&L at each cost level
        net_pnl_by_cost = [
            total_pnl - (num_trades * float(cost))
            for cost in COST_LEVELS
        ]

        models_data[name] = {
            "total_pnl": total_pnl,
            "num_trades": num_trades,
            "break_even_cost": break_even,
            "net_pnl_by_cost": net_pnl_by_cost,
        }

    # --- Save results JSON ---
    TXCOST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_payload = {
        "cost_levels": [round(float(c), 4) for c in COST_LEVELS],
        "models": models_data,
    }
    json_path = TXCOST_RESULTS_DIR / "sensitivity_results.json"
    with open(json_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"Saved: {json_path}")

    # --- Produce sensitivity line plot ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot in _MODEL_ORDER for consistent legend
    cost_pct = COST_LEVELS * 100  # convert to percentage points for axis

    for name in _MODEL_ORDER:
        if name not in models_data:
            continue
        md = models_data[name]
        color = TIER_COLORS.get(name, "#333333")
        ls = TIER_LINESTYLES.get(name, "-")
        ax.plot(
            cost_pct,
            md["net_pnl_by_cost"],
            label=name,
            color=color,
            linestyle=ls,
            linewidth=2,
        )

        # Annotate break-even point
        be = md["break_even_cost"]
        if 0 < be <= 0.10:  # only annotate if within chart range
            be_pct = be * 100
            ax.plot(be_pct, 0, "o", color=color, markersize=6, zorder=5)

    # Reference lines
    ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(x=5.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="~Kalshi taker fee (5pp)")
    ax.axvline(x=7.0, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="~High fee scenario (7pp)")

    ax.set_xlabel("Transaction Cost (percentage points per trade)", fontsize=12)
    ax.set_ylabel("Net P&L", fontsize=12)
    ax.set_title("Transaction Cost Sensitivity: Net P&L vs Per-Trade Cost", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = FIGURES_DIR / "transaction_cost_sensitivity.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")

    # --- Print summary table ---
    print()
    header = (
        f"{'Model':<35s} | {'Gross P&L':>9s} | {'Trades':>6s} "
        f"| {'Break-Even':>12s} | {'Net @5pp':>10s} | {'Net @7pp':>10s}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for name in _MODEL_ORDER:
        if name not in models_data:
            continue
        md = models_data[name]
        total_pnl = md["total_pnl"]
        num_trades = md["num_trades"]
        be = md["break_even_cost"]

        # Net P&L at 5pp (index 10: 0.05) and 7pp (index 14: 0.07)
        net_5pp = total_pnl - (num_trades * 0.05)
        net_7pp = total_pnl - (num_trades * 0.07)

        print(
            f"{name:<35s} | {total_pnl:>9.2f} | {num_trades:>6d} "
            f"| {be * 100:>9.4f} pp | {net_5pp:>10.2f} | {net_7pp:>10.2f}"
        )

    print(sep)
    print()
    print("Break-even cost = gross P&L / num_trades (per-trade fee that zeroes out P&L)")
    print("Kalshi taker fee ~5-7pp. Models profitable above break-even survive real fees.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
