"""Experiment 1: Cross-tier complexity-vs-performance comparison.

Loads existing per-model JSON results from tier1/, tier2/, tier3/ and
produces publication-quality visualizations and a LaTeX table.

NO model retraining -- purely a post-hoc synthesis of existing results.

Outputs:
  - experiments/results/experiment1/cross_tier_summary.json
  - experiments/figures/experiment1_rmse_bar.png
  - experiments/figures/experiment1_pnl_curves.png
  - experiments/figures/experiment1_comparison_table.tex

Run:
    python -m experiments.run_experiment1_comparison
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.evaluation.results_store import load_all_results  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402

# ---- Paths ----
RESULTS_TIER1 = Path("experiments/results/tier1")
RESULTS_TIER2 = Path("experiments/results/tier2")
RESULTS_TIER3 = Path("experiments/results/tier3")
SUMMARY_DIR = Path("experiments/results/experiment1")
FIGURES_DIR = Path("experiments/figures")

# ---- Display configuration ----
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
    1: "#4C72B0",   # blue
    2: "#DD8452",   # orange
    3: "#C44E52",   # red
}

_TIER_LABELS = {
    1: "Tier 1 (Regression)",
    2: "Tier 2 (Time Series)",
    3: "Tier 3 (RL)",
}


def _sort_key(result: dict) -> int:
    """Sort by fixed model display order."""
    name = result["model_name"]
    try:
        return _MODEL_ORDER.index(name)
    except ValueError:
        return len(_MODEL_ORDER)


def load_all_tier_results() -> list[dict]:
    """Load results from all 3 tier directories and combine."""
    tier1 = load_all_results(RESULTS_TIER1)
    tier2 = load_all_results(RESULTS_TIER2)
    tier3 = load_all_results(RESULTS_TIER3)
    combined = tier1 + tier2 + tier3
    combined.sort(key=_sort_key)
    return combined


def produce_summary_json(results: list[dict]) -> Path:
    """Write consolidated cross-tier summary JSON."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    # Rank by RMSE (lower is better)
    by_rmse = sorted(results, key=lambda r: r["metrics"]["rmse"])
    rmse_rank = {r["model_name"]: rank + 1 for rank, r in enumerate(by_rmse)}

    entries = []
    for r in results:
        name = r["model_name"]
        tier = _TIER_MAP.get(name, 0)
        m = r["metrics"]
        entry = {
            "model_name": name,
            "tier": tier,
            "rmse_rank": rmse_rank[name],
            "metrics": {
                "rmse": m["rmse"],
                "mae": m["mae"],
                "directional_accuracy": m["directional_accuracy"],
                "total_pnl": m["total_pnl"],
                "num_trades": m["num_trades"],
                "win_rate": m["win_rate"],
                "sharpe_ratio": m["sharpe_ratio"],
                "sharpe_per_trade": m.get("sharpe_per_trade", 0.0),
            },
        }
        # Add seed statistics for Tier 2/3
        if "mean_rmse" in r:
            entry["mean_rmse"] = r["mean_rmse"]
            entry["std_rmse"] = r["std_rmse"]
            entry["seeds"] = r.get("seeds", [])
            entry["seed_rmses"] = r.get("seed_rmses", [])
        entries.append(entry)

    summary = {
        "experiment": "Experiment 1: Cross-Tier Complexity vs Performance",
        "dataset": {
            "n_train": 6802,
            "n_test": 1673,
            "n_features": 31,
            "threshold": 0.02,
        },
        "models": entries,
    }

    path = SUMMARY_DIR / "cross_tier_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to {path}")
    return path


def produce_rmse_bar_chart(results: list[dict]) -> Path:
    """Create horizontal bar chart of RMSE by model, color-coded by tier."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Reverse for horizontal bar chart (top model at top)
    ordered = list(reversed(results))

    names = [r["model_name"] for r in ordered]
    rmses = [r["metrics"]["rmse"] for r in ordered]
    colors = [_TIER_COLORS[_TIER_MAP[n]] for n in names]

    # Error bars: only for Tier 2/3
    xerr = []
    for r in ordered:
        if "std_rmse" in r:
            xerr.append(r["std_rmse"])
        else:
            xerr.append(0.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(names))
    bars = ax.barh(
        y_pos, rmses, xerr=xerr, color=colors, edgecolor="white",
        capsize=4, height=0.6,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("RMSE (lower is better)", fontsize=12)
    ax.set_title(
        "Experiment 1: RMSE by Model (Cross-Tier Comparison)",
        fontsize=14, fontweight="bold",
    )
    ax.invert_yaxis()  # Not needed since we reversed, but ensures top-to-bottom

    # Add value labels on bars
    for bar, rmse, err in zip(bars, rmses, xerr):
        label = f"{rmse:.4f}"
        if err > 0:
            label += f" +/-{err:.4f}"
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=9,
        )

    # Legend for tier colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_TIER_COLORS[t], label=_TIER_LABELS[t])
        for t in [1, 2, 3]
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()
    path = FIGURES_DIR / "experiment1_rmse_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"RMSE bar chart saved to {path}")
    return path


def produce_pnl_curves(results: list[dict]) -> Path:
    """Create overlaid equity curves for all 8 models."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    # Line styles per tier to aid distinction
    linestyles = {1: "-", 2: "--", 3: ":"}

    for r in results:
        name = r["model_name"]
        tier = _TIER_MAP[name]
        pnl = r.get("pnl_series", [])
        if not pnl:
            continue
        color = _TIER_COLORS[tier]
        final_pnl = pnl[-1] if pnl else 0.0
        label = f"{name} (P&L={final_pnl:.1f})"
        ax.plot(
            range(len(pnl)), pnl,
            color=color, linestyle=linestyles[tier],
            linewidth=1.5, alpha=0.85, label=label,
        )

    ax.set_xlabel("Bar Index (4h intervals)", fontsize=12)
    ax.set_ylabel("Cumulative P&L (percentage points)", fontsize=12)
    ax.set_title(
        "Experiment 1: Cumulative P&L Equity Curves",
        fontsize=14, fontweight="bold",
    )
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "experiment1_pnl_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"P&L equity curves saved to {path}")
    return path


def produce_latex_table(results: list[dict]) -> Path:
    """Generate a LaTeX table ready for the Phase 8 paper."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Collect metrics for bolding best values
    all_metrics: dict[str, list[float]] = {
        "rmse": [], "mae": [], "dir_acc": [],
        "pnl": [], "trades": [], "win_rate": [], "sharpe": [],
    }
    for r in results:
        m = r["metrics"]
        all_metrics["rmse"].append(m["rmse"])
        all_metrics["mae"].append(m["mae"])
        all_metrics["dir_acc"].append(m["directional_accuracy"])
        all_metrics["pnl"].append(m["total_pnl"])
        all_metrics["trades"].append(m["num_trades"])
        all_metrics["win_rate"].append(m["win_rate"])
        all_metrics["sharpe"].append(m["sharpe_ratio"])

    # Best: lowest RMSE/MAE, highest for the rest
    best = {
        "rmse": min(all_metrics["rmse"]),
        "mae": min(all_metrics["mae"]),
        "dir_acc": max(all_metrics["dir_acc"]),
        "pnl": max(all_metrics["pnl"]),
        "trades": max(all_metrics["trades"]),
        "win_rate": max(all_metrics["win_rate"]),
        "sharpe": max(all_metrics["sharpe"]),
    }

    def _bold_if_best(val: float, key: str, fmt: str) -> str:
        formatted = f"{val:{fmt}}"
        if abs(val - best[key]) < 1e-9:
            return f"\\textbf{{{formatted}}}"
        return formatted

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Experiment 1: Cross-Tier Model Comparison. "
        "Best value in each column is bolded. "
        "RMSE and MAE are lower-is-better; all others are higher-is-better.}",
        "\\label{tab:experiment1}",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Model & Tier & RMSE & MAE & Dir Acc & P\\&L & Trades & Win Rate & Sharpe \\\\",
        "\\midrule",
    ]

    prev_tier = None
    for r in results:
        name = r["model_name"]
        tier = _TIER_MAP[name]
        m = r["metrics"]

        # Add horizontal rule between tiers
        if prev_tier is not None and tier != prev_tier:
            lines.append("\\midrule")
        prev_tier = tier

        # Escape special LaTeX characters in model name
        latex_name = name.replace("&", "\\&")

        row = (
            f"{latex_name} & {tier} & "
            f"{_bold_if_best(m['rmse'], 'rmse', '.4f')} & "
            f"{_bold_if_best(m['mae'], 'mae', '.4f')} & "
            f"{_bold_if_best(m['directional_accuracy'], 'dir_acc', '.4f')} & "
            f"{_bold_if_best(m['total_pnl'], 'pnl', '.1f')} & "
            f"{_bold_if_best(m['num_trades'], 'trades', 'd')} & "
            f"{_bold_if_best(m['win_rate'], 'win_rate', '.4f')} & "
            f"{_bold_if_best(m['sharpe_ratio'], 'sharpe', '.2f')} \\\\"
        )
        lines.append(row)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    path = FIGURES_DIR / "experiment1_comparison_table.tex"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"LaTeX table saved to {path}")
    return path


def format_text_table(results: list[dict]) -> str:
    """Render a fixed-width comparison table (same style as run_baselines.py)."""
    model_width = max(
        (len(r["model_name"]) for r in results), default=0
    )
    model_width = max(model_width, len("Model"))

    header = (
        f"{'Model':<{model_width}} | {'RMSE':>7} | {'MAE':>7} "
        f"| {'Dir Acc':>7} | {'P&L':>8} | {'Trades':>6} "
        f"| {'Win Rate':>8} | {'Sharpe':>7} | {'Raw SR':>7}"
    )
    separator = (
        f"{'-' * model_width}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}-+-"
        f"{'-' * 8}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 7}-+-{'-' * 7}"
    )

    title = "====== Experiment 1: Full Cross-Tier Comparison ======"

    lines = [
        title,
        "",
        "  Sharpe: annualized time-series Sharpe (4h bars, 24/7, sqrt(2190))",
        "  Raw SR: unannualized per-trade mean/std (raw trade quality)",
        "",
        header,
        separator,
    ]

    for r in results:
        m = r["metrics"]

        rmse_val = m["rmse"]
        rmse_str = f"{rmse_val:>7.4f}"
        seed_note = ""
        if "mean_rmse" in r and "std_rmse" in r:
            seed_note = f" (+/-{r['std_rmse']:.4f})"

        lines.append(
            f"{r['model_name']:<{model_width}} | "
            f"{rmse_str}{seed_note} | "
            f"{m['mae']:>7.4f} | "
            f"{m['directional_accuracy']:>7.4f} | "
            f"{m['total_pnl']:>8.4f} | "
            f"{m['num_trades']:>6d} | "
            f"{m['win_rate']:>8.4f} | "
            f"{m['sharpe_ratio']:>7.4f} | "
            f"{m.get('sharpe_per_trade', 0.0):>7.4f}"
        )

    return "\n".join(lines)


def main() -> int:
    """Load all results, produce summary JSON, figures, and LaTeX table."""
    set_all_seeds(42)
    print("=" * 60)
    print("Experiment 1: Cross-Tier Complexity vs Performance")
    print("=" * 60)
    print()

    results = load_all_tier_results()
    if len(results) != 8:
        print(
            f"WARNING: Expected 8 models, found {len(results)}. "
            f"Models: {[r['model_name'] for r in results]}"
        )

    print(f"Loaded {len(results)} model results.")
    print()

    # 1. Summary JSON
    produce_summary_json(results)

    # 2. RMSE bar chart
    produce_rmse_bar_chart(results)

    # 3. P&L equity curves
    produce_pnl_curves(results)

    # 4. LaTeX table
    produce_latex_table(results)

    # 5. Print text table
    print()
    print(format_text_table(results))
    print()

    print("Experiment 1 complete. All outputs saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
