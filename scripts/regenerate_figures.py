"""Regenerate all 11 paper-referenced figures under IEEE SciencePlots style
at 300 DPI, reading result JSONs/CSVs already on disk.

Usage:
    PYTHONPATH=$(pwd) python scripts/regenerate_figures.py

Outputs (11 PNG + sibling PDF files):
    experiments/figures/walk_forward_pnl.png
    experiments/figures/walk_forward_sharpe.png
    experiments/figures/transaction_cost_sensitivity.png
    experiments/figures/shap_bar_plot.png
    experiments/figures/backtest_equity_curves.png
    experiments/figures/bootstrap_ci_rmse.png
    experiments/figures/experiment2_lookback_pnl.png
    experiments/figures/experiment3_threshold_heatmap.png
    experiments/figures/tft_variable_importance.png
    experiments/figures/ensemble_weight_sweep.png
    experiments/results/data_scaling/pnl_at_2pp_vs_data.png

Each renderer reads its source JSON/CSV from experiments/results/... and
uses src.plotting.ieee_style.apply_ieee_style + save_ieee_fig. Figure 2
carries the SCAL-03/POL-08 cap annotation on-figure.

If a source data file is missing or malformed, the renderer raises
RuntimeError with the missing path so the user sees the specific gap.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.plotting.ieee_style import OKABE_ITO, apply_ieee_style, save_ieee_fig

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "experiments" / "figures"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require(path: Path) -> Path:
    if not path.exists():
        raise RuntimeError(f"Missing required data file: {path}")
    return path


def _load_jsonl(path: Path) -> list[dict]:
    _require(path)
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_json(path: Path) -> dict:
    _require(path)
    with path.open() as f:
        return json.load(f)


MODEL_KEYS = ["naive", "volume", "linear_regression", "xgboost", "gru", "lstm"]
MODEL_LABELS = {
    "naive": "Naive",
    "volume": "Volume",
    "linear_regression": "LR",
    "xgboost": "XGBoost",
    "gru": "GRU",
    "lstm": "LSTM",
}


# ---------------------------------------------------------------------------
# Figure 1: walk-forward P&L
# ---------------------------------------------------------------------------


def render_fig1_walk_forward_pnl() -> None:
    apply_ieee_style()
    rows = _load_jsonl(RESULTS / "walk_forward" / "log.jsonl")
    windows = [r["window_idx"] for r in rows]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for key in ["linear_regression", "xgboost", "gru", "lstm"]:
        pnl = [r["models"][key]["pnl"] for r in rows]
        ax.plot(windows, pnl, label=MODEL_LABELS[key])
    ax.set_xlabel("Walk-forward window (index)")
    ax.set_ylabel("P&L at 2 pp fees ($)")
    ax.set_title("Out-of-sample P&L (11 windows)")
    ax.legend(loc="best", frameon=False, fontsize=6)
    save_ieee_fig(fig, FIGURES / "walk_forward_pnl.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: data-scaling curve with cap annotation (POL-08 / SCAL-03)
# ---------------------------------------------------------------------------


def render_fig2_data_scaling() -> None:
    apply_ieee_style()
    rows = _load_jsonl(RESULTS / "data_scaling" / "log.jsonl")
    if not rows:
        raise RuntimeError(f"Empty data-scaling log at {RESULTS/'data_scaling'/'log.jsonl'}")

    # Keep one row per distinct training_rows value (latest run wins).
    dedup: dict[int, dict] = {}
    for r in rows:
        dedup[int(r["training_rows"])] = r
    ordered = sorted(dedup.values(), key=lambda r: r["training_rows"])
    training_rows = [r["training_rows"] for r in ordered]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for key_model, label in [
        ("linear_regression", "LR"),
        ("xgboost", "XGBoost"),
    ]:
        pnl_curve = []
        for r in ordered:
            mb = r.get("metrics_by_model", {})
            if key_model in mb:
                pnl_curve.append(mb[key_model].get("pnl_at_2pp", float("nan")))
            else:
                pnl_curve.append(float("nan"))
        ax.plot(training_rows, pnl_curve, label=label)
    # Optional sequence models if present
    for key_model, label in [("gru", "GRU"), ("lstm", "LSTM")]:
        if any(key_model in r.get("metrics_by_model", {}) for r in ordered):
            pnl_curve = [
                r.get("metrics_by_model", {}).get(key_model, {}).get("pnl_at_2pp", float("nan"))
                for r in ordered
            ]
            ax.plot(training_rows, pnl_curve, label=label)

    ax.set_xlabel("Training rows (bars/pair cap)")
    ax.set_ylabel("P&L at 2 pp fees ($)")
    ax.set_title("P&L vs training-set size — plateau at N=6,802")

    # POL-08 / SCAL-03 cap annotation (readable on-figure)
    ax.axvline(x=6802, color="red", linestyle=":", alpha=0.6, linewidth=0.9)
    ymin, ymax = ax.get_ylim()
    ax.text(
        6802,
        ymin + 0.95 * (ymax - ymin),
        " plateau at N=6,802\n fixed pair universe",
        fontsize=6,
        color="red",
        ha="left",
        va="top",
    )

    ax.legend(loc="best", frameon=False, fontsize=6)
    save_ieee_fig(fig, RESULTS / "data_scaling" / "pnl_at_2pp_vs_data.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: walk-forward Sharpe trajectory
# ---------------------------------------------------------------------------


def render_fig3_walk_forward_sharpe() -> None:
    apply_ieee_style()
    rows = _load_jsonl(RESULTS / "walk_forward" / "log.jsonl")
    windows = [r["window_idx"] for r in rows]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for key in ["linear_regression", "xgboost", "gru", "lstm"]:
        sharpe = [r["models"][key].get("sharpe_per_trade", float("nan")) for r in rows]
        ax.plot(windows, sharpe, label=MODEL_LABELS[key])
    ax.set_xlabel("Walk-forward window (index)")
    ax.set_ylabel("Per-trade Sharpe ratio")
    ax.set_title("Walk-forward Sharpe trajectory")
    ax.legend(loc="best", frameon=False, fontsize=6)
    save_ieee_fig(fig, FIGURES / "walk_forward_sharpe.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: transaction cost sensitivity
# ---------------------------------------------------------------------------


def render_fig4_transaction_cost() -> None:
    apply_ieee_style()
    data = _load_json(RESULTS / "transaction_costs" / "sensitivity_results.json")
    cost_pp = [c * 100.0 for c in data["cost_levels"]]  # fractional → pp

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for model_name in ["Linear Regression", "XGBoost", "Naive (Spread Closes)", "Volume (Higher Volume Correct)"]:
        if model_name in data["models"]:
            y = data["models"][model_name]["net_pnl_by_cost"]
            ax.plot(cost_pp, y, label=model_name.split(" ")[0])
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Round-trip fee (pp)")
    ax.set_ylabel("P&L ($)")
    ax.set_title("Transaction-cost sensitivity")
    ax.legend(loc="best", frameon=False, fontsize=6)
    save_ieee_fig(fig, FIGURES / "transaction_cost_sensitivity.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: SHAP bar plot
# ---------------------------------------------------------------------------


def render_fig5_shap_bar() -> None:
    apply_ieee_style()
    shap_path = RESULTS / "shap" / "xgboost_feature_importance.csv"
    _require(shap_path)
    df = pd.read_csv(shap_path)
    df = df.sort_values("mean_abs_shap", ascending=False).head(20)
    df = df.iloc[::-1]  # reverse so highest bar is at top

    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.barh(range(len(df)), df["mean_abs_shap"].values, color=OKABE_ITO[5])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature_name"].values, fontsize=5)
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_ylabel("Feature")
    ax.set_title("Feature importance (top 20)")
    save_ieee_fig(fig, FIGURES / "shap_bar_plot.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 6: equity curves
# ---------------------------------------------------------------------------


def render_fig6_equity_curves() -> None:
    apply_ieee_style()
    backtest_dir = RESULTS / "backtest"
    if not backtest_dir.exists():
        raise RuntimeError(f"Missing directory: {backtest_dir}")

    model_files = {
        "LR": "linear_regression.json",
        "XGBoost": "xgboost.json",
        "GRU": "gru.json",
        "LSTM": "lstm.json",
    }

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for label, fname in model_files.items():
        path = backtest_dir / fname
        if not path.exists():
            continue
        data = _load_json(path)
        eq = data.get("equity_curve")
        if not eq:
            continue
        xs = [p[0] for p in eq]
        ys = [p[1] for p in eq]
        ax.plot(xs, ys, label=label)
    ax.set_xlabel("Bar index")
    ax.set_ylabel("Equity ($)")
    ax.set_title("Equity curves by model")
    ax.legend(loc="best", frameon=False, fontsize=6)
    save_ieee_fig(fig, FIGURES / "backtest_equity_curves.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: bootstrap 95% CI on RMSE
# ---------------------------------------------------------------------------


def render_fig7_bootstrap_rmse() -> None:
    apply_ieee_style()
    data = _load_json(RESULTS / "bootstrap_ci" / "bootstrap_results.json")
    models = list(data["models"].keys())
    short_labels = [m.split(" ")[0] for m in models]
    means = [data["models"][m]["rmse"]["mean"] for m in models]
    lowers = [data["models"][m]["rmse"]["ci_lower"] for m in models]
    uppers = [data["models"][m]["rmse"]["ci_upper"] for m in models]
    err_low = [mean - lo for mean, lo in zip(means, lowers)]
    err_hi = [hi - mean for mean, hi in zip(means, uppers)]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(models))
    ax.errorbar(x, means, yerr=[err_low, err_hi], fmt="o", capsize=3, color=OKABE_ITO[5])
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=6)
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE (spread points)")
    ax.set_title("Bootstrap 95% CI on RMSE")
    save_ieee_fig(fig, FIGURES / "bootstrap_ci_rmse.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: lookback window sensitivity
# ---------------------------------------------------------------------------


def render_fig8_lookback_pnl() -> None:
    apply_ieee_style()
    lookback_dir = RESULTS / "ablation_lookback"
    _require(lookback_dir)
    hours_map = {2: 8, 6: 24, 12: 48, 18: 72}  # bars → hours

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for model_name, label in [("gru", "GRU"), ("lstm", "LSTM")]:
        hours, pnls = [], []
        for bars, hrs in hours_map.items():
            path = lookback_dir / f"{model_name}_lookback_{bars}.json"
            if not path.exists():
                continue
            d = _load_json(path)
            hours.append(hrs)
            pnls.append(d["metrics"].get("total_pnl", float("nan")))
        if hours:
            ax.plot(hours, pnls, label=label)
    ax.set_xlabel("Lookback window (hours)")
    ax.set_ylabel("P&L ($)")
    ax.set_title("P&L vs lookback window")
    ax.legend(loc="best", frameon=False, fontsize=6)
    save_ieee_fig(fig, FIGURES / "experiment2_lookback_pnl.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9: threshold x model heatmap (double-column)
# ---------------------------------------------------------------------------


def render_fig9_threshold_heatmap() -> None:
    apply_ieee_style()
    thr_dir = RESULTS / "ablation_threshold"
    _require(thr_dir)

    models = [
        ("linear_regression", "LR"),
        ("xgboost", "XGBoost"),
        ("gru", "GRU"),
        ("lstm", "LSTM"),
        ("naive_spread_closes", "Naive"),
        ("volume_higher_volume_correct", "Volume"),
        ("ppo_raw", "PPO"),
        ("ppo_filtered", "PPO+AE"),
    ]
    thresholds = [0.00, 0.02, 0.05, 0.10]

    matrix = np.full((len(models), len(thresholds)), np.nan)
    for i, (key, _lbl) in enumerate(models):
        for j, t in enumerate(thresholds):
            path = thr_dir / f"{key}_threshold_{t:.2f}.json"
            if path.exists():
                d = _load_json(path)
                matrix[i, j] = d["metrics"].get("total_pnl", float("nan"))

    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    im = ax.imshow(matrix, cmap="cividis", aspect="auto")
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"{int(t*100)}" for t in thresholds])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([lbl for _, lbl in models])
    ax.set_xlabel("Minimum spread threshold (pp)")
    ax.set_ylabel("Model")
    ax.set_title("P&L ($) by model x threshold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P&L ($)")
    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                txt = "—"
            else:
                txt = f"{val:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="white")
    save_ieee_fig(fig, FIGURES / "experiment3_threshold_heatmap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10: TFT VSN encoder importances
# ---------------------------------------------------------------------------


def render_fig10_tft_vsn() -> None:
    apply_ieee_style()
    vsn_path = RESULTS / "tft" / "vsn_importance.json"
    _require(vsn_path)
    d = _load_json(vsn_path)
    names = d["features"][:15]
    weights = d["weights"][:15]
    # Reverse so the largest bar sits at the top of the figure
    names = list(reversed(names))
    weights = list(reversed(weights))

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.barh(range(len(names)), weights, color=OKABE_ITO[5])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=5)
    ax.set_xlabel("VSN variable-selection weight")
    ax.set_ylabel("Feature")
    ax.set_title("TFT variable-selection weights (top 15)")
    save_ieee_fig(fig, FIGURES / "tft_variable_importance.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 11: ensemble weight sweep
# ---------------------------------------------------------------------------


def render_fig11_ensemble_sweep() -> None:
    apply_ieee_style()
    d = _load_json(RESULTS / "ensemble" / "summary.json")
    sweep = d.get("weight_sweep")
    if not sweep:
        raise RuntimeError("ensemble/summary.json missing 'weight_sweep' array")
    lr_w = [p["lr_weight"] for p in sweep]
    pnl_f = [p["pnl_filtered"] for p in sweep]
    pnl_uf = [p["pnl_unfiltered"] for p in sweep]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(lr_w, pnl_f, label="Filtered (concordance)")
    ax.plot(lr_w, pnl_uf, label="Unfiltered")
    ax.set_xlabel("LR weight (1 - XGBoost weight)")
    ax.set_ylabel("P&L at 2 pp fees ($)")
    ax.set_title("Ensemble weight-sensitivity sweep")
    ax.legend(loc="best", frameon=False, fontsize=6)
    save_ieee_fig(fig, FIGURES / "ensemble_weight_sweep.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    renderers = [
        render_fig1_walk_forward_pnl,
        render_fig2_data_scaling,
        render_fig3_walk_forward_sharpe,
        render_fig4_transaction_cost,
        render_fig5_shap_bar,
        render_fig6_equity_curves,
        render_fig7_bootstrap_rmse,
        render_fig8_lookback_pnl,
        render_fig9_threshold_heatmap,
        render_fig10_tft_vsn,
        render_fig11_ensemble_sweep,
    ]
    for fn in renderers:
        try:
            fn()
            print(f"OK  {fn.__name__}")
        except Exception as e:
            print(f"ERR {fn.__name__}: {e}")
            raise


if __name__ == "__main__":
    main()
