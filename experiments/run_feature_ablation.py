"""LOGO Feature Ablation — Phase 12.
Pre-registration: .planning/ablation_protocol.md (committed before this file).
Runs 12 configs: {LR, XGBoost} x {baseline, drop_A..E}. Usage: [--dry-run]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.verify_headline import TARGET, build, feature_cols, simulate_pnl  # noqa: E402
from src.models.linear_regression import LinearRegressionPredictor  # noqa: E402
from src.models.xgboost_model import XGBoostPredictor  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402

FEATURE_GROUPS: dict[str, list[str]] = {
    "A": [
        "kalshi_vwap", "kalshi_open", "kalshi_high", "kalshi_low", "kalshi_close",
        "kalshi_volume", "kalshi_trade_count",
        "polymarket_vwap", "polymarket_open", "polymarket_high", "polymarket_low",
        "polymarket_close", "polymarket_volume", "polymarket_trade_count",
        "polymarket_dollar_volume",
    ],
    "B": [
        "spread", "mid_price", "volume_ratio", "dollar_volume_ratio",
        "price_divergence_pct", "spread_range", "trade_count_ratio",
        "price_velocity", "boundary_distance", "kalshi_dollar_volume",
    ],
    "C": [
        "spread_momentum", "spread_momentum_6", "spread_momentum_12",
        "spread_volatility", "spread_volatility_6", "spread_zscore",
    ],
    "D": [
        "polymarket_realized_spread", "kalshi_amihud", "polymarket_amihud",
        "kalshi_kyle_lambda", "polymarket_kyle_lambda",
        "kalshi_roll_spread", "polymarket_roll_spread",
        "kalshi_cs_spread", "polymarket_cs_spread",
        "kalshi_hl_vol", "polymarket_hl_vol",
        "polymarket_order_flow_imbalance", "ofi_differential",
    ],
    "E": [
        "longshot_score",
        "kalshi_max_trade_size", "polymarket_max_trade_size",
        "kalshi_hours_since_last_trade", "polymarket_hours_since_last_trade",
        "polymarket_buy_volume", "polymarket_sell_volume",
    ],
}

MODELS = {"LR": LinearRegressionPredictor, "XGBoost": XGBoostPredictor}
RESULTS_DIR = Path(__file__).parent / "results" / "ablation"


def temporal_split(df: pd.DataFrame, train_frac: float = 0.85):
    """Split df chronologically (must be pre-sorted)."""
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def bootstrap_delta_pnl(
    preds_config: np.ndarray,
    preds_baseline: np.ndarray,
    actuals: np.ndarray,
    n_boot: int = 1000,
    fee: float = 0.02,
    rng: np.random.Generator | None = None,
):
    """Paired bootstrap 95% CI for delta-P&L (config minus baseline)."""
    rng = rng or np.random.default_rng(42)
    n = len(actuals)
    deltas = [
        simulate_pnl(preds_config[idx := rng.integers(0, n, n)], actuals[idx], fee)["pnl"]
        - simulate_pnl(preds_baseline[idx], actuals[idx], fee)["pnl"]
        for _ in range(n_boot)
    ]
    arr = np.array(deltas)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), arr


def dry_run(train_df: pd.DataFrame, test_df: pd.DataFrame, feats: list[str]) -> None:
    """Validate feature groups and print split sizes."""
    all_feats = [f for g in FEATURE_GROUPS.values() for f in g]
    assert len(all_feats) == len(set(all_feats)), "Feature overlap detected"
    missing = [f for f in all_feats if f not in feats]
    not_grouped = [f for f in feats if f not in all_feats]
    tp, ah = temporal_split(train_df)
    print(f"Dry-run validation:")
    print(f"  Total features: {len(feats)} (expected 51)")
    print(f"  Group feature total: {len(all_feats)} (expected 51)")
    print(f"  Missing from actual features (in protocol but not in data): {missing}")
    print(f"  In data but not in any group: {not_grouped}")
    print(f"  train_proper rows: {len(tp)}")
    print(f"  ablation_holdout rows: {len(ah)}")
    print(f"  final_test rows: {len(test_df)}")
    assert len(feats) == 51, f"Expected 51 features, got {len(feats)}"
    assert not missing, f"Protocol features missing from data: {missing}"
    print("  OK — dry-run passed.")


def _classify(ci_lo: float, ci_hi: float, delta: float, da_drop: float) -> str:
    if ci_hi < 0 and abs(delta) > 10 and da_drop > 2.0:
        return "load-bearing"
    if ci_lo < 0 < ci_hi and abs(delta) > 10:
        return "inconclusive"
    return "droppable"


def _metrics(preds: np.ndarray, y: np.ndarray) -> tuple[float, float, dict]:
    rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
    mask = y != 0
    da = float(np.mean(np.sign(preds[mask]) == np.sign(y[mask]))) if mask.sum() else 0.0
    return rmse, da, simulate_pnl(preds, y, fee=0.02)


def _write_report(configs: list[dict], tp: int, ah: int, ft: int) -> None:
    LABELS = {
        "none": "— (baseline)", "A": "A — Raw OHLCV", "B": "B — Cross-platform",
        "C": "C — Rolling/momentum", "D": "D — Microstructure", "E": "E — Pred-market",
    }
    rows = [
        "# LOGO Feature Ablation — Results Report", "",
        f"**Split:** train_proper={tp}, ablation_holdout={ah}, final_test={ft}", "",
        "**Note:** P&L/RMSE on `ablation_holdout` (selection metric). 1,000 bootstrap resamples.",
        "",
        "| Model | Dropped Group | # Features | P&L @ 2pp | ΔP&L | 95% CI of ΔP&L | RMSE | Dir. Acc. | Classification |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in configs:
        g = c["dropped_group"]
        ci = "—" if g == "none" else f"[{c['ci_lower']:+.2f}, {c['ci_upper']:+.2f}]"
        dp = "$0.00" if g == "none" else f"${c['delta_pnl']:+.2f}"
        rows.append(
            f"| {c['model']} | {LABELS[g]} | {c['feature_count']} | ${c['pnl']:+.2f} | "
            f"{dp} | {ci} | {c['rmse']:.4f} | {c['directional_accuracy']:.1f}% | {c['classification']} |"
        )
    rows += [
        "", "## Classification Key",
        "- **load-bearing**: 95% CI < 0, |ΔP&L| > $10, Dir.Acc. drop > 2pp",
        "- **inconclusive**: CI straddles zero with |ΔP&L| > $10",
        "- **droppable**: CI straddles zero with |ΔP&L| <= $10",
        "- **baseline**: all 51 features (reference)",
    ]
    (RESULTS_DIR / "report.md").write_text("\n".join(rows) + "\n")


def main() -> int:
    set_all_seeds(42)
    data = Path("data/processed")
    train_df = build(pd.read_parquet(data / "train.parquet"))
    test_df = build(pd.read_parquet(data / "test.parquet"))
    feats = feature_cols(train_df)

    if "--dry-run" in sys.argv:
        dry_run(train_df, test_df, feats)
        return 0

    train_proper, ablation_holdout = temporal_split(train_df, train_frac=0.85)
    y_train = train_proper[TARGET].to_numpy()
    y_hold = ablation_holdout[TARGET].to_numpy()
    print(f"Split — train_proper: {len(train_proper)}, ablation_holdout: {len(ablation_holdout)}, final_test: {len(test_df)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    configs: list[dict] = []
    base_preds: dict[str, np.ndarray] = {}
    base_pnl: dict[str, float] = {}
    base_da: dict[str, float] = {}
    rng = np.random.default_rng(42)
    boot_arrays: dict[str, np.ndarray] = {}

    # Baselines (all 51 features) — must run first to compute deltas
    for mname, Cls in MODELS.items():
        m = Cls(); m.fit(train_proper[feats], y_train)
        preds = m.predict(ablation_holdout[feats])
        base_preds[mname] = preds
        rmse, da, trade = _metrics(preds, y_hold)
        base_pnl[mname] = trade["pnl"]; base_da[mname] = da
        configs.append({"model": mname, "dropped_group": "none", "feature_count": len(feats),
                        "pnl": round(trade["pnl"], 4), "delta_pnl": 0.0,
                        "rmse": round(rmse, 6), "directional_accuracy": round(da * 100, 2),
                        "ci_lower": 0.0, "ci_upper": 0.0, "num_trades": trade["num_trades"],
                        "num_bootstrap": 1000, "classification": "baseline"})
        print(f"  {mname} baseline: P&L={trade['pnl']:+.2f} DA={da:.4f} RMSE={rmse:.4f}")

    # LOGO: drop each group
    for mname, Cls in MODELS.items():
        for g in ("A", "B", "C", "D", "E"):
            af = [f for f in feats if f not in set(FEATURE_GROUPS[g])]
            m = Cls(); m.fit(train_proper[af], y_train)
            preds = m.predict(ablation_holdout[af])
            rmse, da, trade = _metrics(preds, y_hold)
            delta = trade["pnl"] - base_pnl[mname]
            da_drop = (base_da[mname] - da) * 100
            ci_lo, ci_hi, arr = bootstrap_delta_pnl(preds, base_preds[mname], y_hold, rng=rng)
            boot_arrays[f"{mname}_drop{g}"] = arr
            cls = _classify(ci_lo, ci_hi, delta, da_drop)
            configs.append({"model": mname, "dropped_group": g, "feature_count": len(af),
                            "pnl": round(trade["pnl"], 4), "delta_pnl": round(delta, 4),
                            "rmse": round(rmse, 6), "directional_accuracy": round(da * 100, 2),
                            "ci_lower": round(ci_lo, 4), "ci_upper": round(ci_hi, 4),
                            "num_trades": trade["num_trades"], "num_bootstrap": 1000,
                            "classification": cls})
            print(f"  {mname} drop_{g} ({len(af)} feats): P&L={trade['pnl']:+.2f} (Δ={delta:+.2f} CI=[{ci_lo:+.2f},{ci_hi:+.2f}]) → {cls}")

    summary = {"train_proper_rows": len(train_proper), "ablation_holdout_rows": len(ablation_holdout),
                "final_test_rows": len(test_df), "configs": configs}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(configs).to_csv(RESULTS_DIR / "per_config.csv", index=False)
    np.savez(str(RESULTS_DIR / "bootstrap_distributions.npz"), **boot_arrays)
    _write_report(configs, len(train_proper), len(ablation_holdout), len(test_df))
    print(f"\nWrote results to {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
