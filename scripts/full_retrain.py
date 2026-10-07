#!/usr/bin/env python3
"""Full retrain on combined historical + all live data.

Concatenates data/processed/train.parquet (historical) with the
new-format rows from data/live/bars.parquet, builds the spread-change
target, then trains Tier 1 (LR + XGB) and Tier 2 (GRU + LSTM) on a
time-aware 80/20 per-pair split.

Outputs:
    models/deployed/xgboost.pkl           (overwritten if better)
    models/deployed/linear_regression.pkl (overwritten if better)
    models/deployed/feature_columns.json
    experiments/results/full_retrain/metrics.json
    experiments/results/full_retrain/report.md

Why this exists separately from scripts/scc_retrain_batch.sh: the SCC
batch retrain only fits LR + XGB and uses no held-out test split — it
just refits on everything for production deployment. This script
explicitly evaluates Tier 1 vs Tier 2 with a proper held-out test set
so the deploy/no-deploy decision is data-backed, not blind.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineering import compute_derived_features
from src.features.schemas import ALIGNED_COLUMNS
from src.models.base import BasePredictor
from src.models.linear_regression import LinearRegressionPredictor
from src.models.xgboost_model import XGBoostPredictor

TARGET = "future_spread_change"
TIER2_MIN_BARS = 100  # GRU/LSTM need at least this many bars per pair


def build_combined() -> pd.DataFrame:
    print("[1/6] Loading historical + live data...")
    hist = pd.read_parquet("data/processed/train.parquet")
    live_all = pd.read_parquet("data/live/bars.parquet")
    live = live_all[~live_all["pair_id"].str.startswith("live_")]

    common = sorted(set(hist.columns) & set(live.columns))
    combined = pd.concat([hist[common], live[common]], ignore_index=True)
    print(f"  Historical: {len(hist):,} rows / Live new-format: {len(live):,} rows")
    print(f"  Combined:   {len(combined):,} rows / {combined['pair_id'].nunique():,} pairs")

    for col in ALIGNED_COLUMNS:
        if col not in combined.columns:
            combined[col] = 0.0

    combined = combined.sort_values(["pair_id", "timestamp"]).reset_index(drop=True)
    combined = compute_derived_features(combined).fillna(0.0)

    # Compute target AFTER feature derivation (some derived feature
    # helpers re-build the dataframe column set, which can drop ad-hoc
    # columns added beforehand).
    combined[TARGET] = combined.groupby("pair_id")["spread"].shift(-1) - combined["spread"]
    combined = combined.dropna(subset=["spread", TARGET]).reset_index(drop=True)
    print(f"  After feature derivation + target: {len(combined):,} rows")
    return combined


def time_aware_split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["pair_id", "timestamp"])
    train_parts, test_parts = [], []
    for _, g in df.groupby("pair_id"):
        n = len(g)
        cut = max(1, int(n * train_frac))
        train_parts.append(g.iloc[:cut])
        if n - cut > 0:
            test_parts.append(g.iloc[cut:])
    return pd.concat(train_parts), pd.concat(test_parts)


def select_features(df: pd.DataFrame) -> list[str]:
    exclude = {"timestamp", "pair_id", "group_id", "time_idx", "kalshi_has_trade",
               "polymarket_has_trade", "spread", "future_spread", TARGET}
    return [c for c in df.columns
            if c not in exclude
            and df[c].dtype in ("float64", "float32", "int64", "int32", "bool")]


def metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    nz = (y_true != 0) & (y_pred != 0)
    dir_acc = float(np.mean(np.sign(y_true[nz]) == np.sign(y_pred[nz]))) if nz.sum() > 0 else float("nan")
    pnl_gross = float(np.sum(np.sign(y_pred) * y_true))
    pnl_at_2pp = float(np.sum((np.abs(y_pred) > 0.02) * np.sign(y_pred) * y_true))
    pnl_at_3pp = float(np.sum((np.abs(y_pred) > 0.03) * np.sign(y_pred) * y_true))
    return {"name": name, "n": int(len(y_true)),
            "rmse": rmse, "mae": mae, "dir_acc": dir_acc,
            "pnl_gross": pnl_gross, "pnl_at_2pp": pnl_at_2pp, "pnl_at_3pp": pnl_at_3pp}


def train_tier1(train_df: pd.DataFrame, test_df: pd.DataFrame,
                feature_cols: list[str]) -> dict:
    print("[3/6] Training Tier 1 (LR + XGBoost) on full combined...")
    Xtr = train_df[feature_cols].astype(float)
    ytr = train_df[TARGET].astype(float)
    Xte = test_df[feature_cols].astype(float)
    yte = test_df[TARGET].astype(float)

    t0 = time.time()
    lr = LinearRegressionPredictor(); lr.fit(Xtr, ytr)
    lr_time = time.time() - t0

    t0 = time.time()
    xgb = XGBoostPredictor(); xgb.fit(Xtr, ytr)
    xgb_time = time.time() - t0

    pred_lr = lr.predict(Xte)
    pred_xgb = xgb.predict(Xte)

    m_lr = metrics(yte.values, pred_lr, "linear_regression_full")
    m_xgb = metrics(yte.values, pred_xgb, "xgboost_full")
    m_lr["train_seconds"] = lr_time
    m_xgb["train_seconds"] = xgb_time

    for m in (m_lr, m_xgb):
        print(f"  {m['name']:30s} RMSE={m['rmse']:.4f} dir_acc={m['dir_acc']:.4f} "
              f"PnL@3pp=${m['pnl_at_3pp']:+.2f} ({m['train_seconds']:.1f}s)")

    return {"lr": lr, "xgb": xgb, "metrics": {"lr": m_lr, "xgb": m_xgb}}


def train_tier2(train_df: pd.DataFrame, test_df: pd.DataFrame,
                feature_cols: list[str]) -> dict:
    print("[4/6] Training Tier 2 (GRU + LSTM)...")
    bpp = train_df.groupby("pair_id").size()
    long_pairs = set(bpp[bpp >= TIER2_MIN_BARS].index)
    train_t2 = train_df[train_df["pair_id"].isin(long_pairs)].copy()
    test_t2 = test_df[test_df["pair_id"].isin(long_pairs)].copy()
    print(f"  Tier 2 cohort: {len(long_pairs)} pairs, {len(train_t2):,} train rows, {len(test_t2):,} test rows")

    if len(train_t2) == 0 or len(test_t2) == 0:
        print("  Tier 2 cohort empty — skipping.")
        return {"metrics": {}}

    # Sequence models require group_id + a continuous time_idx per pair.
    # We synthesize a per-pair sequential index if not present.
    for df in (train_t2, test_t2):
        if "group_id" not in df.columns or df["group_id"].isna().any():
            df["group_id"] = df["pair_id"]
        if "time_idx" not in df.columns or df["time_idx"].isna().any():
            df["time_idx"] = df.groupby("pair_id").cumcount().astype(int)

    # Drop zero-variance features within the Tier-2 cohort — sequence
    # models scale features and zero-variance columns would produce NaN.
    # This typically hits features that are 0 across the whole dataset
    # because the underlying exchange doesn't expose them (e.g. Kalshi
    # trade-level buy/sell volume).
    variances = train_t2[feature_cols].var(numeric_only=True)
    zero_var_cols = variances[variances == 0].index.tolist()
    if zero_var_cols:
        print(f"  Dropping zero-variance features for Tier 2: {zero_var_cols}")
    tier2_features = [c for c in feature_cols if c not in zero_var_cols]

    Xtr = train_t2[tier2_features + ["group_id", "time_idx"]].copy()
    ytr = train_t2[TARGET].astype(float)
    Xte = test_t2[tier2_features + ["group_id", "time_idx"]].copy()
    yte = test_t2[TARGET].astype(float)

    out = {"metrics": {}}
    try:
        from src.models.gru import GRUPredictor
        t0 = time.time()
        gru = GRUPredictor(); gru.fit(Xtr, ytr)
        pred = gru.predict(Xte)
        m = metrics(yte.values, np.asarray(pred), "gru_full")
        m["train_seconds"] = time.time() - t0
        out["gru"] = gru
        out["metrics"]["gru"] = m
        print(f"  {m['name']:30s} RMSE={m['rmse']:.4f} dir_acc={m['dir_acc']:.4f} "
              f"PnL@3pp=${m['pnl_at_3pp']:+.2f} ({m['train_seconds']:.1f}s)")
    except Exception as e:
        print(f"  GRU failed: {type(e).__name__}: {e}")
        out["metrics"]["gru"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        from src.models.lstm import LSTMPredictor
        t0 = time.time()
        lstm = LSTMPredictor(); lstm.fit(Xtr, ytr)
        pred = lstm.predict(Xte)
        m = metrics(yte.values, np.asarray(pred), "lstm_full")
        m["train_seconds"] = time.time() - t0
        out["lstm"] = lstm
        out["metrics"]["lstm"] = m
        print(f"  {m['name']:30s} RMSE={m['rmse']:.4f} dir_acc={m['dir_acc']:.4f} "
              f"PnL@3pp=${m['pnl_at_3pp']:+.2f} ({m['train_seconds']:.1f}s)")
    except Exception as e:
        print(f"  LSTM failed: {type(e).__name__}: {e}")
        out["metrics"]["lstm"] = {"error": f"{type(e).__name__}: {e}"}

    return out


def eval_deployed_baseline(test_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    print("[5/6] Evaluating CURRENTLY DEPLOYED pickles on the same test set...")
    deployed_lr = BasePredictor.load(Path("models/deployed/linear_regression.pkl"))
    deployed_xgb = BasePredictor.load(Path("models/deployed/xgboost.pkl"))

    Xte = test_df[feature_cols].astype(float)
    yte = test_df[TARGET].astype(float).values

    m_lr = metrics(yte, deployed_lr.predict(Xte), "linear_regression_deployed")
    m_xgb = metrics(yte, deployed_xgb.predict(Xte), "xgboost_deployed")

    for m in (m_lr, m_xgb):
        print(f"  {m['name']:30s} RMSE={m['rmse']:.4f} dir_acc={m['dir_acc']:.4f} "
              f"PnL@3pp=${m['pnl_at_3pp']:+.2f}")
    return {"lr": m_lr, "xgb": m_xgb}


def decide_and_deploy(tier1: dict, deployed_metrics: dict, feature_cols: list[str]) -> dict:
    print("[6/6] Deploy decision...")
    decisions = {}
    out_dir = Path("models/deployed")

    new_xgb = tier1["metrics"]["xgb"]
    old_xgb = deployed_metrics["xgb"]
    # Deploy if new model improves OR ties on dir_acc AND has competitive RMSE.
    # PnL is informative but noisier; we prioritize calibration + direction.
    if (new_xgb["dir_acc"] >= old_xgb["dir_acc"]
            and new_xgb["rmse"] <= old_xgb["rmse"] * 1.02):
        tier1["xgb"].save(out_dir / "xgboost.pkl")
        decisions["xgboost"] = "deployed"
        print(f"  ✓ XGBoost deployed (new dir_acc {new_xgb['dir_acc']:.4f} ≥ "
              f"old {old_xgb['dir_acc']:.4f}, RMSE within 2%)")
    else:
        decisions["xgboost"] = "kept_old"
        print(f"  ✗ XGBoost kept (new no better — old dir_acc {old_xgb['dir_acc']:.4f}, "
              f"old RMSE {old_xgb['rmse']:.4f})")

    new_lr = tier1["metrics"]["lr"]
    old_lr = deployed_metrics["lr"]
    if (new_lr["dir_acc"] >= old_lr["dir_acc"]
            and new_lr["rmse"] <= old_lr["rmse"] * 1.02):
        tier1["lr"].save(out_dir / "linear_regression.pkl")
        decisions["linear_regression"] = "deployed"
        print(f"  ✓ LR deployed (new dir_acc {new_lr['dir_acc']:.4f} ≥ old {old_lr['dir_acc']:.4f})")
    else:
        decisions["linear_regression"] = "kept_old"
        print(f"  ✗ LR kept (new no better)")

    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    return decisions


def write_report(metrics_all: dict, decisions: dict, report_path: Path) -> None:
    lines = ["# Full Retrain Report", ""]
    lines.append(f"Generated: {pd.Timestamp.now(tz='UTC').isoformat()}")
    lines.append("")
    lines.append("## Dataset")
    ds = metrics_all["dataset"]
    lines.append(f"- Combined rows: **{ds['n_rows']:,}**")
    lines.append(f"- Pairs: **{ds['n_pairs']:,}**")
    lines.append(f"- Train/test split: time-aware, 80/20 per pair")
    lines.append(f"- Train rows: {ds['n_train']:,} / Test rows: {ds['n_test']:,}")
    lines.append(f"- Features: {ds['n_features']}")
    lines.append("")
    lines.append("## Tier 1 vs Deployed (held-out test set)")
    lines.append("")
    lines.append("| Model | RMSE | MAE | dir_acc | PnL@3pp | Notes |")
    lines.append("|---|---|---|---|---|---|")
    for label, m in [
        ("LR (deployed)", metrics_all["deployed"]["lr"]),
        ("LR (full retrain)", metrics_all["tier1"]["lr"]),
        ("XGB (deployed)", metrics_all["deployed"]["xgb"]),
        ("XGB (full retrain)", metrics_all["tier1"]["xgb"]),
    ]:
        lines.append(f"| {label} | {m['rmse']:.4f} | {m['mae']:.4f} | "
                     f"{m['dir_acc']:.4f} | ${m['pnl_at_3pp']:+.2f} | n={m['n']} |")
    lines.append("")
    lines.append("## Tier 2 (sequence models, ≥100-bar cohort)")
    lines.append("")
    lines.append("| Model | RMSE | dir_acc | PnL@3pp |")
    lines.append("|---|---|---|---|")
    for name in ("gru", "lstm"):
        m = metrics_all["tier2"].get(name, {})
        if "error" in m:
            lines.append(f"| {name.upper()} | — | — | error: {m['error']} |")
        elif m:
            lines.append(f"| {name.upper()} | {m['rmse']:.4f} | "
                         f"{m['dir_acc']:.4f} | ${m['pnl_at_3pp']:+.2f} |")
        else:
            lines.append(f"| {name.upper()} | (skipped) | — | — |")
    lines.append("")
    lines.append("## Deploy decisions")
    for k, v in decisions.items():
        emoji = "✓" if v == "deployed" else "—"
        lines.append(f"- {emoji} `{k}`: **{v}**")
    lines.append("")
    lines.append("## Interpretation")
    t1_xgb = metrics_all["tier1"]["xgb"]
    t2_gru = metrics_all["tier2"].get("gru", {})
    t2_lstm = metrics_all["tier2"].get("lstm", {})
    if t2_gru and "error" not in t2_gru:
        delta_gru = t2_gru["pnl_at_3pp"] - t1_xgb["pnl_at_3pp"]
        lines.append(f"- GRU vs XGBoost (full): PnL@3pp delta ${delta_gru:+.2f}")
    if t2_lstm and "error" not in t2_lstm:
        delta_lstm = t2_lstm["pnl_at_3pp"] - t1_xgb["pnl_at_3pp"]
        lines.append(f"- LSTM vs XGBoost (full): PnL@3pp delta ${delta_lstm:+.2f}")
    lines.append("")
    report_path.write_text("\n".join(lines))


def main() -> None:
    out_dir = Path("experiments/results/full_retrain")
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = build_combined()
    feature_cols = select_features(combined)
    print(f"  Features selected: {len(feature_cols)}")

    print("[2/6] Time-aware 80/20 split per pair...")
    train_df, test_df = time_aware_split(combined, 0.8)
    print(f"  Train: {len(train_df):,} rows / Test: {len(test_df):,} rows")

    tier1 = train_tier1(train_df, test_df, feature_cols)
    tier2 = train_tier2(train_df, test_df, feature_cols)
    deployed = eval_deployed_baseline(test_df, feature_cols)
    decisions = decide_and_deploy(tier1, deployed, feature_cols)

    metrics_all = {
        "dataset": {
            "n_rows": int(len(combined)),
            "n_pairs": int(combined["pair_id"].nunique()),
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_features": len(feature_cols),
        },
        "tier1": tier1["metrics"],
        "tier2": tier2["metrics"],
        "deployed": deployed,
        "decisions": decisions,
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_all, f, indent=2)
    write_report(metrics_all, decisions, out_dir / "report.md")

    print(f"\n=== Done. Artifacts in {out_dir} ===")


if __name__ == "__main__":
    main()
