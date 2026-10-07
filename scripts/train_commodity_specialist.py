#!/usr/bin/env python3
"""Train a commodity-specialist XGBoost predictor.

Rationale: live data shows the pooled model averages the commodity edge
away with money-losing categories (crypto KXDOGED at 34% WR, employment
KXPAYROLLS at 33% WR). A specialist trained only on oil + non-oil
commodity rows should produce sharper predictions on those pairs.

Outputs:
    models/deployed/xgboost_commodity.pkl
    models/deployed/feature_columns_commodity.json  (same as pooled)
    experiments/results/commodity_specialist/metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.category import derive_category_from_ticker
from src.features.engineering import compute_derived_features
from src.features.schemas import ALIGNED_COLUMNS
from src.models.xgboost_model import XGBoostPredictor


COMMODITY_CATEGORIES = {"oil", "commodities"}


def main() -> None:
    print("=== Commodity Specialist Trainer ===")

    hist = pd.read_parquet("data/processed/train.parquet")
    live_all = pd.read_parquet("data/live/bars.parquet")
    live = live_all[~live_all["pair_id"].str.startswith("live_")]
    common = sorted(set(hist.columns) & set(live.columns))
    combined = pd.concat([hist[common], live[common]], ignore_index=True)
    print(f"  Combined: {len(combined):,} rows / {combined['pair_id'].nunique():,} pairs")

    mapping = json.load(open("data/live/pair_mapping.json"))
    pid_to_ticker = {pid: m["kalshi_market_id"] for pid, m in mapping.items()}

    def category_of(pair_id: str) -> str:
        ticker = pid_to_ticker.get(pair_id, "")
        if not ticker:
            ticker = pair_id.upper().split("-")[0]
        return derive_category_from_ticker(ticker)

    combined["category"] = combined["pair_id"].map(category_of)
    print(f"  Categories: {combined['category'].value_counts().head(10).to_dict()}")

    commodity = combined[combined["category"].isin(COMMODITY_CATEGORIES)].copy()
    pooled = combined.copy()
    print(f"  Commodity subset: {len(commodity):,} rows / {commodity['pair_id'].nunique()} pairs")

    for col in ALIGNED_COLUMNS:
        if col not in commodity.columns:
            commodity[col] = 0.0
        if col not in pooled.columns:
            pooled[col] = 0.0
    commodity = compute_derived_features(commodity).fillna(0.0)
    pooled = compute_derived_features(pooled).fillna(0.0)

    feature_cols = json.load(open("models/deployed/feature_columns.json"))
    print(f"  Using {len(feature_cols)} features (matching pooled)")

    if "future_spread" in commodity.columns:
        target = "future_spread"
    else:
        target = "spread"

    # Time-aware split: train on 80% earliest, test on 20% latest per pair
    def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = df.sort_values(["pair_id", "timestamp"])
        train_parts = []
        test_parts = []
        for _, g in df.groupby("pair_id"):
            n = len(g)
            cut = int(n * 0.8)
            train_parts.append(g.iloc[:cut])
            test_parts.append(g.iloc[cut:])
        return pd.concat(train_parts), pd.concat(test_parts)

    comm_train, comm_test = split(commodity)
    pool_train, pool_test = split(pooled)
    print(f"  Commodity train/test: {len(comm_train):,} / {len(comm_test):,}")

    for c in feature_cols:
        if c not in comm_train.columns:
            comm_train[c] = 0.0
            comm_test[c] = 0.0
            pool_train[c] = 0.0
            pool_test[c] = 0.0

    def prep(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = df[feature_cols].astype(float)
        y = df[target].astype(float)
        m = ~(X.isna().any(axis=1) | y.isna())
        return X[m], y[m]

    Xtr_c, ytr_c = prep(comm_train)
    Xte_c, yte_c = prep(comm_test)
    Xtr_p, ytr_p = prep(pool_train)
    Xte_p, yte_p = prep(pool_test)

    # --- Train commodity specialist ---
    print("\n[1/2] Training commodity-specialist XGBoost...")
    comm_xgb = XGBoostPredictor()
    comm_xgb.fit(Xtr_c, ytr_c)
    pred_comm_on_comm = comm_xgb.predict(Xte_c)

    # --- Pooled model: reuse the deployed pickle and predict on commodity test ---
    print("[2/2] Loading pooled XGBoost and evaluating on commodity test set...")
    from src.models.base import BasePredictor
    pooled_xgb = BasePredictor.load(Path("models/deployed/xgboost.pkl"))
    pred_pool_on_comm = pooled_xgb.predict(Xte_c)

    def metrics(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        # Direction accuracy: sign of prediction vs sign of realized
        nz = (y_true != 0) & (y_pred != 0)
        if nz.sum() > 0:
            dir_acc = float(np.mean(np.sign(y_true[nz]) == np.sign(y_pred[nz])))
        else:
            dir_acc = float("nan")
        # PnL proxy: prediction * realized (trade direction = sign of prediction)
        pnl = float(np.sum(np.sign(y_pred) * y_true))
        pnl_at_3pp = float(np.sum((np.abs(y_pred) > 0.03) * np.sign(y_pred) * y_true))
        return {
            "name": name,
            "n": int(len(y_true)),
            "rmse": rmse,
            "mae": mae,
            "dir_acc": dir_acc,
            "pnl_gross": pnl,
            "pnl_at_3pp": pnl_at_3pp,
        }

    m_comm = metrics(yte_c.values, pred_comm_on_comm, "commodity_specialist")
    m_pool = metrics(yte_c.values, pred_pool_on_comm, "pooled (on commodity test)")

    print(f"\n=== Results on commodity test set ===")
    for m in (m_comm, m_pool):
        print(f"  {m['name']:40s} RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
              f"dir_acc={m['dir_acc']:.4f}  PnL@3pp=${m['pnl_at_3pp']:+.2f}")

    delta_pnl = m_comm["pnl_at_3pp"] - m_pool["pnl_at_3pp"]
    delta_rmse = m_pool["rmse"] - m_comm["rmse"]  # positive = specialist better
    print(f"\n  PnL delta: ${delta_pnl:+.2f}   RMSE improvement: {delta_rmse:+.4f}")

    out_dir = Path("models/deployed")
    out_metrics_dir = Path("experiments/results/commodity_specialist")
    out_metrics_dir.mkdir(parents=True, exist_ok=True)

    if m_comm["pnl_at_3pp"] >= m_pool["pnl_at_3pp"] and m_comm["dir_acc"] >= m_pool["dir_acc"]:
        comm_xgb.save(out_dir / "xgboost_commodity.pkl")
        with open(out_dir / "feature_columns_commodity.json", "w") as f:
            json.dump(feature_cols, f)
        print(f"\n  ✓ Specialist wins → saved to {out_dir}/xgboost_commodity.pkl")
        winner = "specialist"
    else:
        print(f"\n  ✗ Specialist did not beat pooled — not deploying")
        winner = "pooled"

    with open(out_metrics_dir / "metrics.json", "w") as f:
        json.dump({
            "commodity_specialist": m_comm,
            "pooled_on_commodity": m_pool,
            "winner": winner,
            "commodity_categories": sorted(COMMODITY_CATEGORIES),
            "n_commodity_pairs": int(commodity["pair_id"].nunique()),
            "n_commodity_rows": int(len(commodity)),
        }, f, indent=2)


if __name__ == "__main__":
    main()
