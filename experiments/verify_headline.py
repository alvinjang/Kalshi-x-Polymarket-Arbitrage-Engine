"""Fresh verification run for paper's headline table — Tier 1 + Tier 2 only.

Bypasses PPO imports. Uses the same data pipeline and evaluation as
run_walk_forward.py so numbers are directly comparable with the walk-forward
results. Prints a clean Markdown table ready for paste into the paper.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.engineering import compute_derived_features
from src.models.linear_regression import LinearRegressionPredictor
from src.models.naive import NaivePredictor
from src.models.volume import VolumePredictor
from src.models.xgboost_model import XGBoostPredictor
from src.utils.seed import set_all_seeds

NON_FEATURE_COLUMNS = {
    "timestamp", "pair_id", "time_idx", "group_id", "spread_change_target",
    "kalshi_order_flow_imbalance", "kalshi_buy_volume",
    "kalshi_sell_volume", "kalshi_realized_spread",
}
TARGET = "spread_change_target"


def build(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_derived_features(df).fillna(0.0)
    sort_cols = [c for c in ["pair_id", "time_idx", "timestamp"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df[TARGET] = df.groupby("pair_id")["spread"].shift(-1) - df["spread"]
    df = df.dropna(subset=["spread", TARGET]).reset_index(drop=True).fillna(0.0)
    if "group_id" not in df.columns:
        df["group_id"] = df["pair_id"].astype("category").cat.codes
    return df


def feature_cols(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in numeric if c not in NON_FEATURE_COLUMNS]


def simulate_pnl(preds: np.ndarray, actuals: np.ndarray, fee: float = 0.02) -> dict:
    pnls = []
    for p, a in zip(preds, actuals):
        if abs(p) > fee:
            if np.sign(p) == np.sign(a):
                pnls.append(abs(a) - fee)
            else:
                pnls.append(-(abs(a) + fee))
    if not pnls:
        return {"num_trades": 0, "pnl": 0.0, "win_rate": 0.0, "sharpe_per_trade": 0.0}
    arr = np.array(pnls)
    return {
        "num_trades": int(len(arr)),
        "pnl": float(np.sum(arr)),
        "win_rate": float(np.mean(arr > 0)),
        "sharpe_per_trade": float(np.mean(arr) / np.std(arr, ddof=1)) if np.std(arr, ddof=1) > 0 else 0.0,
    }


def main():
    set_all_seeds(42)
    data_dir = Path("data/processed")
    train = build(pd.read_parquet(data_dir / "train.parquet"))
    test = build(pd.read_parquet(data_dir / "test.parquet"))
    feats = feature_cols(train)

    print(f"Train: {len(train):,} rows, Test: {len(test):,} rows, Features: {len(feats)}")
    print(f"Train pairs: {train['pair_id'].nunique()}, Test pairs: {test['pair_id'].nunique()}")

    X_train = train[feats]
    y_train = train[TARGET]
    X_test = test[feats]
    y_test = test[TARGET].to_numpy()

    # Sequence-model prep (drop zero-variance cols)
    nonzero = [c for c in feats if train[c].std() > 1e-10]
    seq_cols = nonzero + ["group_id"]
    X_train_seq = train[seq_cols]
    X_test_seq = test[seq_cols]

    results = {}
    models = {
        "naive": NaivePredictor,
        "volume": VolumePredictor,
        "linear_regression": LinearRegressionPredictor,
        "xgboost": XGBoostPredictor,
    }

    # Tier 2 optional
    try:
        from src.models.gru import GRUPredictor
        from src.models.lstm import LSTMPredictor
        models["gru"] = GRUPredictor
        models["lstm"] = LSTMPredictor
        print("Tier 2 enabled (GRU + LSTM)")
    except ImportError as e:
        print(f"Tier 2 skipped: {e}")

    for name, factory in models.items():
        print(f"Training {name}...", flush=True)
        model = factory()
        try:
            if name in ("gru", "lstm"):
                model.fit(X_train_seq, y_train.to_numpy())
                preds = model.predict(X_test_seq)
            else:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
        da_mask = y_test != 0
        da = float(np.mean(np.sign(preds[da_mask]) == np.sign(y_test[da_mask]))) if da_mask.sum() else 0.0
        trade = simulate_pnl(preds, y_test, fee=0.02)
        results[name] = {
            "rmse": round(rmse, 5),
            "directional_accuracy": round(da, 4),
            **{k: round(v, 5) if isinstance(v, float) else v for k, v in trade.items()},
        }

    out = Path("experiments/results/verify_headline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "features": len(feats),
                               "train_rows": len(train), "test_rows": len(test),
                               "timestamp": datetime.now(timezone.utc).isoformat()},
                              indent=2))
    print(f"\nWrote {out}")

    # Markdown table
    print("\n## Fresh Headline Table\n")
    print("| Model | RMSE | Dir. Acc. | P&L @2pp | Win Rate | Sharpe/trade | # trades |")
    print("|---|---|---|---|---|---|---|")
    order = ["naive", "volume", "linear_regression", "xgboost", "gru", "lstm"]
    for name in order:
        if name not in results:
            continue
        r = results[name]
        print(f"| {name} | {r['rmse']:.4f} | {r['directional_accuracy']:.4f} | "
              f"${r['pnl']:+.2f} | {r['win_rate']:.4f} | "
              f"{r['sharpe_per_trade']:.4f} | {r['num_trades']} |")


if __name__ == "__main__":
    sys.exit(main() or 0)
