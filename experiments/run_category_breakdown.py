"""Per-category model performance breakdown.

Trains Tier 1 models on full train.parquet, predicts on test.parquet, then
stratifies results by category (oil, crypto, politics, etc.) to answer:
    "Does GRU/LSTM beat XGBoost on ANY category, even if it loses overall?"

A "yes" anywhere would be a nuanced finding for the paper. A "no" strengthens
the simpler-is-better finding by showing it holds across every subset.

Run:
    python -m experiments.run_category_breakdown
    python -m experiments.run_category_breakdown --min-trades 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.engineering import compute_derived_features
from src.features.category import derive_category_from_pair_id, CATEGORIES
from src.models.linear_regression import LinearRegressionPredictor
from src.models.xgboost_model import XGBoostPredictor

logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = {
    "timestamp", "pair_id", "time_idx", "group_id", "spread_change_target",
    "kalshi_order_flow_imbalance", "kalshi_buy_volume",
    "kalshi_sell_volume", "kalshi_realized_spread",
}
TARGET = "spread_change_target"


def _build(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_derived_features(df).fillna(0)
    sort_cols = [c for c in ["pair_id", "time_idx", "timestamp"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df[TARGET] = df.groupby("pair_id")["spread"].shift(-1) - df["spread"]
    return df.dropna(subset=["spread", TARGET]).reset_index(drop=True).fillna(0)


def _feature_cols(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in numeric if c not in NON_FEATURE_COLUMNS]


def _metrics_for_subset(preds: np.ndarray, actuals: np.ndarray, fee: float = 0.02) -> dict:
    trade_pnls = []
    for p, a in zip(preds, actuals):
        if abs(p) > fee:
            if np.sign(p) == np.sign(a):
                trade_pnls.append(abs(a) - fee)
            else:
                trade_pnls.append(-(abs(a) + fee))
    if not trade_pnls:
        return {"num_trades": 0, "pnl": 0.0, "win_rate": 0.0,
                "sharpe_per_trade": 0.0, "avg_pnl": 0.0}
    arr = np.array(trade_pnls)
    return {
        "num_trades": int(len(arr)),
        "pnl": float(np.sum(arr)),
        "avg_pnl": float(np.mean(arr)),
        "win_rate": float(np.mean(arr > 0)),
        "sharpe_per_trade": float(np.mean(arr) / np.std(arr, ddof=1)) if np.std(arr, ddof=1) > 0 else 0.0,
    }


def run_category_breakdown(
    data_dir: Path = Path("data/processed"),
    min_trades: int = 10,
    output_dir: Path = Path("experiments/results"),
) -> dict:
    train = _build(pd.read_parquet(data_dir / "train.parquet"))
    test = _build(pd.read_parquet(data_dir / "test.parquet"))
    feature_cols = _feature_cols(train)

    logger.info(
        "Train: %d rows, Test: %d rows, Features: %d",
        len(train), len(test), len(feature_cols),
    )

    # Tag each test row with a category
    test["category"] = test["pair_id"].apply(derive_category_from_pair_id)
    logger.info("Test categories: %s", dict(test["category"].value_counts()))

    # Train models
    models = {
        "linear_regression": LinearRegressionPredictor(),
        "xgboost": XGBoostPredictor(max_depth=3, learning_rate=0.01, n_estimators=100),
    }

    # Try to include GRU/LSTM if torch is available
    try:
        from src.models.gru import GRUPredictor
        from src.models.lstm import LSTMPredictor
        models["gru"] = GRUPredictor()
        models["lstm"] = LSTMPredictor()
        logger.info("Including Tier 2 (GRU + LSTM)")
    except ImportError:
        logger.info("torch not available — skipping Tier 2")

    X_train = train[feature_cols]
    y_train = train[TARGET]
    X_test = test[feature_cols]
    y_test = test[TARGET].to_numpy()

    # Train + predict
    preds_by_model: dict[str, np.ndarray] = {}
    for name, model in models.items():
        logger.info("Training %s on %d rows...", name, len(X_train))
        try:
            if name in ("gru", "lstm"):
                # Sequence models need different prep — skip if it fails
                from experiments.run_baselines import prepare_xy_for_seq
                X_seq_tr, _ = prepare_xy_for_seq(train, feature_cols)
                X_seq_te, _ = prepare_xy_for_seq(test, feature_cols)
                model.fit(X_seq_tr, y_train.to_numpy())
                preds_by_model[name] = model.predict(X_seq_te)
            else:
                model.fit(X_train, y_train)
                preds_by_model[name] = model.predict(X_test)
        except Exception as e:
            logger.warning("Skipping %s: %s", name, e)

    # Stratify metrics by category
    results: dict = {m: {} for m in preds_by_model}

    # Overall (for reference)
    for name, preds in preds_by_model.items():
        results[name]["_overall"] = _metrics_for_subset(preds, y_test)

    # Per-category
    for cat in sorted(set(test["category"].tolist())):
        mask = (test["category"] == cat).to_numpy()
        if mask.sum() < min_trades:
            continue
        for name, preds in preds_by_model.items():
            results[name][cat] = _metrics_for_subset(preds[mask], y_test[mask])

    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON output
    json_path = output_dir / "category_breakdown.json"
    json_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote %s", json_path)

    # Pretty text table
    lines = []
    lines.append("=" * 110)
    lines.append("PER-CATEGORY MODEL PERFORMANCE BREAKDOWN")
    lines.append("=" * 110)
    lines.append("")

    # Determine which categories have data for at least one model
    all_cats = sorted(set(k for m in results.values() for k in m.keys()))
    all_cats = [c for c in all_cats if c != "_overall"]
    all_cats.insert(0, "_overall")  # show overall first

    header = f"{'Category':<20s} " + "  ".join(
        f"{name:<15s}" for name in preds_by_model.keys()
    )
    lines.append(header)
    lines.append("-" * len(header))

    for cat in all_cats:
        row = f"{cat:<20s} "
        cells = []
        for name in preds_by_model.keys():
            m = results[name].get(cat)
            if m and m["num_trades"] >= min_trades:
                cells.append(f"${m['pnl']:>+6.2f} WR={m['win_rate']:.0%} n={m['num_trades']:>4d}")
            elif m:
                cells.append(f"(n={m['num_trades']}, skipped)")
            else:
                cells.append("—")
        row += "  ".join(f"{c:<22s}" for c in cells)
        lines.append(row)

    lines.append("")
    lines.append("Best model per category (by P&L, min_trades filter applied):")
    for cat in all_cats:
        scores = [
            (name, results[name].get(cat, {}).get("pnl", 0),
             results[name].get(cat, {}).get("num_trades", 0))
            for name in preds_by_model.keys()
        ]
        scores = [(n, p, t) for n, p, t in scores if t >= min_trades]
        if not scores:
            continue
        scores.sort(key=lambda x: -x[1])
        winner = scores[0]
        lines.append(f"  {cat:<20s}  winner: {winner[0]:<20s}  P&L=${winner[1]:+.2f}")

    out_text = "\n".join(lines)
    txt_path = output_dir / "category_breakdown_table.txt"
    txt_path.write_text(out_text)
    print(out_text)

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-category performance breakdown")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--min-trades", type=int, default=10,
                        help="Only report categories with this many trades")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_category_breakdown(
        data_dir=Path(args.data_dir),
        min_trades=args.min_trades,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
