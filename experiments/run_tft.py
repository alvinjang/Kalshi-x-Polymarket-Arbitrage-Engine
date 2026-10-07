"""Single-split TFT experiment runner.

Runs TFTPredictor with 3 seeds on the standard single-split (train.parquet /
test.parquet). Implements Option B gate: always writes TFT.json — either with
real results or a documented negative-result sentinel when training fails or
val_loss >= GRU baseline RMSE (0.2928).

GRU baseline to beat (TFT-04 gate): RMSE 0.2928, P&L +$212.50
LSTM reference: RMSE 0.2915, P&L +$221.84

Usage:
    python -m experiments.run_tft
"""
from __future__ import annotations

import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to sys.path (same pattern as other experiment scripts)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.engineering import compute_derived_features
from src.utils.seed import set_all_seeds
from src.models.tft import TFTPredictor
from experiments.run_baselines import (
    prepare_xy_for_seq,
    DEFAULT_THRESHOLD as THRESHOLD,
    _build_split,
    _feature_columns,
)

GRU_RMSE_BASELINE = 0.2928
SEEDS = [42, 7, 123]
RESULTS_DIR = Path("experiments/results/tier2")


def _load_with_group_id(parquet_path: Path) -> pd.DataFrame:
    """Load parquet, compute derived features, and re-attach group_id.

    compute_derived_features() filters to OUTPUT_COLUMNS which does not
    include group_id. We preserve group_id from the raw parquet before
    calling compute_derived_features, then re-attach it so that
    prepare_xy_for_seq (which needs group_id) works correctly.
    """
    raw = pd.read_parquet(parquet_path)
    group_ids = raw["group_id"].copy()  # int64 pair index
    derived = compute_derived_features(raw)
    derived["group_id"] = group_ids.values
    return derived


def main() -> None:
    set_all_seeds(42)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare data, preserving group_id for TFT per-pair windowing.
    # Note: load_train_test() calls compute_derived_features() which drops
    # group_id (not in OUTPUT_COLUMNS schema). We use _load_with_group_id()
    # to re-attach it from the raw parquet.
    data_dir = Path("data/processed")
    if not (data_dir / "train.parquet").exists():
        print(f"Error: {data_dir / 'train.parquet'} not found", file=sys.stderr)
        sys.exit(1)

    train_raw = _load_with_group_id(data_dir / "train.parquet")
    test_raw = _load_with_group_id(data_dir / "test.parquet")
    train = _build_split(train_raw)
    test = _build_split(test_raw)

    feature_cols_all = _feature_columns(train)
    # Filter out zero-variance columns (same guard as run_walk_forward.py)
    # kalshi_kyle_lambda has zero variance in the full training set.
    nonzero_var_cols = [
        c for c in feature_cols_all if train[c].std() > 1e-10
    ]
    feature_cols = nonzero_var_cols
    print(
        f"Data loaded: {len(train)} train rows, {len(test)} test rows, "
        f"{len(feature_cols)} features ({len(feature_cols_all) - len(feature_cols)} "
        f"zero-variance columns removed)."
    )

    X_train, y_train = prepare_xy_for_seq(train, feature_cols)
    X_test, y_test = prepare_xy_for_seq(test, feature_cols)

    seed_rmses: list[float] = []
    seed_pnls: list[float] = []
    last_metrics: dict | None = None
    last_audit: dict | None = None

    for seed in SEEDS:
        print(f"\n{'=' * 60}")
        print(f"Training TFT seed={seed}")
        print(f"{'=' * 60}")
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = TFTPredictor(random_state=seed)
                model.fit(X_train, y_train)
                metrics = model.evaluate(X_test, y_test, threshold=THRESHOLD)
                metrics.pop("pnl_series", None)
                audit = model._audit_attention()

            elapsed = time.time() - t0
            print(
                f"  Seed {seed} done in {elapsed:.0f}s — "
                f"RMSE={metrics['rmse']:.4f}  P&L=${metrics['total_pnl']:.2f}  "
                f"entropy={audit['entropy']:.3f}  degenerate={audit['is_degenerate']}"
            )
            seed_rmses.append(metrics["rmse"])
            seed_pnls.append(metrics["total_pnl"])
            last_metrics = metrics
            last_audit = audit

        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            # Write negative result artifact (Option B gate)
            negative: dict = {
                "model": "TFT",
                "converged": False,
                "failure_reason": str(e),
                "note": (
                    "TFT did not converge at N=6802. "
                    "Documented as negative result per TFT-04 Option B."
                ),
                "gru_baseline_rmse": GRU_RMSE_BASELINE,
                "seeds_attempted": SEEDS,
                "seeds_completed": len(seed_rmses),
                "seed_rmses_so_far": seed_rmses,
            }
            out = RESULTS_DIR / "TFT.json"
            out.write_text(json.dumps(negative, indent=2))
            print(f"\nNegative result written to {out}")
            return

    avg_rmse = float(np.mean(seed_rmses))
    avg_pnl = float(np.mean(seed_pnls))
    converged = avg_rmse < GRU_RMSE_BASELINE

    result: dict = {
        "model": "TFT",
        "converged": converged,
        "rmse": avg_rmse,
        "rmse_std": float(np.std(seed_rmses)),
        "seed_rmses": seed_rmses,
        "total_pnl": avg_pnl,
        "seed_pnls": seed_pnls,
        "gru_baseline_rmse": GRU_RMSE_BASELINE,
        "beats_gru": converged,
        "attention_audit": last_audit,
        "note": (
            "TFT converged and beats GRU baseline (RMSE < 0.2928)."
            if converged
            else (
                f"TFT did not beat GRU at N=6802 (avg RMSE={avg_rmse:.4f} vs "
                f"GRU={GRU_RMSE_BASELINE}). "
                "Documented negative result per TFT-04 Option B. "
                "Extends the simplicity-wins thesis to transformer architectures."
            )
        ),
        **{k: v for k, v in (last_metrics or {}).items()},
    }

    out = RESULTS_DIR / "TFT.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nResult written to {out}")
    print(f"Converged: {converged} | Avg RMSE: {avg_rmse:.4f} | GRU baseline: {GRU_RMSE_BASELINE}")
    if converged:
        print("TFT BEATS GRU baseline — new best Tier 2 model!")
    else:
        print("TFT does not beat GRU — simplicity-wins thesis extended to transformers.")


if __name__ == "__main__":
    main()
