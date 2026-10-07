"""Tier 1: Sharpe 3.2 audit.

Recomputes per-trade and per-pair Sharpe from raw test-set trade ledger,
applies cross-sectional pair-correlation correction (Lo / Bailey-López
de Prado framework), bootstraps 95% CIs, writes
experiments/results/audit/sharpe_audit.json.

This is the kill-or-confirm script for the headline per-pair Sharpe ≈ 3.2
claim: it either confirms 3.2 with documented caveats (verdict=PASS) or
surfaces the corrected number (verdict=CORRECTED), in which case Plan 07
propagates the correction into the abstract / §5 / Table 8 footnote.

Verdict logic:
    * FAILED   -- raw recompute drifted from canonical headline.json by > 0.01
    * CORRECTED -- avg_pair_corr > 0.10 AND corrected < 0.5 * naive
    * PASS     -- otherwise (recompute matches canonical AND correlation small)

All randomness goes through src.utils.seed.set_all_seeds(SEED=42); bootstrap
uses np.random.default_rng(seed=42). Re-running the script must produce
byte-identical JSON output.

Implements requirement: AUDIT-01.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.run_baselines import (
    NON_FEATURE_COLUMNS, TARGET_COLUMN, _build_split,
    _feature_columns, load_train_test, prepare_xy,
)
from src.models.linear_regression import LinearRegressionPredictor
from src.utils.seed import set_all_seeds

CANONICAL = Path("experiments/results/canonical/headline.json")
OUT_PATH = Path("experiments/results/audit/sharpe_audit.json")

# Locked: paper headline uses LR (per-trade Sharpe 0.501). Audit defends LR.
HEADLINE_MODEL = "linear_regression"
N_BOOTSTRAP = 10_000
SEED = 42
THRESHOLD = 0.02  # canonical
POSITION_SIZE = 100.0  # canonical


def build_trade_ledger() -> pd.DataFrame:
    """Reproduce LR test-set predictions and emit per-trade ledger.

    Returns DataFrame with columns:
        pair_id, entry_ts (epoch s), entry_day (day ordinal),
        pred, actual, pnl_pp, traded (bool)
    Filtered to traded == True.
    """
    set_all_seeds(SEED)
    train_raw, test_raw = load_train_test(Path("data/processed"))
    train_df = _build_split(train_raw)
    test_df = _build_split(test_raw)
    feature_cols = _feature_columns(train_df)

    X_train, y_train = prepare_xy(train_df, feature_cols)
    X_test, _ = prepare_xy(test_df, feature_cols)

    model = LinearRegressionPredictor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    actuals = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    traded = np.abs(preds) > THRESHOLD
    direction = np.sign(preds)
    pnl_pp = direction * actuals

    # NOTE on timestamp normalization: the canonical processed parquet stores
    # `timestamp` as int64 epoch SECONDS, not pandas datetime64 (nanoseconds).
    # The original RESEARCH.md skeleton assumed datetime64 and divided by 1e9
    # which collapsed all entries to the same day (test_span_days = 0).
    # Auto-detected here: if values look like nanoseconds (> 10**12), divide;
    # otherwise treat as already-epoch-seconds (this is the canonical case).
    ts_raw = test_df["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts_raw):
        entry_ts = ts_raw.astype("int64").values // 10**9
    else:
        ts_int = ts_raw.astype("int64").values
        # Heuristic: epoch seconds in 2024-2026 are ~1.7e9; nanoseconds ~1.7e18.
        if ts_int.max() > 10**12:
            entry_ts = ts_int // 10**9
        else:
            entry_ts = ts_int

    ledger = pd.DataFrame({
        "pair_id": test_df["pair_id"].values,
        "entry_ts": entry_ts,
        "pred": preds,
        "actual": actuals,
        "pnl_pp": pnl_pp,
        "traded": traded,
    })
    ledger["entry_day"] = ledger["entry_ts"] // 86400
    return ledger.loc[ledger["traded"]].copy()


def per_trade_sharpe(pnl_pp: np.ndarray) -> float:
    """Raw mean/std of per-trade returns (unannualized)."""
    if len(pnl_pp) < 2 or pnl_pp.std(ddof=1) == 0:
        return 0.0
    return float(pnl_pp.mean() / pnl_pp.std(ddof=1))


def per_pair_returns(ledger: pd.DataFrame) -> pd.Series:
    """Sum pnl_pp per pair_id -> 1 return per pair."""
    return ledger.groupby("pair_id")["pnl_pp"].sum()


def per_pair_sharpe_naive(pair_returns: pd.Series) -> float:
    """mean / std (ddof=1) over the per-pair return vector."""
    if len(pair_returns) < 2 or pair_returns.std(ddof=1) == 0:
        return 0.0
    return float(pair_returns.mean() / pair_returns.std(ddof=1))


def avg_pairwise_correlation(ledger: pd.DataFrame) -> tuple[float, int]:
    """Compute mean of upper-triangle of contemporaneous-return correlation matrix.

    Pivot ledger to (entry_day x pair_id) of summed pnl_pp; use day as the
    "calendar bar" for cross-sectional correlation. Pairs that don't trade
    together on the same day contribute NaN, which corr() ignores pairwise.
    """
    panel = ledger.pivot_table(
        index="entry_day", columns="pair_id",
        values="pnl_pp", aggfunc="sum",
    )
    corr = panel.corr()  # NaN-aware pairwise correlation
    # Upper triangle, exclude diagonal
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    triu_vals = corr.values[mask]
    triu_vals = triu_vals[~np.isnan(triu_vals)]
    if len(triu_vals) == 0:
        return 0.0, 0
    return float(triu_vals.mean()), len(triu_vals)


def correlation_corrected_sharpe(
    sharpe_naive: float, n_pairs: int, avg_corr: float
) -> tuple[float, float]:
    """Apply effective-sample-size correction.

    n_eff = N / (1 + (N - 1) * avg_corr)
    sharpe_corrected = sharpe_naive * sqrt(n_eff / N)
    """
    if avg_corr <= 0 or n_pairs <= 1:
        return sharpe_naive, float(n_pairs)
    n_eff = n_pairs / (1.0 + (n_pairs - 1) * avg_corr)
    correction_factor = np.sqrt(n_eff / n_pairs)
    return float(sharpe_naive * correction_factor), float(n_eff)


def bootstrap_sharpe_ci(
    pair_returns: pd.Series, n_boot: int = N_BOOTSTRAP, seed: int = SEED
) -> tuple[float, float]:
    """Percentile-method 95% CI on per-pair Sharpe via 10,000 resamples."""
    rng = np.random.default_rng(seed)
    arr = pair_returns.to_numpy()
    boot_sharpes = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        if sample.std(ddof=1) == 0:
            boot_sharpes[i] = 0.0
        else:
            boot_sharpes[i] = sample.mean() / sample.std(ddof=1)
    return float(np.percentile(boot_sharpes, 2.5)), float(np.percentile(boot_sharpes, 97.5))


def annualization_factor(ledger: pd.DataFrame) -> dict:
    """Document how the per-pair Sharpe is annualized.

    Per-pair-once-per-year framework:
        annualized_sharpe = per_pair_sharpe * sqrt(N_pairs_per_year)

    where N_pairs_per_year is derived from average pair lifecycle length.
    """
    n_pairs = ledger["pair_id"].nunique()
    test_span_seconds = ledger["entry_ts"].max() - ledger["entry_ts"].min()
    test_span_days = test_span_seconds / 86400.0
    pairs_per_year = n_pairs * (365.0 / max(test_span_days, 1.0))
    return {
        "n_pairs": int(n_pairs),
        "test_span_days": round(test_span_days, 2),
        "pairs_per_year": round(pairs_per_year, 2),
        "annualization_factor": round(np.sqrt(pairs_per_year), 4),
        "formula": "annualized_sharpe = per_pair_sharpe_naive * sqrt(pairs_per_year)",
    }


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ledger = build_trade_ledger()
    pt_sharpe = per_trade_sharpe(ledger["pnl_pp"].to_numpy())

    pair_ret = per_pair_returns(ledger)
    pp_sharpe_raw = per_pair_sharpe_naive(pair_ret)

    avg_corr, n_pairs_compared = avg_pairwise_correlation(ledger)
    pp_sharpe_corrected, n_eff = correlation_corrected_sharpe(
        pp_sharpe_raw, len(pair_ret), avg_corr
    )

    ci_low, ci_high = bootstrap_sharpe_ci(pair_ret)
    ann = annualization_factor(ledger)
    pp_sharpe_annualized = pp_sharpe_raw * ann["annualization_factor"]
    pp_sharpe_annualized_corrected = pp_sharpe_corrected * ann["annualization_factor"]

    # Cross-check vs canonical headline.json
    canonical = json.loads(CANONICAL.read_text())
    canon_pt = canonical["models"][HEADLINE_MODEL]["sharpe_per_trade"]
    pt_drift = abs(pt_sharpe - canon_pt)

    verdict = "PASS"
    if pt_drift > 0.01:
        verdict = "FAILED"  # raw recompute drifted from canonical headline
    elif avg_corr > 0.10 and pp_sharpe_corrected < 0.5 * pp_sharpe_raw:
        # Correction collapsed the headline by >50%: the paper claim is INFLATED
        verdict = "CORRECTED"

    out = {
        "audit": "sharpe",
        "tier": 1,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "canonical_input": CANONICAL.as_posix(),
        "model_audited": HEADLINE_MODEL,
        "per_trade_sharpe_recomputed": pt_sharpe,
        "per_trade_sharpe_canonical": canon_pt,
        "per_trade_sharpe_drift": pt_drift,
        "per_pair_sharpe_naive": pp_sharpe_raw,
        "per_pair_sharpe_naive_ci_95": [ci_low, ci_high],
        "avg_pairwise_corr": avg_corr,
        "n_pairs_compared": n_pairs_compared,
        "n_eff": n_eff,
        "per_pair_sharpe_corr_corrected": pp_sharpe_corrected,
        "annualization": ann,
        "per_pair_sharpe_annualized_naive": pp_sharpe_annualized,
        "per_pair_sharpe_annualized_corrected": pp_sharpe_annualized_corrected,
        "assumptions": [
            "Per-pair returns are stationary (no regime change within test window).",
            "Pair-correlation effective-sample correction follows Bailey-López de Prado (2012) "
            "framework: n_eff = N / (1 + (N-1) * avg_corr); sharpe_corrected = "
            "sharpe_naive * sqrt(n_eff/N).",
            f"Annualization uses pairs_per_year = {ann['pairs_per_year']}, derived from "
            f"test_span_days = {ann['test_span_days']} and N_pairs = {ann['n_pairs']}. "
            "Assumes pair-lifecycle distribution in test window is representative of "
            "annual operation (likely violated; flag for §6.4).",
            "Bootstrap CI uses simple resample-with-replacement (10,000 resamples). "
            "Does NOT correct for autocorrelation in per-pair returns "
            "(use Politis-Romano stationary bootstrap if AR(1) coef is large).",
        ],
        "n_bootstrap": N_BOOTSTRAP,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH} verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
