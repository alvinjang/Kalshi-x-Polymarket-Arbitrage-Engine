"""Tier 1 Sharpe audit redo on the leakage-free pair-stratified split.

Sister script to ``experiments/audit/audit_sharpe.py``. Reuses every
metric helper from the canonical audit (``per_pair_returns``,
``per_pair_sharpe_naive``, ``avg_pairwise_correlation``,
``correlation_corrected_sharpe``, ``bootstrap_sharpe_ci``,
``annualization_factor``) — DO NOT duplicate that code. The only thing
this script overrides is the ledger source: it builds the LR per-trade
ledger from the **purged** test parquet
(``data/processed/purged_split/test.parquet``) and the purged headline
(``experiments/results/canonical_purged/headline.json``).

Output:
    experiments/results/audit/sharpe_audit_purged.json

The output JSON has the standard sharpe_audit schema PLUS a
``comparison`` block with side-by-side leaky-canonical vs purged numbers
and an ``interpretation`` string quantifying the leakage attribution.

Verdict logic (per Plan 18-08):
    PASS      -- per_pair_sharpe_corrected > 0.5 AND ci_lower > 0.0
                 AND |drift_pct| < 50%
    CORRECTED -- corrected > 0.0 AND ci_lower > -0.2
    FAILED    -- otherwise (statistically indistinguishable from zero
                 or negative)

Whatever the verdict, the audit reports what it finds. No "soft fail."
This is the kill-or-confirm moment.

Implements requirement: AUDIT-07.
"""
# AI-assisted authorship: written with Anthropic Claude (Opus 4.7) as
# pair-programming assistant. All design decisions are the authors'.
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# REUSE the canonical audit helpers verbatim — do NOT duplicate.
from experiments.audit.audit_sharpe import (
    annualization_factor,
    avg_pairwise_correlation,
    bootstrap_sharpe_ci,
    correlation_corrected_sharpe,
    per_pair_returns,
    per_pair_sharpe_naive,
    per_trade_sharpe,
)
from experiments.run_baselines import (
    TARGET_COLUMN,
    _build_split,
    _feature_columns,
    prepare_xy,
)
from src.features.engineering import compute_derived_features
from src.models.linear_regression import LinearRegressionPredictor
from src.utils.seed import set_all_seeds


PURGED_TRAIN = Path("data/processed/purged_split/train.parquet")
PURGED_TEST = Path("data/processed/purged_split/test.parquet")
PURGED_HEADLINE = Path("experiments/results/canonical_purged/headline.json")
LEAKY_AUDIT = Path("experiments/results/audit/sharpe_audit.json")
LEAKY_HEADLINE = Path("experiments/results/canonical/headline.json")
OUT_PATH = Path("experiments/results/audit/sharpe_audit_purged.json")

HEADLINE_MODEL = "linear_regression"
N_BOOTSTRAP = 10_000
SEED = 42
THRESHOLD = 0.02
POSITION_SIZE = 100.0


def build_purged_trade_ledger() -> pd.DataFrame:
    """Build the LR per-trade ledger on the PURGED split.

    Mirrors ``audit_sharpe.build_trade_ledger`` exactly except for the
    data source: trains LR on the purged training rows, predicts on
    purged test rows, applies threshold, returns ledger with
    ``pair_id, entry_ts, entry_day, pred, actual, pnl_pp, traded``.
    """
    set_all_seeds(SEED)

    train_raw = compute_derived_features(pd.read_parquet(PURGED_TRAIN))
    test_raw = compute_derived_features(pd.read_parquet(PURGED_TEST))
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

    # Same int64-epoch-seconds convention as canonical audit (timestamp
    # column already stored as int64 epoch seconds in the parquet).
    ts_raw = test_df["timestamp"]
    if pd.api.types.is_datetime64_any_dtype(ts_raw):
        entry_ts = ts_raw.astype("int64").values // 10**9
    else:
        ts_int = ts_raw.astype("int64").values
        if ts_int.max() > 10**12:
            entry_ts = ts_int // 10**9
        else:
            entry_ts = ts_int

    ledger = pd.DataFrame(
        {
            "pair_id": test_df["pair_id"].values,
            "entry_ts": entry_ts,
            "pred": preds,
            "actual": actuals,
            "pnl_pp": pnl_pp,
            "traded": traded,
        }
    )
    ledger["entry_day"] = ledger["entry_ts"] // 86400
    return ledger.loc[ledger["traded"]].copy()


def _drift_pct(purged: float, canonical: float) -> float:
    """Signed percentage change from canonical to purged.

    Returns +inf if canonical is exactly 0 (avoids ZeroDivisionError);
    NEGATIVE values mean the leakage correction reduced the metric (the
    expected direction).
    """
    if canonical == 0:
        return float("inf")
    return (purged - canonical) / abs(canonical) * 100.0


def _verdict_logic(
    pp_sharpe_corrected: float,
    ci_lower: float,
    drift_pct_corrected: float,
) -> str:
    """Map (corrected Sharpe, CI lower, drift) to PASS/CORRECTED/FAILED."""
    # PASS: leakage-free corrected Sharpe is healthy AND the leakage
    # correction did not collapse the headline by more than half.
    if (
        pp_sharpe_corrected > 0.5
        and ci_lower > 0.0
        and abs(drift_pct_corrected) < 50.0
    ):
        return "PASS"
    # CORRECTED: still positive and not statistically indistinguishable
    # from zero, but the leakage correction moved the headline materially.
    if pp_sharpe_corrected > 0.0 and ci_lower > -0.2:
        return "CORRECTED"
    return "FAILED"


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ---- Build purged ledger and run the audit machinery ----
    ledger = build_purged_trade_ledger()
    pt_sharpe = per_trade_sharpe(ledger["pnl_pp"].to_numpy())

    pair_ret = per_pair_returns(ledger)
    pp_sharpe_naive = per_pair_sharpe_naive(pair_ret)

    avg_corr, n_pairs_compared = avg_pairwise_correlation(ledger)
    pp_sharpe_corrected, n_eff = correlation_corrected_sharpe(
        pp_sharpe_naive, len(pair_ret), avg_corr
    )

    ci_low, ci_high = bootstrap_sharpe_ci(pair_ret)
    ann = annualization_factor(ledger)
    pp_sharpe_annualized_naive = pp_sharpe_naive * ann["annualization_factor"]
    pp_sharpe_annualized_corrected = (
        pp_sharpe_corrected * ann["annualization_factor"]
    )

    # ---- Load purged headline (single source of truth for purged P&L) ----
    purged_headline = json.loads(PURGED_HEADLINE.read_text())
    purged_lr = purged_headline["models"][HEADLINE_MODEL]
    purged_pt_canonical_field = purged_lr["sharpe_per_trade"]
    pt_drift = abs(pt_sharpe - purged_pt_canonical_field)

    # ---- Load leaky canonical audit + headline for comparison block ----
    leaky_audit = json.loads(LEAKY_AUDIT.read_text())
    leaky_headline = json.loads(LEAKY_HEADLINE.read_text())
    leaky_lr = leaky_headline["models"][HEADLINE_MODEL]

    canonical_pt = leaky_audit["per_trade_sharpe_recomputed"]
    canonical_pp_naive = leaky_audit["per_pair_sharpe_naive"]
    canonical_pp_corrected = leaky_audit["per_pair_sharpe_corr_corrected"]
    canonical_pnl = leaky_lr["total_pnl"]
    canonical_n_pairs = leaky_audit["n_pairs_compared"]

    purged_pnl = purged_lr["total_pnl"]
    purged_n_pairs = int(ledger["pair_id"].nunique())

    drift_pt = _drift_pct(pt_sharpe, canonical_pt)
    drift_pp_naive = _drift_pct(pp_sharpe_naive, canonical_pp_naive)
    drift_pp_corrected = _drift_pct(pp_sharpe_corrected, canonical_pp_corrected)
    drift_pnl = _drift_pct(purged_pnl, canonical_pnl)

    pct_retained_corrected = (
        100.0
        if canonical_pp_corrected == 0
        else (pp_sharpe_corrected / canonical_pp_corrected) * 100.0
    )
    leakage_attribution_pct = 100.0 - pct_retained_corrected
    interpretation = (
        f"Purged corrected Sharpe is {pct_retained_corrected:.1f}% of leaky "
        f"corrected Sharpe ({pp_sharpe_corrected:.4f} vs "
        f"{canonical_pp_corrected:.4f}); "
        f"{leakage_attribution_pct:+.1f}% of apparent edge attributable to "
        "leakage (positive = leakage inflated; negative = leakage masked "
        "real edge). Leaky-canonical avg pairwise correlation was "
        f"{leaky_audit['avg_pairwise_corr']:.4f} on N={canonical_n_pairs} "
        f"contemporaneous-day pair-pairs; purged avg pairwise correlation "
        f"is {avg_corr:.4f} on N={n_pairs_compared}."
    )

    verdict = _verdict_logic(pp_sharpe_corrected, ci_low, drift_pp_corrected)

    out = {
        "audit": "sharpe_purged",
        "tier": 1,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "split_type": "pair_stratified_80_20",
        "purged_headline_input": PURGED_HEADLINE.as_posix(),
        "leaky_audit_input": LEAKY_AUDIT.as_posix(),
        "model_audited": HEADLINE_MODEL,
        # Per-trade
        "per_trade_sharpe_recomputed": pt_sharpe,
        "per_trade_sharpe_purged_headline": purged_pt_canonical_field,
        "per_trade_sharpe_drift_vs_purged_headline": pt_drift,
        # Per-pair
        "per_pair_sharpe_naive": pp_sharpe_naive,
        "per_pair_sharpe_naive_ci_95": [ci_low, ci_high],
        "per_pair_sharpe_corr_corrected": pp_sharpe_corrected,
        "avg_pairwise_corr": avg_corr,
        "n_pairs_compared": n_pairs_compared,
        "n_eff": n_eff,
        # Annualized
        "annualization": ann,
        "per_pair_sharpe_annualized_naive": pp_sharpe_annualized_naive,
        "per_pair_sharpe_annualized_corrected": pp_sharpe_annualized_corrected,
        # Comparison block (the load-bearing addition vs canonical audit)
        "comparison": {
            "canonical_sharpe_per_trade": canonical_pt,
            "purged_sharpe_per_trade": pt_sharpe,
            "delta_per_trade": pt_sharpe - canonical_pt,
            "drift_pct_per_trade": drift_pt,
            "canonical_sharpe_per_pair_naive": canonical_pp_naive,
            "purged_sharpe_per_pair_naive": pp_sharpe_naive,
            "delta_per_pair_naive": pp_sharpe_naive - canonical_pp_naive,
            "drift_pct_per_pair_naive": drift_pp_naive,
            "canonical_sharpe_per_pair_corrected": canonical_pp_corrected,
            "purged_sharpe_per_pair_corrected": pp_sharpe_corrected,
            "delta_per_pair_corrected": pp_sharpe_corrected
            - canonical_pp_corrected,
            "drift_pct_per_pair_corrected": drift_pp_corrected,
            "canonical_total_pnl": canonical_pnl,
            "purged_total_pnl": purged_pnl,
            "delta_total_pnl": purged_pnl - canonical_pnl,
            "drift_pct_total_pnl": drift_pnl,
            "canonical_n_pairs": canonical_n_pairs,
            "purged_n_pairs": purged_n_pairs,
            "interpretation": interpretation,
        },
        "verdict_logic": {
            "PASS": (
                "corrected > 0.5 AND CI_lower > 0.0 AND |drift_pct| < 50%"
            ),
            "CORRECTED": "corrected > 0.0 AND CI_lower > -0.2",
            "FAILED": "otherwise (statistically indistinguishable from zero)",
        },
        "assumptions": [
            "Purged split is pair-atomic by construction "
            "(see data/processed/purged_split/split_metadata.json) — every "
            "pair appears entirely in train OR entirely in test, never both.",
            "Per-pair returns are stationary (no regime change within "
            "purged test window).",
            "Pair-correlation effective-sample correction follows Bailey-"
            "Lopez de Prado (2012) framework: n_eff = N / (1 + (N-1) * "
            "avg_corr); sharpe_corrected = sharpe_naive * sqrt(n_eff/N).",
            f"Annualization uses pairs_per_year = {ann['pairs_per_year']}, "
            f"derived from test_span_days = {ann['test_span_days']} and "
            f"N_pairs = {ann['n_pairs']}. Assumes pair-lifecycle "
            "distribution in purged test window is representative of "
            "annual operation (likely violated; flag for paper §6.4).",
            "Bootstrap CI uses simple resample-with-replacement (10,000 "
            "resamples). Does NOT correct for autocorrelation in per-pair "
            "returns.",
            "LR is the headline model audited (per Phase 17 conclusion: "
            "LR wins 4 of 5 metrics in canonical/headline.json). XGBoost "
            "purged numbers are in canonical_purged/headline.json but are "
            "not the load-bearing audit target.",
            "Comparison delta interpretation: NEGATIVE drift means the "
            "leakage correction reduced the metric (expected direction "
            "for inflated metrics); POSITIVE drift means the leakage "
            "correction increased it (suggests the leakage was masking "
            "real edge OR the purged test set is a more favorable sample).",
        ],
        "n_bootstrap": N_BOOTSTRAP,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))

    # ---- Console summary ----
    print(f"Wrote {OUT_PATH} verdict={verdict}")
    print()
    print("=" * 70)
    print(f"PURGED Tier 1 audit -- verdict {verdict}")
    print("=" * 70)
    print(
        f"Per-trade Sharpe          : leaky {canonical_pt:>+8.4f} -> "
        f"purged {pt_sharpe:>+8.4f}  (drift {drift_pt:+.2f}%)"
    )
    print(
        f"Per-pair Sharpe naive     : leaky {canonical_pp_naive:>+8.4f} -> "
        f"purged {pp_sharpe_naive:>+8.4f}  (drift {drift_pp_naive:+.2f}%)"
    )
    print(
        f"Per-pair Sharpe corrected : leaky {canonical_pp_corrected:>+8.4f} -> "
        f"purged {pp_sharpe_corrected:>+8.4f}  (drift {drift_pp_corrected:+.2f}%)"
    )
    print(
        f"Per-pair CI 95%           : [{ci_low:+.4f}, {ci_high:+.4f}]"
    )
    print(
        f"Avg pairwise corr         : leaky {leaky_audit['avg_pairwise_corr']:>+8.4f} -> "
        f"purged {avg_corr:>+8.4f}"
    )
    print(
        f"N pairs (effective)       : leaky {canonical_n_pairs} -> "
        f"purged {n_pairs_compared} (n_eff={n_eff:.2f})"
    )
    print(
        f"Total P&L                 : leaky ${canonical_pnl:>+9.2f} -> "
        f"purged ${purged_pnl:>+9.2f}  (drift {drift_pnl:+.2f}%)"
    )
    print()
    print(f"Interpretation: {interpretation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
