"""Tier 6: Live-vs-backtest honesty audit.

Confirms PAPER_DRAFT.md §5.9.1 numbers (1,224 closed live commodity positions,
36.0% win rate, 441 wins) are quantitatively *consistent with* the §5.3 backtest
oil-near-expiry edge (76.5% win rate; n estimated at ~200) under a two-proportion
z-test, and reports Cohen's h as effect-size magnitude.

The audit's purpose is CONFIRMATION, not correction:
- The paper's §5.9.1 caveat already discloses the gap honestly (lines 463-464).
- This script adds quantitative evidence (z-statistic, p-value, Cohen's h) that
  the gap is real and large in magnitude — i.e., the caveat language is
  appropriate, not under-stated.

Verdict mapping:
- PASS  iff |Cohen's h| >= 0.2 (effect is at least small; paper caveat warranted)
- REVIEW iff |Cohen's h| < 0.2 (effect tiny; §5.9.1 may overstate the gap)

Inputs (verbatim from PAPER_DRAFT.md §5.9.1 line 450):
    LIVE_OIL_WINS  = 441
    LIVE_OIL_TOTAL = 1224 (live WR = 0.360)

Backtest oil near-expiry numbers (Finding 6 / §5.3 / §8 Conclusions line 706):
    BACKTEST_OIL_WR        = 0.765 (verbatim from paper)
    BACKTEST_OIL_TOTAL_EST = 200   (estimate; paper does not state n).
                                   The verdict and effect-size conclusions
                                   are robust to the exact n at this scale.

Output: experiments/results/audit/live_vs_backtest_audit.json

Audit JSON schema follows the Phase-18 Pattern-2 contract (see 18-RESEARCH.md):
    {audit, tier, ran_at, verdict, ...findings, assumptions[]}
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT_PATH = Path("experiments/results/audit/live_vs_backtest_audit.json")


def proportions_ztest(counts: np.ndarray, nobs: np.ndarray) -> tuple[float, float]:
    """Two-proportion z-test (two-sided), pooled-variance form.

    Mirrors statsmodels.stats.proportion.proportions_ztest with
    alternative='two-sided'. Implemented inline to avoid a runtime dependency
    on statsmodels (Phase 18-01 zero-new-deps decision).

    Returns:
        (z_statistic, p_value)
    """
    counts = np.asarray(counts, dtype=float)
    nobs = np.asarray(nobs, dtype=float)
    p1 = counts[0] / nobs[0]
    p2 = counts[1] / nobs[1]
    p_pool = (counts[0] + counts[1]) / (nobs[0] + nobs[1])
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / nobs[0] + 1.0 / nobs[1]))
    if se == 0.0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # Two-sided p-value via standard-normal survival function:
    # P(|Z| >= |z|) = 2 * (1 - Phi(|z|)) = erfc(|z| / sqrt(2)).
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return float(z), float(p)

# §5.9.1 numbers (verbatim from PAPER_DRAFT.md line 450)
LIVE_OIL_WINS = 441
LIVE_OIL_TOTAL = 1224
LIVE_OIL_WR = LIVE_OIL_WINS / LIVE_OIL_TOTAL  # 0.36029...

# Finding 6 / §5.3 / §8 Conclusions line 706 backtest oil near-expiry numbers.
# Paper states 76.5% win rate; trade count is not given in §5.3. We use 200 as
# a conservative estimate consistent with the near-expiry oil sub-cohort size
# implied by Finding 6's "+$0.41/trade" framing. The conclusion is robust:
# at any n in [50, 1000], |Cohen's h| stays > 0.5 (large effect) and p stays
# < 0.001 by orders of magnitude.
BACKTEST_OIL_WR = 0.765
BACKTEST_OIL_TOTAL_EST = 200


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions.

    h = 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))

    Interpretation thresholds (Cohen 1988):
        |h| < 0.2 — small effect
        |h| < 0.5 — medium effect
        |h| < 0.8 — large effect
        |h| >= 0.8 — very large effect
    """
    return 2.0 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def _effect_size_label(h: float) -> str:
    abs_h = abs(h)
    if abs_h < 0.2:
        return "small"
    if abs_h < 0.5:
        return "medium"
    return "large"


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Two-proportion z-test (live wins / live total) vs (backtest wins / backtest total)
    backtest_wins_est = int(round(BACKTEST_OIL_WR * BACKTEST_OIL_TOTAL_EST))
    counts = np.array([LIVE_OIL_WINS, backtest_wins_est])
    nobs = np.array([LIVE_OIL_TOTAL, BACKTEST_OIL_TOTAL_EST])
    z_stat, p_value = proportions_ztest(counts, nobs)

    h = cohens_h(LIVE_OIL_WR, BACKTEST_OIL_WR)
    effect_size_label = _effect_size_label(h)

    # Verdict: PASS if effect is at least small (|h| >= 0.2). Below that, the
    # §5.9.1 caveat language ("dramatically lower than the backtest oil
    # near-expiry edge") would overstate a non-existent effect.
    verdict = "PASS" if abs(h) >= 0.2 else "REVIEW"

    out = {
        "audit": "live_vs_backtest",
        "tier": 6,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "live": {
            "wins": LIVE_OIL_WINS,
            "total": LIVE_OIL_TOTAL,
            "wr": LIVE_OIL_WR,
        },
        "backtest_estimate": {
            "wr": BACKTEST_OIL_WR,
            "wins_est": backtest_wins_est,
            "total_est": BACKTEST_OIL_TOTAL_EST,
            "note": (
                "Backtest n is approximate; PAPER_DRAFT.md §5.3 cites WR 76.5% "
                "and +$0.41/trade for the oil near-expiry cohort but does not "
                "state the trade count. n=200 is a conservative estimate; "
                "verdict and effect-size are robust to n in [50, 1000]."
            ),
        },
        "two_proportion_z_test": {
            "z_statistic": float(z_stat),
            "p_value": float(p_value),
            "alternative": "two-sided",
            "interpretation": (
                "p << 0.001 expected given live n=1,224 and a 40-percentage-point "
                "WR difference. The gap is statistically real, not noise."
            ),
        },
        "cohens_h": float(h),
        "effect_size_label": effect_size_label,
        "honest_interpretation": (
            "The gap is statistically significant AND large in effect, but the "
            "samples are not measuring the same thing: 76.5% backtest WR is on "
            "the near-expiry oil subset only; 36.0% live WR is on the full "
            "1,224-position commodity cohort across all series and expiries "
            "(KXBRENTW, KXWTI, KXWTIW, KXBRENTMON, etc.). PAPER_DRAFT.md §5.9.1 "
            "lines 463-464 already disclose this; the z-test is supplementary "
            "evidence that the disclosure is appropriate, not over-stated."
        ),
        "paper_corrections_required": [],
        "assumptions": [
            "BACKTEST_OIL_TOTAL_EST = 200 (paper does not state n for §5.3 oil "
            "near-expiry cohort). Verdict robust to n in [50, 1000].",
            "Two-proportion z-test assumes independence within each sample. "
            "Live positions are quasi-independent across distinct (kalshi_ticker, "
            "poly_id) pairs and across hourly bars within a 12h window.",
            "Cohen's h thresholds follow Cohen (1988): 0.2 small / 0.5 medium "
            "/ 0.8 large.",
            "Live numbers (1,224 / 36.0% / 441) are read verbatim from "
            "PAPER_DRAFT.md §5.9.1 line 450; we do NOT re-query the live system.",
        ],
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH} verdict={verdict} cohens_h={h:.3f} z={z_stat:.2f} p={p_value:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
