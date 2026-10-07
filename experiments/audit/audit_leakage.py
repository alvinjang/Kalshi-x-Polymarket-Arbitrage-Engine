"""Tier 2: Leakage / look-ahead bias audit.

(a) Per-feature classification of src/features/engineering.py.
(b) Walk-forward embargo verification.
(c) Quality-filter rule-by-rule retroactive-info audit.

Verdict logic:
    FAILED if  n_leaking > 0
            OR walk_forward_embargo.n_embargo_violations > 0
            OR n_qf_retroactive > 0
    PASS otherwise.

Suspicious findings do NOT trigger FAILED — they require manual review only.
This script is the source of truth for Tier 2 of Phase 18.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6)
# as pair-programming assistant. All design decisions and interpretations are
# the authors'.
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OUT_PATH = Path("experiments/results/audit/leakage_audit.json")
ENG_PATH = Path("src/features/engineering.py")
QF_PATH = Path("src/matching/quality_filter.py")
ALIGNED = Path("data/processed/aligned_pairs.parquet")
TRAIN = Path("data/processed/train.parquet")
TEST = Path("data/processed/test.parquet")

# Embargo policy: a bridging pair is a violation if the gap between its last
# train timestamp and first test timestamp is < 1 day (86400 seconds).
EMBARGO_SECONDS = 86400

# Patterns that indicate look-ahead (LEAKING) or rolling-with-future-data
# (SUSPICIOUS). Regex order matters: LEAK_PATTERNS are checked first; if any
# match, verdict = "Leaking" and SUSPICIOUS_PATTERNS are skipped.
LEAK_PATTERNS = [
    (r"center\s*=\s*True", "rolling_center_true"),
    (r"\.shift\s*\(\s*-\d", "negative_shift"),  # df.shift(-1) leaks future
    (r"bfill|backfill", "backward_fill"),
    (r"fillna\(method\s*=\s*['\"]bfill", "bfill_fillna"),
]
# Patterns that are fine but worth flagging for manual review.
SUSPICIOUS_PATTERNS = [
    (r"\.rolling\(", "rolling_window_endpoint_check_required"),
    (r"\.expanding\(", "expanding_window_endpoint_check_required"),
    (r"\.transform\(lambda", "transform_lambda_review"),
]


def classify_features(eng_src: str) -> list[dict]:
    """Walk source code line-by-line; classify any line assigning to result["<col>"].

    For each ``result["<feat>"] = ...`` assignment, scan the assignment line
    plus the next 4 lines for known leak/suspicious patterns. Returns one
    finding dict per feature with keys:

        feature        — the column name
        line           — 1-indexed source line number
        verdict        — "Safe" | "Suspicious" | "Leaking"
        evidence       — list of pattern names that matched
        code_snippet   — the source line (stripped)
    """
    findings: list[dict] = []
    lines = eng_src.splitlines()
    feature_def_re = re.compile(r'result\["([^"]+)"\]\s*=')
    for i, line in enumerate(lines, start=1):
        m = feature_def_re.search(line)
        if not m:
            continue
        feat = m.group(1)
        # Look at this line + 4 lines context for leak/suspicious patterns.
        ctx = "\n".join(lines[max(0, i - 1) : min(len(lines), i + 4)])
        verdict = "Safe"
        evidence: list[str] = []
        for pattern, name in LEAK_PATTERNS:
            if re.search(pattern, ctx):
                verdict = "Leaking"
                evidence.append(name)
        if verdict == "Safe":
            for pattern, name in SUSPICIOUS_PATTERNS:
                if re.search(pattern, ctx):
                    verdict = "Suspicious"
                    evidence.append(name)
        findings.append(
            {
                "feature": feat,
                "line": i,
                "verdict": verdict,
                "evidence": evidence,
                "code_snippet": line.strip(),
            }
        )
    return findings


def _gap_seconds(train_end_ts, test_start_ts) -> float:
    """Return gap in seconds between train_end and test_start.

    Handles both pandas datetime dtypes and plain integer epoch-seconds.
    The canonical pipeline writes int64 epoch seconds (verified 2026-04-25),
    but we tolerate datetime types defensively in case the schema migrates.
    """
    diff = test_start_ts - train_end_ts
    if hasattr(diff, "total_seconds"):
        return float(diff.total_seconds())
    return float(diff)


def audit_walk_forward_embargo() -> dict:
    """Check if any pair_id has rows in BOTH train.parquet AND test.parquet.

    Per RESEARCH.md, the canonical pipeline splits 80/20 by row index, not by
    pair lifecycle. If a pair appears in both partitions, its features in
    train rows directly inform its target in test rows — that is the textbook
    walk-forward embargo violation. This audit reports it explicitly.
    """
    train = pd.read_parquet(TRAIN, columns=["pair_id", "timestamp"])
    test = pd.read_parquet(TEST, columns=["pair_id", "timestamp"])
    train_pairs = set(train["pair_id"].unique())
    test_pairs = set(test["pair_id"].unique())
    bridging = train_pairs & test_pairs

    train_end = train.groupby("pair_id")["timestamp"].max()
    test_start = test.groupby("pair_id")["timestamp"].min()

    embargo_violations: list[dict] = []
    for pid in bridging:
        gap = _gap_seconds(train_end.loc[pid], test_start.loc[pid])
        if gap < EMBARGO_SECONDS:
            embargo_violations.append(
                {
                    "pair_id": pid,
                    "train_end": str(train_end.loc[pid]),
                    "test_start": str(test_start.loc[pid]),
                    "gap_seconds": gap,
                    "gap_hours": round(gap / 3600.0, 2),
                }
            )

    return {
        "n_train_pairs": len(train_pairs),
        "n_test_pairs": len(test_pairs),
        "n_bridging_pairs": len(bridging),
        "n_embargo_violations": len(embargo_violations),
        "embargo_seconds": EMBARGO_SECONDS,
        "violations_sample": embargo_violations[:10],
    }


def audit_quality_filter() -> list[dict]:
    """Inspect each rule in src/matching/quality_filter.py for retroactive-info usage.

    Hand-curated manifest (verbatim from 18-RESEARCH.md). The rule
    ``stale_ticker`` calls ``_current_year()`` at runtime, which technically
    makes its behavior depend on the audit-execution clock. This is benign
    for a 2026-test-data audit run in 2026, but flagged for transparency.
    """
    rules = [
        {
            "rule": "MIN_CONFIDENCE",
            "uses": ["confidence_score"],
            "retroactive": False,
            "evidence": "confidence is a pre-trade match score; not outcome-aware",
        },
        {
            "rule": "MAX_RESOLUTION_GAP_DAYS (rule 2)",
            "uses": [
                "kalshi_resolution_date",
                "polymarket_resolution_date",
            ],
            "retroactive": False,
            "evidence": (
                "resolution DATE is a contract metadata field, not the OUTCOME. "
                "Known at listing."
            ),
        },
        {
            "rule": "directions_compatible (rule 3)",
            "uses": ["question text"],
            "retroactive": False,
            "evidence": "question text is fixed at contract listing",
        },
        {
            "rule": "thresholds_compatible (rule 4)",
            "uses": ["question text"],
            "retroactive": False,
            "evidence": "thresholds are fixed at listing",
        },
        {
            "rule": "Rule 1 season-wins vs champion",
            "uses": ["ticker prefix", "title keywords"],
            "retroactive": False,
            "evidence": "ticker + title fixed at listing",
        },
        {
            "rule": "Rule 2 Fed year/month mismatch",
            "uses": ["ticker date encoding", "title month/year"],
            "retroactive": False,
            "evidence": "Fed contract dates are fixed at listing",
        },
        {
            "rule": "Rule 3 cabinet vs nomination",
            "uses": ["title keywords"],
            "retroactive": False,
            "evidence": "title keywords fixed at listing",
        },
        {
            "rule": "Rule 3b threshold vs ranking",
            "uses": ["ticker structure", "poly title"],
            "retroactive": False,
            "evidence": "structural at listing",
        },
        {
            "rule": "Rule 3c threshold vs policy",
            "uses": ["title keywords"],
            "retroactive": False,
            "evidence": "structural",
        },
        {
            "rule": "Rule 3d AAA gas geography",
            "uses": ["ticker suffix", "poly title geography"],
            "retroactive": False,
            "evidence": "structural",
        },
        {
            "rule": "Rule stale_ticker",
            "uses": ["ticker year vs current year"],
            "retroactive": True,
            "evidence": (
                "FLAG: uses _current_year() at audit time, NOT at backtest "
                "time. If audit runs on 2026-04-25 and backtest data is from "
                "2026-01, a 2026 ticker passing the stale_ticker rule today "
                "would also have passed in January. Likely benign because "
                "rejection is a coarse 'past year' check, but MUST be "
                "verified with timestamp-aware version."
            ),
        },
        {
            "rule": "Rule 10 asset-class consistency",
            "uses": ["Kalshi ticker prefix", "title tokens"],
            "retroactive": False,
            "evidence": "asset class fixed at listing",
        },
    ]
    return rules


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    feature_findings = classify_features(ENG_PATH.read_text())
    embargo = audit_walk_forward_embargo()
    qf_findings = audit_quality_filter()

    n_leaking = sum(1 for f in feature_findings if f["verdict"] == "Leaking")
    n_suspicious = sum(1 for f in feature_findings if f["verdict"] == "Suspicious")
    n_qf_retro = sum(1 for r in qf_findings if r["retroactive"])

    verdict = "PASS"
    if (
        n_leaking > 0
        or embargo["n_embargo_violations"] > 0
        or n_qf_retro > 0
    ):
        verdict = "FAILED"

    out = {
        "audit": "leakage",
        "tier": 2,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "feature_classification": feature_findings,
        "n_features_classified": len(feature_findings),
        "n_leaking": n_leaking,
        "n_suspicious": n_suspicious,
        "walk_forward_embargo": embargo,
        "quality_filter_rules": qf_findings,
        "n_qf_retroactive": n_qf_retro,
        "assumptions": [
            "Train/test split in canonical pipeline is row-based 80/20, not pair-based.",
            "Suspicious-pattern flagging is regex-based; manual review required for "
            "each Suspicious entry to confirm rolling endpoint <= entry_ts.",
            "Quality-filter rule-by-rule analysis is hand-curated; if new rules are "
            "added in src/matching/quality_filter.py, this list must be updated.",
            "Timestamp column in train.parquet / test.parquet is int64 epoch "
            "seconds; embargo-gap computation uses plain integer subtraction.",
            "Embargo policy: gap_seconds < 86400 (1 day) constitutes a violation; "
            "longer embargoes (e.g. 7 days for prediction-market positions) would "
            "tighten the audit further.",
        ],
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH} verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
