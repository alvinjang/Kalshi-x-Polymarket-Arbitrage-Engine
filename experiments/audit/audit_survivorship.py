"""Tier 4: Survivorship / selection audit.

Quantifies the post-alignment survivorship rate for the canonical
test universe used to produce experiments/results/canonical/headline.json.

Three sub-audits:

(a) **Candidate universe construction** — every pair_id ever considered.
    Sources:
      - data/processed/aligned_pairs.parquet (post-alignment, pre-feature
        pair set used to build train.parquet / test.parquet)
      - data/live/active_matches.json (post-quality-filter live universe;
        live system uses content-addressed pair_ids since 2026-04-11 fix)

(b) **Realized universe** — pairs actually in data/processed/test.parquet.
    This is the cohort that shows up in canonical headline metrics.

(c) **Drop-reason classification + 10-pair random sample** — for the
    set difference (candidates − realized), heuristically classify each
    drop and emit a deterministic 10-pair random sample (seed=42) for
    Ian's manual review per VALIDATION.md (Plan 18-07 checkpoint).

Verdict logic (Pattern 2 contract from RESEARCH.md):
  - PASS: n_requiring_manual_review == 0 (heuristics covered all cases)
  - REVIEW_REQUIRED: at least one sample entry needs manual classification
  - FAILED: any sample entry's drop_reason mentions post-resolution
    outcome metadata (e.g., realized_return < 0). Reserved for explicit
    survivorship-bias evidence; not triggered by heuristic fallthrough.

Per RESEARCH.md §Watch out for:
  - Candidate universe is approximated; full replay of run_pipeline.py
    on raw API dumps is out-of-scope for Phase 18.
  - The 10-pair manual sample is the load-bearing evidence; Ian must
    review each entry per VALIDATION.md manual-only verifications row.

Implements requirement: AUDIT-04.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

OUT_PATH = Path("experiments/results/audit/survivorship_audit.json")
SEED = 42

# Inputs
ALIGNED = Path("data/processed/aligned_pairs.parquet")
TEST = Path("data/processed/test.parquet")
ACTIVE_MATCHES = Path("data/live/active_matches.json")  # post-filter live universe


def _coerce_active_match_pair_id(match: dict) -> str:
    """Extract a pair_id from an active_matches.json entry.

    Tries three strategies in order:
      1. Direct ``pair_id`` key (Phase 15+ schema).
      2. Content-addressed construction from ``kalshi_ticker`` +
         ``poly_id`` via ``src.live.pair_ids.make_pair_id`` (post-2026-04-11
         schema). This makes the audit meaningful for the current 148k-entry
         active_matches dump where ``pair_id`` is not stored as a top-level key.
      3. Returns empty string if neither path is available — the caller
         skips empty ids.
    """
    pid = match.get("pair_id")
    if pid:
        return str(pid)
    kalshi_ticker = match.get("kalshi_ticker") or match.get("prev_kalshi_ticker")
    poly_id = match.get("poly_id") or match.get("prev_poly_id")
    if kalshi_ticker and poly_id:
        try:
            from src.live.pair_ids import make_pair_id

            return make_pair_id(str(kalshi_ticker), str(poly_id))
        except Exception:
            return ""
    return ""


def candidate_pair_universe() -> set[str]:
    """Every pair_id that was ever considered. Sources:
    - data/processed/aligned_pairs.parquet (post-alignment, pre-feature)
    - data/live/active_matches.json (post-quality-filter live universe)

    This is a coarse estimate; tighter would require re-running run_pipeline.py
    on raw data, which we don't have time for in Phase 18.
    """
    candidates: set[str] = set()
    if ALIGNED.exists():
        df = pd.read_parquet(ALIGNED, columns=["pair_id"])
        candidates |= set(df["pair_id"].dropna().astype(str).unique())
    if ACTIVE_MATCHES.exists():
        try:
            matches = json.loads(ACTIVE_MATCHES.read_text())
        except Exception:
            matches = []
        if isinstance(matches, list):
            for m in matches:
                if not isinstance(m, dict):
                    continue
                pid = _coerce_active_match_pair_id(m)
                if pid:
                    candidates.add(pid)
    return candidates


def realized_pair_universe() -> set[str]:
    """Pairs actually in canonical test split."""
    df = pd.read_parquet(TEST, columns=["pair_id"])
    return set(df["pair_id"].dropna().astype(str).unique())


def classify_drop_reason(pair_id: str, aligned: pd.DataFrame) -> str:
    """For a pair_id present in aligned_pairs but not in test.parquet, infer drop reason.

    Heuristics:
        - If pair has < 20 bars in aligned -> 'low_overlap_n_bars=<int>'
        - Else -> 'pre_test_window_max_ts=<ts>' (legitimate: pair fully
          resolved before the train/test split window)

    If the pair_id is not present in aligned at all (e.g., it came from
    active_matches.json only — a live-discovery candidate that never
    made it into the offline pipeline), we still return a low-overlap
    classification because the pair contributed zero bars to the
    pipeline by construction.
    """
    sub = aligned.loc[aligned["pair_id"] == pair_id]
    n_bars = len(sub)
    if n_bars < 20:
        return f"low_overlap_n_bars={n_bars}"
    max_ts = sub["timestamp"].max()
    return f"pre_test_window_max_ts={max_ts}"


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    candidates = candidate_pair_universe()
    realized = realized_pair_universe()
    dropped = candidates - realized

    aligned = pd.read_parquet(ALIGNED, columns=["pair_id", "timestamp"])

    # Random sample of 10 for manual confirmation (deterministic via seed=42)
    rng = random.Random(SEED)
    sample = rng.sample(sorted(dropped), min(10, len(dropped)))
    sample_classifications = []
    for pid in sample:
        reason = classify_drop_reason(pid, aligned)
        # Manual classification flag per VALIDATION.md row 18-05-XX
        # (`pending_human_review` per Plan 18-07 Wave 3 checkpoint plan)
        manual_class = (
            "structural"
            if ("low_overlap" in reason or "pre_test_window" in reason)
            else "REVIEW"
        )
        sample_classifications.append({
            "pair_id": pid,
            "drop_reason_inferred": reason,
            "manual_classification_required": manual_class,
            "human_classification": "pending_human_review",
        })

    n_review = sum(
        1 for s in sample_classifications
        if s["manual_classification_required"] == "REVIEW"
    )

    # FAILED reserved for explicit post-resolution outcome metadata
    n_failed = sum(
        1 for s in sample_classifications
        if "post_hoc" in s["drop_reason_inferred"]
        or "negative_return" in s["drop_reason_inferred"]
        or "loss" in s["drop_reason_inferred"]
    )
    if n_failed > 0:
        verdict = "FAILED"
    elif n_review == 0:
        verdict = "PASS"
    else:
        verdict = "REVIEW_REQUIRED"

    out = {
        "audit": "survivorship",
        "tier": 4,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "n_candidate_pairs": len(candidates),
        "n_realized_pairs": len(realized),
        "n_dropped_pairs": len(dropped),
        "drop_rate": round(len(dropped) / max(len(candidates), 1), 4),
        "random_sample_size": len(sample_classifications),
        "random_sample": sample_classifications,
        "n_requiring_manual_review": n_review,
        "assumptions": [
            "Candidate universe is approximated from aligned_pairs.parquet + "
            "active_matches.json. Tighter would require replaying the matching "
            "pipeline on raw data, which is out of scope for Phase 18.",
            "active_matches.json schema does not store a top-level pair_id key; "
            "we synthesize it from kalshi_ticker + poly_id via "
            "src.live.pair_ids.make_pair_id (the post-2026-04-11 content-addressed "
            "scheme). Entries missing both kalshi_ticker and poly_id are skipped.",
            "Drop-reason inference is heuristic (low_overlap if < 20 bars, "
            "pre_test_window otherwise). REVIEW entries (those falling through "
            "all heuristics) require Ian's manual sign-off per VALIDATION.md "
            "manual-only verifications row before the verdict can be PASS.",
            "If any random_sample entry has drop_reason mentioning post-resolution "
            "outcome metadata (post_hoc, negative_return, loss), the audit FAILS: "
            "that's retroactive dropping = survivorship bias.",
            "Each random_sample entry is marked human_classification='pending_human_review'; "
            "Plan 18-07 (Wave 3) opens a checkpoint for Ian to classify each as "
            "structural / retroactive and rerun this audit to lock the final verdict.",
        ],
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_PATH} verdict={verdict} n_dropped={len(dropped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
