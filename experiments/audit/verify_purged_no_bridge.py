"""Sanity rerun: confirm purged split has zero embargo violations.

The whole point of Plan 18-08 is to eliminate the 142 embargo violations
the canonical 80/20 row-index split exhibited (per
``experiments/results/audit/leakage_audit.json``). This script independently
re-verifies that property on the purged parquets and writes a tiny JSON
artifact so the result is auditable from the SUMMARY.

If ANY pair bridges the purged train/test boundary, this script raises.
That would mean the split builder is broken and the entire 18-08
recompute is invalid.

Output:
    experiments/results/audit/leakage_audit_purged_check.json

Usage:
    PYTHONPATH=. python experiments/audit/verify_purged_no_bridge.py

Implements requirement: AUDIT-07 (final-step verification).
"""
# AI-assisted authorship: written with Anthropic Claude (Opus 4.7) as
# pair-programming assistant. All design decisions are the authors'.
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PURGED_TRAIN = Path("data/processed/purged_split/train.parquet")
PURGED_TEST = Path("data/processed/purged_split/test.parquet")
OUT_PATH = Path(
    "experiments/results/audit/leakage_audit_purged_check.json"
)


def main() -> int:
    train = pd.read_parquet(PURGED_TRAIN)
    test = pd.read_parquet(PURGED_TEST)

    train_pairs = set(train["pair_id"])
    test_pairs = set(test["pair_id"])
    bridging = train_pairs & test_pairs

    out = {
        "audit": "leakage_purged_check",
        "tier": 2,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "split_source": PURGED_TRAIN.parent.as_posix(),
        "n_embargo_violations": 0 if not bridging else len(bridging),
        "n_bridging_pairs": len(bridging),
        "n_train_pairs": len(train_pairs),
        "n_test_pairs": len(test_pairs),
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "verdict": "PASS" if not bridging else "FAILED",
        "comparison_to_canonical": (
            "Canonical 80/20 row-index split had 142 embargo violations "
            "(see experiments/results/audit/leakage_audit.json). Purged "
            "pair-stratified split has 0 by construction (the split "
            "builder enforces train_pairs.isdisjoint(test_pairs))."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    if bridging:
        print(
            f"FAIL: {len(bridging)} pairs bridge the purged train/test "
            f"boundary."
        )
        raise SystemExit(1)

    print(
        f"Embargo: 0 violations OK ({len(train_pairs)} train pairs / "
        f"{len(test_pairs)} test pairs)"
    )
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
