"""Tier 5: build experiments/results/audit/paper_numbers.csv.

Wraps scripts/audit_paper_numbers.py; instead of producing a Markdown log,
emits one row per numeric claim in CSV format for AUDIT_REPORT.md ingest
(consumed by Plan 18-07).

Headline-section restriction is critical (Phase 17 lesson: all-sections
auditing produced 53+ false positives). This script only walks claims
inside Abstract / §5.1 / §5.8 / §6.3 / §8 Conclusions, matching the
REPL-06c precedent in scripts/check_paper.sh.

Schema (8 columns):
    claim_text, claim_value, kind, paper_section, line_number,
    source_file, source_command, match_status

`match_status` is "PENDING" by default; Plan 18-07 will overwrite values
based on regex tolerance comparison against canonical/headline.json.

Special-case row: the abstract claims annualized per-pair Sharpe ≈ 3.2.
Wave 1 Plan 18-02 reproduced naive=18.6 / corrected=7.0 from the
canonical trade ledger. The MISMATCH for this claim is recorded inline
so Plan 18-07 can resolve it in PAPER_DRAFT.md.

Usage:
    PYTHONPATH=. python3 experiments/audit/build_paper_numbers_csv.py
"""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path

OUT = Path("experiments/results/audit/paper_numbers.csv")
PAPER = Path("PAPER_DRAFT.md")
HEADLINE = Path("experiments/results/canonical/headline.json")
SHARPE_AUDIT = Path("experiments/results/audit/sharpe_audit.json")

# Regex bank (mirrors scripts/audit_paper_numbers.py for consistency).
DOLLAR_RE = re.compile(r"(?:\+|-|−)?\\?\$([0-9,]+(?:\.[0-9]+)?)")
SHARPE_RE = re.compile(r"[Ss]harpe[^0-9]{1,30}([0-9]\.[0-9]+)")
BPS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*bps")
PCT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*%")

# Tier-5 specific: catches longer-form claims like "per-pair annualized Sharpe,
# treating each of 144 pairs as one independent bet, is ≈ 3.2" where the
# sentence between "Sharpe" and the number exceeds SHARPE_RE's 30-char window.
# Used in addition to SHARPE_RE; deduplicated below.
SHARPE_LONG_RE = re.compile(
    r"[Ss]harpe[^.\n]{0,200}?(?:is|≈|approximately|of)\s*≈?\s*([0-9]\.[0-9]+)"
)

# Headline sections only — anything outside these is out-of-scope.
HEADLINE_SECTIONS = ("## Abstract", "### 5.1 ", "### 5.8 ", "### 6.3 ", "## 8.")


def in_headline_section(line_idx: int, all_lines: list[str]) -> str | None:
    """Walk backward; return the most recent headline-section heading, or None.

    Stops walking (returns None) when it encounters a non-headline `##` or
    `### N.M ` heading first, indicating the line is in a different section.
    """
    for i in range(line_idx, -1, -1):
        line = all_lines[i]
        # Match a headline section header.
        for hdr in HEADLINE_SECTIONS:
            if line.startswith(hdr):
                return line.strip()
        # If we hit a different `## ` or `### N.M ` heading first, we're outside.
        if i != line_idx and (
            re.match(r"^## [A-Za-z0-9]", line)
            or re.match(r"^### [0-9]+\.[0-9]+ ", line)
        ):
            return None
    return None


def classify_claim_value(value_str: str, kind: str) -> tuple[str, str]:
    """Heuristic: tag the source_file + source_command for each (kind, value).

    Returns (source_file, source_command). Defaults to canonical/headline.json
    + experiments/run_canonical.py. Special cases handled inline:
      - The "≈ 3.2" per-pair Sharpe claim points to sharpe_audit.json
        because Phase 18-02 reproduced naive=18.6 / corrected=7.0,
        not 3.2 (Mismatch flagged downstream).
    """
    # Default: every paper number traces to the canonical headline.json.
    src_file = "experiments/results/canonical/headline.json"
    src_cmd = "python3 experiments/run_canonical.py"

    # Special-case: per-pair annualized Sharpe ≈ 3.2 doesn't match Wave 1 audit.
    if kind == "sharpe" and value_str == "3.2":
        src_file = "experiments/results/audit/sharpe_audit.json"
        src_cmd = "PYTHONPATH=. python3 experiments/audit/audit_sharpe.py"

    return src_file, src_cmd


def determine_match_status(value_str: str, kind: str) -> tuple[str, str]:
    """Return (match_status, note).

    PENDING is the default — Plan 18-07's audit_paper_numbers.py run
    will refine these. The one MISMATCH we surface here is the
    annualized per-pair Sharpe ≈ 3.2 claim, which Plan 18-02 already
    reproduced as naive=18.6 / corrected=7.0; recording it here gives
    Plan 18-07 a head start.
    """
    if kind == "sharpe" and value_str == "3.2":
        return (
            "MISMATCH",
            "Wave 1 (Plan 18-02) reproduced annualized per-pair Sharpe="
            "18.60 (naive) / 7.04 (BLdP corrected); paper claims 3.2. "
            "Plan 18-07 must reconcile in PAPER_DRAFT.md (likely revise "
            "headline) or document the annualization formula derivation.",
        )
    return ("PENDING", "")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not HEADLINE.exists():
        raise FileNotFoundError(f"Canonical headline.json missing: {HEADLINE}")
    # Load canonical to confirm it parses (we don't use values here; Plan 18-07
    # handles the value-matching step).
    json.loads(HEADLINE.read_text())["models"]

    paper = PAPER.read_text().splitlines()

    rows: list[dict[str, object]] = []
    for i, line in enumerate(paper):
        section = in_headline_section(i, paper)
        if section is None:
            continue
        # Skip the heading lines themselves.
        if any(line.startswith(hdr) for hdr in HEADLINE_SECTIONS):
            continue
        # Skip blockquote / code-fence / table-separator lines.
        if line.strip().startswith(("```", "|---", "---")):
            continue
        seen_on_line: set[tuple[str, str, int]] = set()
        for regex, kind in (
            (SHARPE_RE, "sharpe"),
            (SHARPE_LONG_RE, "sharpe"),
            (BPS_RE, "bps"),
            (DOLLAR_RE, "dollar"),
            (PCT_RE, "pct"),
        ):
            for m in regex.finditer(line):
                claim_text = m.group(0)
                value_str = m.group(1)
                # Dedupe: SHARPE_RE and SHARPE_LONG_RE can collide.
                # Key on (kind, value, end-position-of-numeric-group).
                dedup_key = (kind, value_str, m.end(1))
                if dedup_key in seen_on_line:
                    continue
                seen_on_line.add(dedup_key)
                src_file, src_cmd = classify_claim_value(value_str, kind)
                status, note = determine_match_status(value_str, kind)
                rows.append(
                    {
                        "claim_text": claim_text,
                        "claim_value": value_str,
                        "kind": kind,
                        "paper_section": section,
                        "line_number": i + 1,
                        "source_file": src_file,
                        "source_command": src_cmd,
                        "match_status": status if not note else f"{status}: {note}",
                    }
                )

    # Explicit fieldnames (don't rely on rows[0].keys() — defensive against
    # the empty-rows edge case and ensures a stable column order downstream).
    fieldnames = [
        "claim_text",
        "claim_value",
        "kind",
        "paper_section",
        "line_number",
        "source_file",
        "source_command",
        "match_status",
    ]
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT} ({len(rows)} claims)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
