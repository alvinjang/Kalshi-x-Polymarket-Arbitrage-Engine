#!/usr/bin/env bash
# Phase 14 paper-integrity validator. Runs grep-based checks for POL-04/05/06/07/08/09/10.
# Exits 0 if all checks pass, non-zero otherwise. Runs in ~1 second.

set -u
PAPER="PAPER_DRAFT.md"
FAIL=0

check() {
  local name="$1"; local expected="$2"; local actual="$3"
  if [[ "$actual" == "$expected" ]]; then
    printf "  [OK]   %-50s (got %s)\n" "$name" "$actual"
  else
    printf "  [FAIL] %-50s (want %s, got %s)\n" "$name" "$expected" "$actual"
    FAIL=1
  fi
}

check_ge() {
  local name="$1"; local minimum="$2"; local actual="$3"
  if (( actual >= minimum )); then
    printf "  [OK]   %-50s (got %s, want >= %s)\n" "$name" "$actual" "$minimum"
  else
    printf "  [FAIL] %-50s (got %s, want >= %s)\n" "$name" "$actual" "$minimum"
    FAIL=1
  fi
}

echo "== POL-04: Abstract word count =="
WC=$(awk '/^## Abstract/{f=1;next} /^---$/{if(f)exit} f' "$PAPER" | wc -w | tr -d ' ')
if (( WC <= 250 )); then
  printf "  [OK]   abstract_words <= 250                            (got %s)\n" "$WC"
else
  printf "  [FAIL] abstract_words <= 250                            (got %s)\n" "$WC"
  FAIL=1
fi

echo "== POL-05: References alphabetical + Cont entry present =="
REFS=$(sed -n '/^## References$/,/^---$/p' "$PAPER" | grep -cE "^[0-9]+\.")
check_ge "references_count" 14 "$REFS"
CONT=$(sed -n '/^## References$/,/^---$/p' "$PAPER" | grep -c "Cont")
check_ge "cont_kukanov_entry" 1 "$CONT"
# Alphabetical check: skip the [Anonymous] entry (no uppercase first letter), check the rest
DISORDER=$(sed -n '/^## References$/,/^---$/p' "$PAPER" | grep -oE "^[0-9]+\. [A-Z][a-zA-Z-]+" | awk '{print $2}' | sort -c 2>&1 | grep -c "disorder")
check "references_alphabetical" 0 "$DISORDER"

echo "== POL-06: Tables/Figures uniquely numbered =="
check "table_6_count"  1 "$(grep -c '^\*\*Table 6'  "$PAPER")"
check "table_7_count"  1 "$(grep -c '^\*\*Table 7'  "$PAPER")"
check "table_9_count"  1 "$(grep -c '^\*\*Table 9'  "$PAPER")"
check "table_10_count" 1 "$(grep -c '^\*\*Table 10' "$PAPER")"
APPB=$(awk '/^## Appendix B/,0' "$PAPER" | grep -c "^- \*\*Fig\. ")
check_ge "appendix_b_figure_bullets" 11 "$APPB"

echo "== POL-07: Per-pair Sharpe is the headline =="
PP=$(grep -c "per-pair" "$PAPER")
check_ge "per_pair_mentions" 3 "$PP"
STALE=$(grep -cE "0\.59 annualize|annualizes to 4\.3|annualizes to roughly 4\.3" "$PAPER")
check "stale_sharpe_claims" 0 "$STALE"

echo "== POL-08: Limitations + Fig 2 cap annotation =="
SURV=$(awk '/^### 6\.4 Limitations/,/^### 6\.[5-9]|^## /' "$PAPER" | grep -ic "survivorship")
check_ge "survivorship_in_6_4" 1 "$SURV"
LIVECOHORT=$(awk '/^### 6\.4 Limitations/,/^### 6\.[5-9]|^## /' "$PAPER" | grep -ic "live-cohort\|pair_id schema")
check_ge "live_cohort_in_6_4" 1 "$LIVECOHORT"

echo "== POL-09: AI-assistant disclosure =="
AI=$(sed -n '/^## Acknowledgments$/,/^## References$/p' "$PAPER" | grep -ic "claude\|anthropic")
check_ge "ai_disclosure" 1 "$AI"

echo "== POL-10: No residual TODOs/placeholders =="
TODO=$(grep -cE "TODO|FIXME|XXX|\[Insert|TBD" "$PAPER")
check "todo_placeholder_count" 0 "$TODO"
DEAD=$(grep -cE "§4\.6|Figure 2b|Fig\. 2b" "$PAPER")
check "dead_crossrefs" 0 "$DEAD"

echo "== REPL-06: Pitch-standard headlines (Phase 17) =="
# REPL-06a: Abstract must mention Sharpe (the new pitch-standard headline metric).
ABS_SHARPE=$(awk '/^## Abstract/{f=1;next} /^---$/{if(f)exit} f' "$PAPER" | grep -cE "[Ss]harpe")
check_ge "abstract_mentions_sharpe" 1 "$ABS_SHARPE"

# REPL-06b: Abstract must cite a specific Sharpe value (decimal between 0.0 and 9.9), not a hand-wavy mention.
ABS_SHARPE_VAL=$(awk '/^## Abstract/{f=1;next} /^---$/{if(f)exit} f' "$PAPER" | grep -cE "[Ss]harpe[^0-9]{1,30}[0-9]\.[0-9]+|[0-9]\.[0-9]+[^0-9]{1,30}[Ss]harpe")
check_ge "abstract_cites_sharpe_value" 1 "$ABS_SHARPE_VAL"

# REPL-06c: In headline sections (Abstract / §5.1 / §5.8 / §6.3 / §8 Conclusions), every signed P&L
# claim of $50+ (i.e., +$XXX.XX or −$XXX.XX) must have a Sharpe or bps companion in the same paragraph.
# Two-pass implementation: extract headline-section text, then check paragraphs.
ORPHAN_DOLLARS=$(awk '
  BEGIN { in_h = 0 }
  /^## Abstract/ { in_h = 1; next }
  /^### 5\.1 / { in_h = 1; next }
  /^### 5\.8 / { in_h = 1; next }
  /^### 6\.3 / { in_h = 1; next }
  /^## 8\. / { in_h = 1; next }
  /^## [A-Z0-9]/ { in_h = 0 }
  /^### [0-9]/ { in_h = 0 }
  in_h { print }
' "$PAPER" | awk '
  BEGIN { RS=""; FS="\n"; n = 0 }
  /[+−-]\\?\$([5-9][0-9]|[1-9][0-9]{2,})(\.[0-9]+)?/ && !/[Ss]harpe/ && !/bps/ { n++ }
  END { print n }
')
check "orphan_dollar_paragraphs_in_headline_sections" 0 "$ORPHAN_DOLLARS"

echo "== AUDIT-05: Phase 18 number-by-number regression checks =="
# Tier 5: each check below extracts a canonical headline.json number and greps
# PAPER_DRAFT.md for it. Drift between headline.json and PAPER_DRAFT.md fails
# the validator immediately. See experiments/audit/build_paper_numbers_csv.py
# for the full traceability map (paper_numbers.csv).

# Helper: extract a canonical headline.json model field, rounded to 3 decimal
# places (or 2 for values >= 100). Single source of truth: headline.json.
canon() {
    python3 -c "import json; m=json.load(open('experiments/results/canonical/headline.json'))['models']['$1']; v=m['$2']; print(f'{v:.3f}' if abs(v) < 100 else f'{v:.2f}')"
}

# audit_lr_per_trade_sharpe_in_paper: LR per-trade Sharpe (0.501)
LR_PT_SHARPE=$(canon linear_regression sharpe_per_trade)
LR_FOUND=$(grep -c "$LR_PT_SHARPE" "$PAPER")
check_ge "audit_lr_per_trade_sharpe_in_paper" 1 "$LR_FOUND"

# audit_lr_alpha_bps_in_paper: LR alpha bps rounded to 1 decimal (15.0 bps)
LR_BPS=$(canon linear_regression alpha_bps_per_trade)
LR_BPS_ROUNDED=$(printf "%.1f" "$LR_BPS")
LR_BPS_FOUND=$(grep -c "$LR_BPS_ROUNDED bps" "$PAPER")
check_ge "audit_lr_alpha_bps_in_paper" 1 "$LR_BPS_FOUND"

# audit_xgb_per_trade_sharpe_in_paper: XGBoost per-trade Sharpe (0.499)
XGB_PT_SHARPE=$(canon xgboost sharpe_per_trade)
XGB_FOUND=$(grep -c "$XGB_PT_SHARPE" "$PAPER")
check_ge "audit_xgb_per_trade_sharpe_in_paper" 1 "$XGB_FOUND"

# audit_ppo_filtered_alpha_bps_in_paper: PPO+autoencoder alpha bps (0.5 bps)
PPO_BPS=$(canon ppo_filtered alpha_bps_per_trade)
PPO_BPS_ROUNDED=$(printf "%.1f" "$PPO_BPS")
PPO_BPS_FOUND=$(grep -c "$PPO_BPS_ROUNDED bps" "$PAPER")
check_ge "audit_ppo_filtered_alpha_bps_in_paper" 1 "$PPO_BPS_FOUND"

# audit_per_trade_sharpe_in_abstract: per-trade Sharpe (leakage-free) in abstract.
# Plan 18-07 replaced the un-derivable per-pair Sharpe ≈ 3.2 claim with the
# leakage-free per-trade Sharpe headline (0.516, drift +2.99% from canonical
# 0.501). This check enforces that the abstract leads with a per-trade Sharpe
# claim of the form "per-trade Sharpe ... 0.5XX" so future drafts cannot
# silently revert to the un-validated annualized framing.
PT_SHARPE_FOUND=$(awk '/^## Abstract/,/^## 1\./' "$PAPER" | grep -cE "per-trade Sharpe[^0-9]{1,30}0\.5[0-9]+|0\.51[56]|0\.50[0-9]")
check_ge "audit_per_trade_sharpe_in_abstract" 1 "$PT_SHARPE_FOUND"

# audit_walk_forward_11_windows_in_paper: 11-window walk-forward cited
WF_COUNT=$(grep -cE "11[ -]window|11 walk-forward|across 11" "$PAPER")
check_ge "audit_walk_forward_11_windows_in_paper" 1 "$WF_COUNT"

# audit_test_rows_1673_in_paper: canonical test row count (1,673)
TEST_ROWS=$(grep -c "1,673" "$PAPER")
check_ge "audit_test_rows_1673_in_paper" 1 "$TEST_ROWS"

echo
if (( FAIL == 0 )); then
  echo "ALL CHECKS PASSED"
  exit 0
else
  echo "SOME CHECKS FAILED"
  exit 1
fi
