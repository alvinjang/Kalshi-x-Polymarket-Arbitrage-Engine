# AUDIT_REPORT.md — Phase 18 System Audit

**Audit window:** 2026-04-25 → 2026-04-26
**Scope:** Adversarial verification of every quantitative claim in `PAPER_DRAFT.md`, `experiments/results/canonical/headline.json`, and `slides_deck.html` before the April 27 submission.
**Posture:** Kill-or-confirm. Every Tier verdict is **PASS**, **CORRECTED**, or **FAILED** — no soft fails. A Tier flagged FAILED on the original methodology is reported as RESOLVED only when a leakage-free recompute closes the defect; the original failure is documented either way.

## Bottom line

**Per-trade Sharpe ≈ 0.515 and per-trade alpha ≈ 15.7 bps are robust to leakage correction.** The leaky-canonical and leakage-free pipelines both produce per-trade Sharpe in the +0.50 to +0.52 range (drift +2.99% from canonical to purged). The per-pair annualized Sharpe is regime-dependent — it depends on the cross-pair correlation structure of the test pairs, which is sample-size-sensitive at N=29 purged test pairs. The honest paper headline is **per-trade Sharpe ≈ 0.52, per-trade alpha ≈ 15.7 bps, total P&L ≈ \$200 on 1,398 leakage-free test trades at \$100/trade**, with the per-pair annualized number disclosed as a regime-dependent secondary statistic in §5.8.

The original "≈ 3.2" per-pair annualized Sharpe in the abstract had no derivation path in the codebase. The audit found it, the leakage-free recompute replaces it, and the new framing leads with a number that reproduces from canonical code.

## Summary

| Tier | Audit Name | Verdict | Evidence | Key Findings |
|------|------------|---------|----------|--------------|
| 1 | Headline Sharpe verification | **CORRECTED** | [sharpe_audit.json](experiments/results/audit/sharpe_audit.json) · [sharpe_audit_purged.json](experiments/results/audit/sharpe_audit_purged.json) | Per-trade Sharpe stable: leaky 0.5007 → purged 0.5157 (+2.99% drift, well within bootstrap CI). Per-pair corrected Sharpe is regime-dependent: leaky 0.296 (under +0.042 avg pairwise correlation, BLdP haircut active) → purged 0.814 (under −0.199 avg pairwise correlation on N=29, BLdP short-circuited). Per-pair 95% CI on purged data: [+0.700, +1.067]. The original paper claim of ≈ 3.2 per-pair *annualized* Sharpe has no derivation path; it is replaced by per-trade-led headline framing. |
| 2 | Leakage / look-ahead bias | **RESOLVED** (FAILED on canonical, PASS on purged) | [leakage_audit.json](experiments/results/audit/leakage_audit.json) · [leakage_audit_purged_check.json](experiments/results/audit/leakage_audit_purged_check.json) | Original canonical 80/20 row-index split bridged 144/144 pairs across train/test with 142 embargo violations at 4-hour gap (gap ≪ 86,400 s 1-day embargo). Pair-stratified split (Plan 18-08) rebuilds 80/20 by pair_id with seed 42 → 0 bridging pairs and 0 embargo violations by construction. Feature-level audit found 0 leaking features and 7 Suspicious (rolling-window endpoints — manually reviewed and confirmed safe); 1 quality-filter rule (`stale_ticker`) flagged retroactive (uses `_current_year()` at audit time, benign for 2026 backtest). |
| 3 | Cost realism | **CORRECTED** | [costs_audit.json](experiments/results/audit/costs_audit.json) | `simulate_profit` charges zero fee — its `threshold=0.02` is a SIGNAL gate for trade entry, not a transaction cost. PAPER_DRAFT.md §5.1 misstated this as "2 pp transaction costs" — corrected in this plan. `WalkForwardBacktester` charges 5pp round-trip (3pp entry + 2pp exit), conservative vs realistic Kalshi+Polymarket round-trip of 250–355 bps. Slippage sensitivity sweep (0/5/10/20/50 bps additional haircut): annualized Sharpe 8.95 → 8.81 (−1.6%), P&L \$253,728 → \$243,964 (−3.8%). Cost-robustness claim survives. Polymarket gas + Kalshi fee schedule documented in §6.4. |
| 4 | Survivorship / selection | **PASS** | [survivorship_audit.json](experiments/results/audit/survivorship_audit.json) | Heuristic drop-rate 99.9% (n_dropped=148,094 / n_candidates=148,238). All 10 random-sample dropped pairs classified **structural** by Ian's review: every one is a `26apr*` April-2026-expiry contract discovered live after the canonical `test.parquet` 2026-04-09 snapshot, so post-snapshot pairs cannot retroactively appear in pre-snapshot training data. Drops are structural by construction (low_overlap_n_bars=0, never entered offline pipeline), not retroactive. 0/10 retroactive. |
| 5 | Paper number-by-number trace | **PASS** | [paper_numbers.csv](experiments/results/audit/paper_numbers.csv) · [check_paper.sh](scripts/check_paper.sh) | 86 numeric claims traced through `experiments/results/audit/paper_numbers.csv`. 7 new `audit_*` regression checks added to `scripts/check_paper.sh`. After Tier 1 paper-text edits (this plan), 0 unresolved MISMATCH rows; the original 2 MISMATCH rows for the per-pair Sharpe ≈ 3.2 claim flip to PASS once the abstract and §8 lead with the new per-trade headline. `bash scripts/check_paper.sh` exits 0 with all checks green. |
| 6 | Live-vs-backtest honesty | **PASS** | [live_vs_backtest_audit.json](experiments/results/audit/live_vs_backtest_audit.json) | Two-proportion z-test on live oil WR (36.0% / 1,224 positions) vs backtest oil WR (76.5%): z = −10.76, p = 5.24 × 10⁻²⁷, Cohen's h = −0.842 (large effect). The gap is statistically significant, large in effect size, and **not measuring the same thing** — backtest WR is on the near-expiry oil subset only; live WR is on the full 1,224-position commodity cohort across all series and expiries. PAPER_DRAFT.md §5.9.1 already discloses this scope mismatch honestly; the z-test confirms the disclosure is appropriate. No paper change required. |

## Detailed findings

### Tier 1: Headline Sharpe verification

**Audit script:** `experiments/audit/audit_sharpe.py` (canonical) · `experiments/audit/audit_sharpe_purged.py` (purged)

**Per-trade Sharpe** is invariant to leakage correction:
- Canonical (leaky 80/20 row-index split): **0.5007** [recomputed from raw trade ledger, drift 0.00016 vs canonical headline.json 0.5009]
- Purged (pair-stratified 80/20 split): **0.5157** (drift +2.99%)

**Per-pair Sharpe** is regime-dependent. The Bailey–López de Prado (2012) effective-sample correction `n_eff = N / (1 + (N − 1) × avg_corr)` was the active ingredient compressing the leaky number, and it does not apply to the purged sample because the purged cross-pair correlation is negative:
- Canonical avg pairwise correlation: **+0.0418** (461 contemporaneous-day pair-pairs); BLdP haircut compresses naive 0.781 → corrected 0.296
- Purged avg pairwise correlation: **−0.1986** (89 cell pairs at N=29); BLdP short-circuits (avg_corr ≤ 0 returns naive); naive 0.814 = corrected 0.814
- Purged 95% bootstrap CI: **[+0.700, +1.067]** (10,000 resamples)

**Annualization** (purged): pairs_per_year = 120.51 (test_span_days=87.83, N_pairs=29) → annualization factor √120.51 ≈ 10.98 → annualized per-pair Sharpe ≈ **8.93** under the assumption that pair-lifecycle distribution in test window is representative of annual operation (likely violated; explicitly flagged in §6.4).

**Assumption stack** (verbatim from `sharpe_audit_purged.json`):
1. Purged split is pair-atomic by construction — every pair is entirely in train OR entirely in test, never both.
2. Per-pair returns are stationary (no regime change within purged test window).
3. BLdP correction follows Bailey–López de Prado (2012): n_eff = N / (1 + (N−1) × avg_corr); sharpe_corrected = sharpe_naive × √(n_eff/N).
4. Annualization assumes pair-lifecycle distribution in test window is representative of annual operation (caveat in §6.4).
5. Bootstrap CI uses simple resample-with-replacement (10,000 resamples). Does NOT correct for autocorrelation in per-pair returns.
6. LR is the headline model audited (LR wins 4 of 5 metrics in `canonical/headline.json`).

**Verdict logic** (from JSON):
- PASS: corrected > 0.5 AND CI_lower > 0.0 AND |drift_pct| < 50%
- **CORRECTED**: corrected > 0.0 AND CI_lower > −0.2 → **fires here** (corrected 0.81 > 0, CI lower 0.70 > −0.2, drift_pct 175% exceeds the 50% gate)
- FAILED: otherwise

The +175% drift is favorable in direction (purged HIGHER than leaky), but the audit reports CORRECTED honestly — the leakage-free number is materially different from the leaky number.

### Tier 2: Leakage / look-ahead bias

**Audit script:** `experiments/audit/audit_leakage.py` · `experiments/audit/verify_purged_no_bridge.py`

**Per-feature classification (23 features in `src/features/build_features.py`):**

| Verdict | Count | Features |
|---|---|---|
| Safe | 16 | `price_velocity`, `kalshi_order_flow_imbalance`, `polymarket_order_flow_imbalance`, `spread_zscore`, `dollar_volume_ratio`, `trade_count_ratio`, `mid_price`, `price_divergence_pct`, `kalshi_amihud`, `polymarket_amihud`, `kalshi_hl_vol`, `polymarket_hl_vol`, `ofi_differential` (×2 — null-handling branches), `boundary_distance`, `longshot_score` |
| Suspicious | 7 | `volume_ratio`, `spread_momentum`, `spread_volatility`, `spread_momentum_6`, `spread_momentum_12`, `spread_volatility_6`, `spread_range` (all rolling-window-endpoint patterns; manually reviewed and confirmed `center=False` and window endpoint ≤ entry_ts) |
| Leaking | 0 | — |

**Walk-forward embargo audit on canonical 80/20 row-index split:**
- 144 train pairs, 144 test pairs, **144 bridging pairs**, **142 embargo violations** at 1-day (86,400 s) embargo policy
- Sample violations: typical gap is **4 hours** (14,400 s), one-bar adjacency between train end and test start for the same pair_id
- Verdict on canonical split: **FAILED**

**Walk-forward embargo audit on pair-stratified split (data/processed/purged_split/, seed=42):**
- 115 train pairs, 29 test pairs, **0 bridging pairs**, **0 embargo violations** (by construction — splitter enforces `train_pairs.isdisjoint(test_pairs)`)
- Verdict on purged split: **PASS**

**Quality-filter rule audit (12 rules in `src/matching/quality_filter.py`):**

| Rule | Uses | Retroactive? |
|---|---|---|
| MIN_CONFIDENCE | `confidence_score` | No (pre-trade match score) |
| MAX_RESOLUTION_GAP_DAYS | `kalshi_resolution_date`, `polymarket_resolution_date` | No (resolution DATE, not OUTCOME, fixed at listing) |
| directions_compatible | question text | No |
| thresholds_compatible | question text | No |
| Rule 1 (season-wins vs champion) | ticker prefix, title keywords | No |
| Rule 2 (Fed year/month mismatch) | ticker date, title month/year | No |
| Rule 3 (cabinet vs nomination) | title keywords | No |
| Rule 3b (threshold vs ranking) | ticker, poly title | No |
| Rule 3c (threshold vs policy) | title keywords | No |
| Rule 3d (AAA gas geography) | ticker suffix, poly geography | No |
| **Rule stale_ticker** | ticker year vs **current year** | **Yes (audit-time, benign)** |
| Rule 10 (asset-class consistency) | Kalshi ticker prefix, title tokens | No |

`stale_ticker` is the only rule that uses audit-time (`_current_year()`) information. It is benign for the 2026 audit because rejection is a coarse "past year" check; a 2026 ticker passing the rule today would also have passed in January. Documented as a known minor caveat; tightening to timestamp-aware would not change any past matching decisions.

**Net Tier 2 verdict:** the original canonical split FAILED the embargo gate; the pair-stratified rebuild RESOLVES the failure. The paper now reports both, transparently.

### Tier 3: Cost realism

**Audit script:** `experiments/audit/audit_costs.py`

**`simulate_profit` (canonical headline simulator):**
- Function: `src.evaluation.profit_sim.simulate_profit`
- Fee charged: **0.0**
- Returns raw spread-units P&L (`predicted_direction × actual_change`) for trades passing `|pred| > threshold`
- The `threshold = 0.02` parameter is a SIGNAL gate, NOT a fee deduction
- **Paper claim mismatch:** §5.1 line 213 in PAPER_DRAFT.md said "single-split backtest at 2 pp transaction costs" — this is misleading because Table 2 numbers come from `simulate_profit` (zero fee), not `WalkForwardBacktester`
- **Fix applied in this plan:** §5.1 prose corrected to "single-split backtest with a 2 pp signal threshold for trade entry; transaction-cost sensitivity analyzed separately in §5.6"

**`WalkForwardBacktester` (used in §5.6 transaction-cost sensitivity):**
- Function: `src.evaluation.backtester.WalkForwardBacktester`
- Entry cost: 3 pp · Exit cost: 2 pp · Round-trip: 5 pp = 500 bps
- Compared to realistic Kalshi+Polymarket round-trip 250–355 bps, the backtester is **conservative** (~1.4–2× realistic)

**Realistic 2026 fee references (from public sources):**
- **Kalshi** (`kalshi.com/fee-schedule`): taker = 0.07 × C × (1 − C) per contract; max 1.75¢ at C=0.50; maker = 25% of taker; settlement = 0
- **Polymarket** (`docs.polymarket.com/trading/fees`): taker by category — crypto 1.80%, economics 1.50%, mentions 1.56%, culture 1.25%, weather 1.25%, finance 1.00%, politics 1.00%, tech 1.00%, sports 0.75%, geopolitics 0%; maker = 0; gas ≈ \$0.01/tx
- **Fix applied in this plan:** §6.4 Limitations now documents the explicit Kalshi + Polymarket fee schedules

**Slippage sensitivity** (additional haircut on top of WalkForwardBacktester's 5 pp):

| Haircut (bps) | Annualized Sharpe | Total P&L | Total fees | Win rate |
|---|---|---|---|---|
| 0 | 8.955 | \$253,728 | \$97,644 | 45.45% |
| 5 | 8.940 | \$252,751 | \$98,620 | 45.45% |
| 10 | 8.926 | \$251,775 | \$99,596 | 45.38% |
| 20 | 8.897 | \$249,822 | \$101,549 | 45.26% |
| 50 | 8.807 | \$243,964 | \$107,408 | 44.93% |

Sharpe drops by 1.6% and P&L by 3.8% under +50 bps additional haircut — cost-robustness claim in §5.6 survives.

### Tier 4: Survivorship / selection

**Audit script:** `experiments/audit/audit_survivorship.py`

- **Candidate pair universe** (from `aligned_pairs.parquet` + `active_matches.json`): **148,238**
- **Realized pair universe** (in canonical training set): **144**
- **Drop rate:** 99.9%
- **Random sample of 10 dropped pairs** (with `manual_classification_required` field):

| pair_id | Inferred drop reason | Final classification |
|---|---|---|
| `kxbtc26apr1500t61200-0x58fb4378` | low_overlap_n_bars=0 | structural |
| `kxbnb26apr1822b467-0x1b3b0ec3` | low_overlap_n_bars=0 | structural |
| `kxcpicore26aprt01-0x42df8b1f` | low_overlap_n_bars=0 | structural |
| `kxbtcd26apr2217t8099999-0x23fb92bb` | low_overlap_n_bars=0 | structural |
| `kxbtcd26apr2005t7229999-0x63743613` | low_overlap_n_bars=0 | structural |
| `kxbtc26apr2017b73125-0x63743613` | low_overlap_n_bars=0 | structural |
| `kxbtc26apr1400b64050-0x1c2f06de` | low_overlap_n_bars=0 | structural |
| `kxsole26apr2010b69-0xedfc3d87` | low_overlap_n_bars=0 | structural |
| `kxbnbd26apr2407t63499-0x1b3b0ec3` | low_overlap_n_bars=0 | structural |
| `kxethd26apr2122t152999-0x4c608ba8` | low_overlap_n_bars=0 | structural |

**10/10 structural · 0/10 retroactive.** Every entry is a `26apr*` April-2026-expiry contract discovered live after the canonical `test.parquet` 2026-04-09 snapshot. Post-snapshot pairs cannot retroactively appear in pre-snapshot training data — the drops are structural by construction (the pair never entered the offline pipeline because it didn't exist when the offline data was frozen). Verdict **PASS**.

### Tier 5: Paper number-by-number trace

**Audit script:** `experiments/audit/build_paper_numbers_csv.py` · `scripts/check_paper.sh`

- **86 numeric claims** mechanically extracted from PAPER_DRAFT.md and traced to `experiments/results/canonical/headline.json` and `experiments/results/audit/sharpe_audit*.json`
- **7 new `audit_*` regression checks** added to `scripts/check_paper.sh` (LR per-trade Sharpe, LR alpha bps, XGB per-trade Sharpe, PPO+autoencoder alpha bps, per-pair Sharpe headline value in abstract, 11-window walk-forward count, 1,673 test rows)
- **Pre-edit MISMATCH count:** 2 (the per-pair Sharpe ≈ 3.2 claim in Abstract line 12 and §8 line 710 — both replaced by the per-trade headline in this plan)
- **Post-edit MISMATCH count:** 0
- `bash scripts/check_paper.sh` exits 0 with **all 26 checks green**

### Tier 6: Live-vs-backtest honesty

**Audit script:** `experiments/audit/audit_live_vs_backtest.py`

- **Live oil cohort:** 441 wins / 1,224 positions = **36.0% win rate** (PAPER_DRAFT.md §5.9.1 lines 446–464; full commodity cohort across KXBRENTW, KXWTI, KXWTIW, KXBRENTMON, KXAAAGAS\*)
- **Backtest oil cohort:** 76.5% win rate (PAPER_DRAFT.md §5.3 near-expiry oil subset; n=200 estimated, paper does not state explicit n)
- **Two-proportion z-test:** z = **−10.761**, p = **5.24 × 10⁻²⁷** (two-sided)
- **Cohen's h:** **−0.842** → large effect (Cohen 1988: 0.8+ = large)
- **Honest interpretation** (verbatim from JSON): the gap is statistically significant AND large in effect, but the samples are not measuring the same thing — backtest 76.5% is on the near-expiry oil subset only; live 36.0% is on the full 1,224-position commodity cohort across all series and expiries. PAPER_DRAFT.md §5.9.1 already discloses this scope mismatch; the z-test is supplementary evidence that the disclosure is appropriate, not over-stated.
- **Paper corrections required:** none.
- Verdict: **PASS**.

## Corrections applied

Three Tiers were CORRECTED in this plan; the paper updates that landed alongside `AUDIT_REPORT.md`:

| Tier | Section | Change |
|---|---|---|
| 1 | Abstract | Headline framing reworked. New lead: per-trade Sharpe **0.515** with **+15.7 bps** per-trade alpha on the leakage-free 1,398-row test set (115 train pairs / 29 test pairs); Phase 18 audit motivation noted; per-pair annualized number demoted to §5.8 with regime caveat. |
| 1 | §5.8 (Honest Sharpe-Ratio Accounting) | Rewritten. Per-trade Sharpe leads. Per-pair Sharpe reported as a range bracketing the regime sensitivity: leaky-canonical corrected 0.296 ↔ purged 0.814 (95% CI [+0.70, +1.07] on N=29 purged test pairs). BLdP-correction-applicability mechanism explained. |
| 1 | §8 Conclusions, item 5 | Replaced "≈ 3.2" with leakage-free per-trade headline matching the new abstract. |
| 1 | Table 8 | Rebuilt. Lead row = per-trade Sharpe with both leaky (0.501) and purged (0.516) values; per-pair annualized rows clearly labeled with regime caveat. |
| 2 | §6.4 Limitations | Added paragraph documenting the embargo-violation finding on the canonical split and the pair-stratified rebuild that resolves it. |
| 3 | §5.1 line 213 | Replaced "single-split backtest at 2 pp transaction costs" with "single-split backtest with a 2 pp signal threshold for trade entry; transaction-cost sensitivity analyzed separately in §5.6". |
| 3 | §6.4 Limitations | Added Polymarket gas + Kalshi fee schedule paragraph. |
| 3 | §6.4 Limitations | Added cross-reference to AUDIT_REPORT.md as supplementary methodological-care evidence. |
| — | `slides_deck.html` | Updated Results slide stat card from "Per-pair Sharpe ≈ 3.2" to per-trade Sharpe ≈ 0.52 framing. |
| — | `scripts/check_paper.sh` | Replaced `audit_per_pair_sharpe_3_2_in_abstract` regex check with `audit_per_trade_sharpe_in_abstract` to match the new headline; all 26 checks green. |
| — | `experiments/results/audit/paper_numbers.csv` | Flipped the 2 MISMATCH rows for the 3.2 claim to PASS. |

## Conclusion

Six Tier audits ran adversarially against PAPER_DRAFT.md, the canonical headline JSON, and the live trading data. Three Tiers (1, 3, the canonical-split portion of Tier 2) returned non-PASS verdicts; all three were corrected in the same plan that produced this report. Three Tiers (4, 5, 6) returned PASS without paper changes. The leakage defect on the canonical split is RESOLVED on the purged split. The headline numbers in the paper are now leakage-free and reproducible from canonical code (`experiments/run_canonical_purged.py`).

April 27 submission integrity confirmed.

---

*Generated: 2026-04-26*
*Phase 18 by Ian Sabia (U33871576) and Alvin Jang (U64760665) for DS340 Final Project*
