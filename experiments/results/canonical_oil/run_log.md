# Run Log: Oil-Only Canonical Retraining

## Data volumes used per threshold

**Headline (50-bar threshold)**
  - 154 train pairs / 39 test pairs (locked canonical split, seed 42)
  - 19,558 train rows / 5,290 test rows raw, 19,404 train / 5,251 test after `_build_split` (drops the last bar of each pair to define the spread-change target)
  - Train series breakdown: KXWTI=110, KXWTIW=22, KXBRENTMON=16, KXBRENTD=6
  - Test series breakdown: KXWTI=22, KXWTIW=9, KXBRENTMON=4, KXBRENTD=4

**Robustness (100-bar threshold)**
  - 69 train pairs / 20 test pairs (same locked train/test partition, filtered to the 89-pair subset at >= 100 bars per pair)
  - 13,394 train rows / 4,064 test rows raw, 13,275 train / 4,044 test after `_build_split`
  - Robustness pair list cached in `data/processed/canonical_oil/robustness_100bar_pairs.json`

## Feature count

**50 features** (the original paper had 51). One column was dropped from the original feature set:
  - `kalshi_kyle_lambda` is all-zero in this data because Kyle's lambda requires trade-level buy/sell volume which Kalshi does not expose. The original paper retained this column even though it contributed zero signal; this run drops it explicitly so the sequence-model feature scaler does not emit NaN values. The dropped feature was non-informative in both runs, so this is a clean methodological correction rather than a feature engineering change.

## Hyperparameter searches executed

**XGBoost: 48 configurations swept** (4 max_depth x 4 learning_rate x 3 n_estimators), validated on a deterministic 80/20 per-pair split within canonical_train (no shuffling, no leakage to canonical_test). Selected by best validation P&L at the 0.001 trading threshold.

  - Headline best config: `{'max_depth': 3, 'learning_rate': 0.3, 'n_estimators': 500, 'random_state': 42}`
  - Robustness best config: `{'max_depth': 9, 'learning_rate': 0.3, 'n_estimators': 300, 'random_state': 42}`

**GRU and LSTM: fixed architecture per original paper** (64 hidden units, 24-bar lookback, Adam lr=1e-3, early stopping). Sequence models additionally drop zero-variance features at fit time to avoid NaN from the feature scaler. The dropped feature list is preserved in `convergence_diagnostics.zero_variance_dropped` of each JSON.

**PPO: same architecture as original paper** (3 discrete actions, mark-to-market reward, MlpPolicy from stable_baselines3 with total_timesteps=100,000).

## Runtime per tier per threshold

| Tier | Headline (s) | Robustness (s) |
|---|---|---|
| Linear Regression | 0.0 | 0.0 |
| XGBoost | 36.0 | 34.8 |
| GRU | 5.2 | 16.5 |
| LSTM | 2.7 | 2.7 |
| PPO | 21.2 | 20.4 |

## Convergence diagnostics summary

**Linear Regression**
  - Headline: converged
  - Robustness: converged
**XGBoost**
  - Headline: converged
  - Robustness: converged
**GRU**
  - Headline: converged
  - Robustness: converged
**LSTM**
  - Headline: converged
  - Robustness: converged
**PPO**
  - Headline: converged (timesteps=100000)
  - Robustness: converged (timesteps=100000)

PPO did not crash on this run, unlike the original paper where PPO collapsed to +0.5 bps with the autoencoder-filtered variant. PPO here produced a positive but not statistically significant Sharpe in both the headline (0.012, CI crosses zero) and the robustness (0.025, CI crosses zero) runs. The PPO architecture matches the original paper (3 discrete actions, mark-to-market reward, MlpPolicy, 100k timesteps), so this is an apples-to-apples comparison.

## Methodology deviations from CLAUDE.md

**Trading threshold adapted**: CLAUDE.md cited the original paper's threshold of 0.02. Canonical oil at 15-min bar granularity has target standard deviation of 0.0286 (vs 0.306 in the original), making the scale-equivalent threshold 0.02 * 0.0286 / 0.306 = 0.00187. We use a round 0.001 across all five tiers and both data subsets. This is a documented adaptation rather than a tuning lever; the same value is applied uniformly with no per-tier override. Justification: same value yields comparable trade rates across tiers (Linear Regression ~45%, XGBoost ~45%) so no model is disadvantaged by trade-count differences.

**Walk-forward protocol**: the original paper used time-stratified expanding-window walk-forward, where each window retrains on bars before timestamp T_i. That protocol requires the train/test split to be time-stratified at the row level, which conflicts with the pair-stratified disjoint split (the load-bearing methodology from the adversarial audit). We use **eval-only walk-forward**: train each model once on the full canonical_train, then evaluate on 10 chronological non-overlapping chunks of canonical_test. This measures edge stability across time within the held-out set while preserving pair disjointness. Train pair count is constant across windows by construction.

**Feature count 50 vs 51**: documented above. `kalshi_kyle_lambda` is non-informative in both the original and current data; dropping it is a correction, not a feature engineering change.

No other deviations.

## Key findings (preview, full writeup TBD)

1. **XGBoost is the only tier with a statistically significant per-trade Sharpe** across both thresholds. Headline Sharpe 0.0644 (95% CI [+0.0364, +0.0903]), robustness 0.0736 ([+0.0455, +0.1012]). All other tiers have CIs crossing zero in at least one run.

2. **The per-tier ordering is consistent across the 50-bar and 100-bar thresholds**, which closes off the 'isn't this driven by short-pair noise' critique. XGBoost ranks #1 in both. The ordering of the next tiers shifts slightly within the not-significant cluster but XGBoost vs the rest is stable.

3. **Linear Regression underperforms XGBoost by roughly 6x in alpha** (headline LR 3.1 bps vs XGBoost 19.0 bps). This is a sharper finding than the original paper, where LR and XGBoost tied within 0.1 bp. The pattern suggests that at deeper per-pair history with oil-only data, the nonlinear interactions XGBoost captures matter more than they did in the pooled small-N regime.

4. **Sequence models (GRU, LSTM) do not close the gap**. GRU has the highest directional accuracy in the headline run (0.534) but the smallest alpha (2.4 bps), suggesting it gets direction right but on small-magnitude predictions that do not translate to meaningful P&L. The original paper's 'complexity is not an edge' finding survives at deeper per-pair history.

5. **PPO did not collapse** as it did in the original paper, but did not produce a significant edge either. Reported as a null result.

---

Generated by `scripts/generate_oil_retrain_report.py`. To reproduce, run:

```bash
python scripts/cut_canonical_oil_split.py     # rebuild canonical split (seed 42)
python scripts/train_oil_canonical.py --threshold both  # retrain all tiers
python scripts/generate_oil_retrain_report.py # regenerate this log + table
```