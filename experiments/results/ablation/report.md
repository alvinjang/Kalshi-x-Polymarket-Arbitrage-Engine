# LOGO Feature Ablation — Results Report

**Split:** train_proper=5781, ablation_holdout=1021, final_test=1673

**Note:** P&L/RMSE on `ablation_holdout` (selection metric). 1,000 bootstrap resamples.

| Model | Dropped Group | # Features | P&L @ 2pp | ΔP&L | 95% CI of ΔP&L | RMSE | Dir. Acc. | Classification |
|---|---|---|---|---|---|---|---|---|
| LR | — (baseline) | 51 | $+56.54 | $0.00 | — | 0.2192 | 62.0% | baseline |
| XGBoost | — (baseline) | 51 | $+54.00 | $0.00 | — | 0.2234 | 60.8% | baseline |
| LR | A — Raw OHLCV | 36 | $+53.21 | $-3.33 | [-6.80, -0.12] | 0.2241 | 61.2% | droppable |
| LR | B — Cross-platform | 41 | $+56.72 | $+0.18 | [-0.51, +1.05] | 0.2191 | 62.0% | droppable |
| LR | C — Rolling/momentum | 45 | $+56.23 | $-0.31 | [-2.06, +1.32] | 0.2168 | 61.8% | droppable |
| LR | D — Microstructure | 38 | $+56.54 | $-0.00 | [-0.88, +0.72] | 0.2187 | 61.9% | droppable |
| LR | E — Pred-market | 44 | $+56.77 | $+0.23 | [-0.30, +0.99] | 0.2193 | 61.9% | droppable |
| XGBoost | A — Raw OHLCV | 36 | $+52.32 | $-1.68 | [-5.67, +2.40] | 0.2248 | 61.4% | droppable |
| XGBoost | B — Cross-platform | 41 | $+55.08 | $+1.08 | [-1.48, +4.05] | 0.2265 | 60.7% | droppable |
| XGBoost | C — Rolling/momentum | 45 | $+53.59 | $-0.41 | [-3.74, +2.99] | 0.2242 | 61.2% | droppable |
| XGBoost | D — Microstructure | 38 | $+53.43 | $-0.57 | [-4.30, +2.66] | 0.2295 | 60.7% | droppable |
| XGBoost | E — Pred-market | 44 | $+52.85 | $-1.15 | [-4.56, +2.01] | 0.2316 | 61.1% | droppable |

## Classification Key
- **load-bearing**: 95% CI < 0, |ΔP&L| > $10, Dir.Acc. drop > 2pp
- **inconclusive**: CI straddles zero with |ΔP&L| > $10
- **droppable**: CI straddles zero with |ΔP&L| <= $10
- **baseline**: all 51 features (reference)
