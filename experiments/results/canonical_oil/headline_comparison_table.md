# Headline Comparison Table (Table 1 style)

Oil-only canonical retraining, headline run (50-bar threshold) and robustness rerun (100-bar threshold). Each row is one model tier, ranked by headline per-trade Sharpe. CIs are 95% from 10,000 bootstrap resamples of per-trade outcomes.

Trading rule: take a position in the direction of the model's prediction whenever |prediction| > 0.001. The threshold is scale-equivalent to the original paper's 0.02 after accounting for the roughly 10x smaller target standard deviation at 15-minute bar granularity (target std 0.0286 here vs 0.306 in the original).

## Headline (50-bar threshold: 154 train pairs, 39 test pairs, 24,655 total rows)

| Rank | Model | Trades | Per-trade Sharpe (95% CI) | Alpha (bps) (95% CI) | P&L $ (95% CI) | RMSE | Dir Acc | Win Rate |
|---|---|---|---|---|---|---|---|---|
| 1 | XGBoost | 4580 | +0.0644 [+0.0364, +0.0903] | +18.96 [+10.44, +27.62] | +8.68 [+4.78, +12.65] | 0.0325 | 0.5216 | 0.3371 |
| 2 | PPO | 4418 | +0.0117 [-0.0174, +0.0410] | +3.41 [-5.13, +12.00] | +1.51 [-2.27, +5.30] | 0.0395 | 0.5362 | 0.3407 |
| 3 | Linear Regression | 2394 | +0.0097 [-0.0324, +0.0477] | +3.12 [-9.57, +16.45] | +0.75 [-2.29, +3.94] | 0.0287 | 0.4966 | 0.3530 |
| 4 | GRU | 4362 | +0.0081 [-0.0214, +0.0376] | +2.39 [-6.18, +11.21] | +1.04 [-2.70, +4.89] | 0.0288 | 0.5338 | 0.3423 |
| 5 | LSTM | 3642 | +0.0007 [-0.0332, +0.0322] | +0.20 [-8.70, +9.11] | +0.07 [-3.17, +3.32] | 0.0288 | 0.4936 | 0.3078 |

## Robustness (100-bar threshold: 69 train pairs, 20 test pairs)

| Rank | Model | Trades | Per-trade Sharpe (95% CI) | Alpha (bps) (95% CI) | P&L $ (95% CI) | RMSE | Dir Acc | Win Rate |
|---|---|---|---|---|---|---|---|---|
| 1 | XGBoost | 3669 | +0.0736 [+0.0455, +0.1012] | +20.20 [+11.74, +29.39] | +7.41 [+4.31, +10.79] | 0.0433 | 0.5458 | 0.3312 |
| 2 | PPO | 4052 | +0.0247 [-0.0062, +0.0554] | +6.62 [-1.62, +15.17] | +2.68 [-0.66, +6.15] | 0.0397 | 0.5293 | 0.3139 |
| 3 | Linear Regression | 1878 | +0.0149 [-0.0323, +0.0543] | +4.11 [-7.65, +16.89] | +0.77 [-1.44, +3.17] | 0.0268 | 0.4861 | 0.3019 |
| 4 | GRU | 3490 | +0.0120 [-0.0213, +0.0445] | +2.98 [-5.21, +11.26] | +1.04 [-1.82, +3.93] | 0.0280 | 0.5226 | 0.3086 |
| 5 | LSTM | 2919 | +0.0069 [-0.0317, +0.0406] | +1.86 [-7.90, +11.84] | +0.54 [-2.31, +3.46] | 0.0269 | 0.5127 | 0.2830 |

## Side-by-side: per-trade Sharpe with bootstrap CI

| Model | Headline Sharpe (CI) | Robustness Sharpe (CI) | Consistent rank? |
|---|---|---|---|
| Linear Regression | +0.0097 [-0.0324, +0.0477] | +0.0149 [-0.0323, +0.0543] | yes |
| XGBoost | +0.0644 [+0.0364, +0.0903] | +0.0736 [+0.0455, +0.1012] | yes |
| GRU | +0.0081 [-0.0214, +0.0376] | +0.0120 [-0.0213, +0.0445] | yes |
| LSTM | +0.0007 [-0.0332, +0.0322] | +0.0069 [-0.0317, +0.0406] | yes |
| PPO | +0.0117 [-0.0174, +0.0410] | +0.0247 [-0.0062, +0.0554] | yes |
