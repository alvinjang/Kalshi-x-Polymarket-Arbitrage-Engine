# Full Retrain Report

Generated: 2026-05-13T00:28:48.796190+00:00

## Dataset
- Combined rows: **323,744**
- Pairs: **6,850**
- Train/test split: time-aware, 80/20 per pair
- Train rows: 256,734 / Test rows: 67,010
- Features: 54

## Tier 1 vs Deployed (held-out test set)

| Model | RMSE | MAE | dir_acc | PnL@3pp | Notes |
|---|---|---|---|---|---|
| LR (deployed) | 0.0673 | 0.0196 | 0.5290 | $+294.28 | n=67010 |
| LR (full retrain) | 0.0673 | 0.0196 | 0.5290 | $+294.28 | n=67010 |
| XGB (deployed) | 0.0681 | 0.0191 | 0.5458 | $+295.14 | n=67010 |
| XGB (full retrain) | 0.0681 | 0.0191 | 0.5458 | $+295.14 | n=67010 |

## Tier 2 (sequence models, ≥100-bar cohort)

| Model | RMSE | dir_acc | PnL@3pp |
|---|---|---|---|
| GRU | 0.0544 | 0.5262 | $+51.87 |
| LSTM | 0.0551 | 0.5296 | $+61.98 |

## Deploy decisions
- ✓ `xgboost`: **deployed**
- ✓ `linear_regression`: **deployed**

## Interpretation
- GRU vs XGBoost (full): PnL@3pp delta $-243.27
- LSTM vs XGBoost (full): PnL@3pp delta $-233.16
