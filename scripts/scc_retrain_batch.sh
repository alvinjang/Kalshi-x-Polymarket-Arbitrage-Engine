#!/bin/bash -l
#
# SCC batch job: retrain models on accumulated live + historical data.
#
# Runs on a compute node to avoid login-node CPU limits. Handles both:
#   1. Data-scaling experiment (for the paper's complexity analysis)
#   2. Live model export (LR + XGBoost pickles used by strategy.py)
#
# The script checks whether enough new bars have accumulated to justify
# retraining. If the data checkpoint hasn't advanced, it exits early
# (~10 seconds) so cron submissions are cheap.
#
# Submit manually:
#   qsub /usr4/ds340/iansabia/DS340-Project/scripts/scc_retrain_batch.sh
#
# Auto-submit via crontab (every 6 hours at :47):
#   47 */6 * * * /bin/bash -lc 'qsub /usr4/ds340/iansabia/DS340-Project/scripts/scc_retrain_batch.sh'
#
# Timing estimates (4 cores):
#   - Checkpoint check only (no new data): ~10 seconds
#   - Tier 1 retrain (LR + XGBoost):       ~2 minutes
#   - Tier 2 retrain (GRU + LSTM):         ~15-30 minutes
#   - Full retrain (Tier 1 + 2):           ~30-45 minutes
#
#$ -N retrain-models
#$ -j y
#$ -o $HOME/logs/
#$ -l h_rt=02:00:00
#$ -pe omp 4
#$ -l mem_per_core=4G

set -euo pipefail

PROJECT="/usr4/ds340/iansabia/DS340-Project"
PY="$PROJECT/.venv/bin/python"
LOGPREFIX="[$(date -u +%Y-%m-%dT%H:%M:%SZ)]"

log() { echo "$LOGPREFIX $*"; }

# Pin thread counts for PyTorch / numpy
export OMP_NUM_THREADS=${NSLOTS:-4}
export MKL_NUM_THREADS=${NSLOTS:-4}
export TORCH_NUM_THREADS=${NSLOTS:-4}
export PYTHONPATH="$PROJECT"

cd "$PROJECT"

# Pull latest code + data
log "git pull"
git pull -q --rebase --autostash || log "WARN: git pull failed — using local"

# ── Step 1: Data-scaling experiment (auto-checkpoint) ────────────────
#
# run_data_scaling.py --auto checks if bars/pair has crossed the next
# checkpoint (50, 100, 250, 500, 1000, 2000). If not, it exits
# immediately. If yes, it trains all Tier 1 models at the new
# checkpoint and (when bars >= 100) includes Tier 2 (GRU, LSTM).
#
log "running data-scaling experiment (auto checkpoint)"
if "$PY" scripts/run_data_scaling.py --auto --include-tier2 2>&1; then
    log "data-scaling: complete"
    SCALING_RAN=1
else
    log "data-scaling: no new checkpoint or error (exit $?) — continuing"
    SCALING_RAN=0
fi

# ── Step 2: Retrain deployed models for live trading ─────────────────
#
# Always retrain the pickles strategy.py uses (LR + XGBoost) on the
# COMBINED dataset: historical train.parquet + fresh live bars.
# This is cheap (~2 min) and ensures the live system improves as
# data accumulates.
#
log "retraining deployed models (LR + XGBoost)"
"$PY" -c "
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np

# ── Load historical data ──
hist_path = Path('data/processed/train.parquet')
if hist_path.exists():
    hist = pd.read_parquet(hist_path)
    print(f'  Historical: {len(hist)} rows, {hist[\"pair_id\"].nunique()} pairs')
else:
    hist = pd.DataFrame()
    print('  No historical data')

# ── Load fresh live bars (new content-addressed format only) ──
live_path = Path('data/live/bars.parquet')
if live_path.exists():
    live_all = pd.read_parquet(live_path)
    # Only new-format bars (exclude orphaned live_NNNN from pre-schema-fix)
    live = live_all[~live_all['pair_id'].str.startswith('live_')]
    print(f'  Live (new format): {len(live)} rows, {live[\"pair_id\"].nunique()} pairs')
    if len(live) == 0:
        print('  No new-format live bars yet — using historical only')
else:
    live = pd.DataFrame()
    print('  No live bars.parquet')

# ── Combine ──
if len(hist) > 0 and len(live) > 0:
    # Align columns
    common_cols = sorted(set(hist.columns) & set(live.columns))
    combined = pd.concat([hist[common_cols], live[common_cols]], ignore_index=True)
elif len(hist) > 0:
    combined = hist
elif len(live) > 0:
    combined = live
else:
    print('ERROR: no training data available')
    sys.exit(1)

print(f'  Combined: {len(combined)} rows, {combined[\"pair_id\"].nunique()} pairs')

# ── Feature engineering ──
from src.features.engineering import compute_derived_features
from src.features.schemas import ALIGNED_COLUMNS

# Ensure aligned columns present
for col in ALIGNED_COLUMNS:
    if col not in combined.columns:
        combined[col] = 0.0

combined = compute_derived_features(combined)
combined = combined.fillna(0.0)

# ── Train/export ──
from src.models.linear_regression import LinearRegressionPredictor
from src.models.xgboost_model import XGBoostPredictor

# Feature columns: exclude non-feature columns
EXCLUDE = {'timestamp', 'pair_id', 'group_id', 'time_idx', 'kalshi_has_trade',
           'polymarket_has_trade', 'spread', 'future_spread'}
feature_cols = [c for c in combined.columns if c not in EXCLUDE and combined[c].dtype in ('float64', 'float32', 'int64', 'int32', 'bool')]

# Target
if 'future_spread' in combined.columns:
    y = combined['future_spread']
elif 'spread' in combined.columns:
    y = combined['spread']
else:
    print('ERROR: no target column found')
    sys.exit(1)

X = combined[feature_cols]
mask = ~(X.isna().any(axis=1) | y.isna())
X = X[mask]
y = y[mask]
print(f'  Training on {len(X)} rows, {len(feature_cols)} features')

# Train (fit expects DataFrame, not numpy)
lr = LinearRegressionPredictor()
lr.fit(X, y)

xgb = XGBoostPredictor()
xgb.fit(X, y)

# Save
model_dir = Path('models/deployed')
model_dir.mkdir(parents=True, exist_ok=True)
lr.save(model_dir / 'linear_regression.pkl')
xgb.save(model_dir / 'xgboost.pkl')

with open(model_dir / 'feature_columns.json', 'w') as f:
    json.dump(feature_cols, f)

print(f'  Exported: {model_dir}/linear_regression.pkl, xgboost.pkl')
print(f'  Feature columns: {len(feature_cols)}')
" 2>&1
RETRAIN_STATUS=$?
if [ $RETRAIN_STATUS -eq 0 ]; then
    log "model export: success"
else
    log "model export: FAILED (exit $RETRAIN_STATUS)"
fi

# ── Step 3: Plot scaling results (if scaling ran) ────────────────────
if [ "$SCALING_RAN" -eq 1 ]; then
    log "plotting data-scaling results"
    "$PY" scripts/plot_data_scaling.py 2>&1 || log "WARN: plot failed"
fi

# ── Step 4: Commit and push ──────────────────────────────────────────
log "staging results"
git add models/deployed/*.pkl models/deployed/feature_columns.json 2>/dev/null || true
git add experiments/results/data_scaling/log.jsonl 2>/dev/null || true
git add experiments/results/data_scaling/state.json 2>/dev/null || true
git add 'experiments/results/data_scaling/*.png' 2>/dev/null || true

if git diff --cached --quiet; then
    log "nothing to commit"
else
    TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    git commit -q -m "auto: retrain models ${TIMESTAMP}"
    log "committed"

    # Rebase-retry push (same pattern as trading cycle)
    for attempt in 1 2 3; do
        if git push -q; then
            log "pushed (attempt $attempt)"
            break
        fi
        log "push rejected — rebasing (attempt $attempt)"
        git pull -q --rebase --autostash origin master || true
    done
fi

log "retrain batch complete"
