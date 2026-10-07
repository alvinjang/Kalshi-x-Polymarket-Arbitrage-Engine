# Phase A — Canonical-Oil XGBoost Live Deployment (v3)

**Status:** spec only, NOT authorized for execution. Awaiting stability watch (5+ clean cycles post-Q1/Q2) before user authorization.

**Goal:** deploy the canonical-oil XGBoost (headline +18.96 bps, matched-aggregation Sharpe 0.49) into the live SCC paper-trading loop in narrow-replacement mode on the oil family. Accumulate ~3-4 weeks of forward returns to attach to the Gold email (target send date 2026-07-01).

**Hard turn limit when executed:** 500. Estimated focused time: 30-90 minutes.

---

## What changed since v2

This is the first spec written down after the Q1/Q2 architecture migration (2026-05-23). v1 and v2 existed only conversationally. The substantive deltas from those earlier sketches:

1. **bars.parquet is now off-tree.** Live inference reads `data/live/bars.parquet` which is a symlink to `/projectnb/ds340/projects/iansabia/live_state/bars.parquet`. **Phase A code MUST use the canonical symlink path, never hardcode `/projectnb/...`.** This keeps the access pattern identical to GHA/local-dev and decouples training-time access (`data/processed/canonical_oil/train.parquet`, unchanged, in-tree) from live access.
2. The Polymarket pagination regression is fixed (commit `12c83426`); the oil universe has revived to ~130 alive pairs. Shadow-mode parity therefore runs against a realistic-shaped live feed.
3. Layer 4 staging guards are in place on both `scc_trading_cycle.sh` and `scc_discover_markets.sh`, so any new artifact Phase A writes under `data/live/` should either be added to `.gitignore` (if large/binary/per-cycle) or stay small enough to commit cleanly.

---

## (a) Export step — paths and access pattern

**Outputs land under `models/canonical_oil/`** (NEW directory, parallel to the existing `models/deployed/` which keeps serving the 4-tier paper-trader untouched):

```
models/canonical_oil/
├── xgboost.pkl                 # the canonical headline model
├── feature_columns.json        # 49 cols, exact order from training
├── zero_variance_columns.json  # 30 cols to force-zero at inference time
├── metadata.json               # train sha, canonical-split sha, eval metrics, export ts
└── README.md                   # one-pager: this is the canonical artifact for Phase A
```

The existing `models/deployed/xgboost.pkl` (54-col, older snapshot) is left in place and untouched. Narrow-replacement (section d) routes only oil-family pairs to the canonical model; everything else continues hitting the existing deployed models. This protects the legacy paper-trader's continuity and lets us A/B the two systems on the same live feed.

**Live read path (load once at process start, NOT per-cycle):**

```python
# src/live/canonical_inference.py — NEW MODULE
from pathlib import Path

LIVE_BARS_PATH = Path("data/live/bars.parquet")  # canonical symlink path
CANONICAL_MODEL_DIR = Path("models/canonical_oil")
```

`LIVE_BARS_PATH` is the in-tree symlink. Resolves to the off-tree real file via the symlink — same code works on SCC, GHA, and local. Never reference `/projectnb/...` from Python.

---

## (b) Feature column handling — 49 active + 30 zero-variance

Canonical training used 49 features after dropping 30 columns that were zero-variance on the oil-only training set. At live inference we **rebuild a 49-column row in the training feature order, force-zero any column that's in the zero-variance list, and pass to the model.**

**Generation step (one-shot, during export):** dump the active-49 set and the forced-zero-30 set as two JSON files alongside the pickle. Both are derived from the canonical training artifact, NOT recomputed from scratch:

```python
# scripts/export_canonical_oil.py — NEW SCRIPT
import json
import pickle
import pandas as pd
from pathlib import Path

CANON = Path("data/processed/canonical_oil/train.parquet")
OUT = Path("models/canonical_oil")
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_parquet(CANON)
NON_FEATURE = {"pair_id", "y", "ts", "split", "category", ...}  # to be confirmed against canonical schema
all_cols = [c for c in train.columns if c not in NON_FEATURE]
variances = train[all_cols].var(numeric_only=True)
active = [c for c in all_cols if variances.get(c, 0) > 0]
zero_var = [c for c in all_cols if variances.get(c, 0) == 0]
assert len(active) == 49 and len(zero_var) == 30, (len(active), len(zero_var))

# … fit / load existing canonical XGBoost, pickle it
json.dump(active,  open(OUT / "feature_columns.json", "w"))
json.dump(zero_var, open(OUT / "zero_variance_columns.json", "w"))
```

**Inference step:**

```python
ACTIVE = json.load(open(CANONICAL_MODEL_DIR / "feature_columns.json"))
ZEROS  = set(json.load(open(CANONICAL_MODEL_DIR / "zero_variance_columns.json")))

def build_row(bar: dict) -> np.ndarray:
    return np.array([
        0.0 if c in ZEROS else float(bar.get(c, 0.0))
        for c in ACTIVE
    ], dtype=np.float32)
```

If any of the 30 zero-variance columns develops nonzero variance in live data, that's a **distribution-shift alarm** worth logging (section f), but the model still scores it as zero — that's the contract that matches training.

---

## (c) Shadow mode (week 1) — non-negotiable

**Phase A1 (≈24h shadow):** canonical XGBoost runs alongside the existing models on every 15-min cycle but writes its predictions to a separate log (`data/live/canonical_predictions.jsonl`). No orders, no positions. Existing system continues unmodified.

**Parity gate (must pass before A2):**

Re-load `data/processed/canonical_oil/test.parquet`, run the live-mode inference helper on every row, and assert **byte-for-byte (bitwise float32) equality** with the training-time prediction in `experiments/results/canonical/`. Even one mismatch blocks promotion to A2.

```python
# scripts/verify_canonical_parity.py — NEW SCRIPT
import numpy as np
# compare per-row predictions
diff = np.abs(live_preds - training_preds)
assert diff.max() == 0.0, f"parity break: max abs diff {diff.max()}, n_mismatch {(diff != 0).sum()}"
```

Failures here usually mean one of:
- column order drift (active list out of sync with pickle)
- a zero-variance column has a nonzero value in test that we're failing to force-zero
- numpy/xgboost version skew between training and inference environments

All three must be ruled out before A2.

**Phase A2 (narrow replacement, oil family only):** after shadow + parity pass, route oil-family pair predictions to the canonical model (section d). Original LR/XGBoost continue scoring everything else.

---

## (d) Pair admission filter — oil family only for canonical

```python
# src/live/canonical_inference.py
OIL_FAMILY_PREFIXES = (
    "KXWTI",       # WTI weekly + monthly
    "KXBRENT",     # Brent monthly
    "KXCRUDE",
    "KXDIESEL",
    "KXHEATINGOIL",
    "KXGASOLINE",
    "KXMEXCUBOIL",
)

def use_canonical(kalshi_ticker: str) -> bool:
    return kalshi_ticker.startswith(OIL_FAMILY_PREFIXES)
```

In the trading loop:

```python
if use_canonical(pair.kalshi_ticker):
    pred = canonical_xgb.predict(build_row(bar))[0]
    threshold = 0.001   # canonical scale
else:
    pred = original_models.predict(...)
    threshold = 0.02    # legacy scale, unchanged
```

**The original LR continues running on ALL categories** (oil included — it sits in the legacy path too, providing stability and a known baseline). Narrow replacement only swaps the XGBoost decision on oil-family tickers.

---

## (e) Trade thresholds

| Model path | Threshold | Source |
|---|---|---|
| Canonical XGBoost (oil family) | **0.001** | scale-equivalent adaptation, RESULTS_OIL_RETRAIN_DRAFT §writeup |
| Original LR (all categories) | 0.02 | unchanged from legacy paper-trader |
| Original XGBoost (non-oil categories) | 0.02 | unchanged |

The 20× threshold gap reflects the canonical model's prediction-scale shift documented in the writeup. Do not "round" 0.001 up — it was calibrated against the canonical test slice.

---

## (f) LIVE_TRADING guard — explicit gating

The existing real-money guard pattern (`LIVE_TRADING=false` env var, paper-only by default) extends to Phase A. New env var to add:

```
CANONICAL_OIL_ENABLED=false   # default OFF; flip to true at A2 cutover
CANONICAL_OIL_SHADOW=true     # default TRUE during A1; flip to false at A2
```

State machine:

| Env | Mode | Behavior |
|---|---|---|
| `CANONICAL_OIL_ENABLED=false`, `_SHADOW=false` | OFF | canonical model not loaded; legacy system runs alone |
| `_ENABLED=false`, `_SHADOW=true` | A1 shadow | canonical scores logged but never acted on |
| `_ENABLED=true`, `_SHADOW=false` | A2 narrow | canonical scores route oil-family decisions; logged separately |
| `_ENABLED=true`, `_SHADOW=true` | dual-write | A2 with shadow log kept on for parity drift tracking |

Add `CANONICAL_OIL_ENABLED` and `CANONICAL_OIL_SHADOW` to the `.github/workflows/collect-and-trade.yml` env block (default both false so GHA fallback never trades canonical until SCC has proven it). SCC crontab gets them exported in the trading-cycle wrapper.

**Independent of `LIVE_TRADING`** — Phase A in any mode is paper-only; real-money authorization is a separate, later decision tied to live retrospective results.

---

## (g) Budget

- Export + parity verifier wiring: ~15-25 min
- Shadow-mode plumbing (logging, env gates): ~15-25 min
- Monitoring dashboard hooks (signal log, parity drift counter): ~15-25 min
- Sanity sweep + smoke commit: ~5-15 min

**Total: 30-90 min focused /goal time. Hard turn limit: 500.**

The /goal should produce a single commit (or small commit chain) ending with the system in A1 (shadow on, enabled off). Cutover to A2 is a manual flip after 24h of shadow + parity pass.

---

## Dependencies / preconditions

| Gate | Required state |
|---|---|
| Stability watch | 5+ clean cron cycles post-Q1/Q2 with no GUARD/FATAL log lines |
| Q1/Q2 architecture | DONE (this thread, commits `1ef9df7` → `3df2a3c` → `fa33ca79`) |
| Polymarket filter fix | DONE (commit `12c83426`) |
| Canonical artifacts | `data/processed/canonical_oil/{train,test}.parquet` + `experiments/results/canonical/headline.json` present (verified) |
| User authorization | NOT GRANTED — waiting on stability watch |

---

## Out of scope for Phase A (deferred to later phases)

- **Real-money trading** with the canonical model. Phase A is paper-only.
- **GRU/LSTM/PPO deployment**. Phase A is the XGBoost headline model only; sequence-model deployment requires its own spec.
- **Retraining on rolling windows.** Canonical model is fixed at the audit-pinned weights; no online updates during the 3-4 week forward-data accumulation.
- **Threshold sweep / hyperparameter tuning.** Whatever the writeup pinned is what gets deployed. Tuning post-hoc on live data is the exact kind of selection bias the writeup argued against.

---

## Failure-mode rollback

If parity check fails, A1 logs alarm, or A2 starts producing per-cycle errors:

1. Set `CANONICAL_OIL_ENABLED=false` and `CANONICAL_OIL_SHADOW=false` in `.env` on SCC and in the GHA workflow.
2. Next cycle (≤15 min) will run with the legacy system only — no commit revert needed.
3. Diagnose at leisure; canonical model artifacts stay in place for re-enabling.

This is intentional — the env-gate design means rollback is reversible by flipping two variables, not by reverting code.
