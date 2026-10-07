# Archived non-canonical PPO results

These files were produced by the legacy `experiments/run_backtest.py` path
on 2026-04-06 and show PPO P&L magnitudes inconsistent with the tier3-path
results in `experiments/results/tier3/ppo_*.json`:

| File | total_pnl | num_trades | total_fees |
|------|-----------|-----------|------------|
| `backtest_ppo_raw.json` | +$96,336.84 | 1637 | $96,078.19 |
| `backtest_ppo_filtered.json` | −$87,723.84 | 817 | $27,370.81 |

The canonical (tier3) figures are:

| File | total_pnl | num_trades |
|------|-----------|-----------|
| `experiments/results/tier3/ppo_raw.json` | +$158.15 | 1656 |
| `experiments/results/tier3/ppo_filtered.json` | +$4.61 | 899 |

PPO-Raw legacy/canonical ratio: 609.14×.

## Why the divergence

`experiments/run_backtest.py` invokes `WalkForwardBacktester(position_size=$100)`
which sizes each trade as `num_contracts = $100 / mid_price ≈ 200` and applies
a 5pp round-trip transaction cost. `experiments/run_baselines.py --tier 3`
invokes `simulate_profit()` which returns raw spread-units P&L with no
position scaling and no fees. Multiplying the raw spread-units PPO-Raw P&L
($158.15) by ~200 contracts and adjusting for fees (which dominate the legacy
output: $96,078 fees on $96,336 gross P&L) yields the legacy magnitude.

Full root-cause writeup:
`.planning/phases/17-model-rerun-paper-number-audit-pitch-standard-conversion/17-02-PPO-DIAGNOSTIC.md`

## Single source of truth

The Phase 17 canonical results JSON is
`experiments/results/canonical/headline.json` (produced by
`experiments/run_canonical.py`). Per REPL-07, no metric in any paper,
slide, or downstream artifact may cite a JSON file outside
`experiments/results/canonical/`.

**DO NOT cite the files in this directory.** They are retained only for
historical reference and so the 17-02 diagnostic can quote them verbatim.
