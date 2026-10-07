# Complexity Is Not an Edge

Cross-platform prediction-market arbitrage on Kalshi and Polymarket, DS340 final project, Boston University, Spring 2026.

**Authors:** Ian Sabia (U33871576), Alvin Jang (U64760665).

## What is this?

An empirical study asking whether increasing model complexity improves arbitrage detection across Kalshi and Polymarket. We train four tiers of models (Linear Regression, XGBoost → GRU, LSTM, TFT → PPO → PPO + autoencoder) on a common evaluation protocol and find that **the simplest models consistently dominate**. Linear Regression achieves a per-trade Sharpe of 0.501 with +15.0 bps per-trade alpha, tied with XGBoost; PPO + autoencoder collapses to +0.5 bps — 30× less edge for substantially more compute.

## Where to look

- **`paper.pdf`** — final research paper (29 pages, 11 figures, 10 tables; rendered from `PAPER_DRAFT.md`). **Start here.**
- **`PAPER_DRAFT.md`** — markdown source of the paper (canonical text)
- **`AUDIT_REPORT.md`** — adversarial audit report (Phase 18). Six-tier verification of every quantitative claim, including a leakage-free recompute on a pair-stratified split.
- **`slides_deck.pptx`** — 7-slide lightning talk (4-minute, two-presenter format; open in PowerPoint or Keynote).

## Quick start

```bash
git clone https://github.com/iansabia/DS340-Project.git
cd DS340-Project
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

Python 3.12+ is required. All experiments run on CPU; no GPU needed. Total runtime for the full reproduction pipeline is ≈ 2–3 hours on a single CPU core.

## Reproducing every paper table and figure

| Paper object | Command | Output |
|---|---|---|
| Table 2 (headline backtest) | `python experiments/verify_headline.py` | `experiments/results/tier1/*.json`, `experiments/results/verify_headline.json` |
| Table 3 + Table 3b + Table 4 + Fig. 1 + Fig. 3 (walk-forward) | `python experiments/run_walk_forward.py --windows 12` | `experiments/results/walk_forward/*.json` + `experiments/figures/walk_forward_pnl.png` + `experiments/figures/walk_forward_sharpe.png` |
| §5.3 per-category breakdown | `python experiments/run_category_breakdown.py` | `experiments/results/category_breakdown.json` |
| Table 5 + Fig. 2 (data-scaling curve) | `python scripts/run_data_scaling.py --bars-per-pair 250` | `experiments/results/data_scaling/*.json` + `experiments/results/data_scaling/pnl_at_2pp_vs_data.png` |
| Table 6 (XGBoost hyperparameter sweep, 48 configs) | Pre-computed artifact at `experiments/results/xgb_hyperparam_sweep.json`; cross-tier aggregation via `python experiments/run_experiment1_comparison.py` | `experiments/results/experiment1/*.json` |
| Table 7 + Fig. 4 (transaction costs) | `python experiments/run_transaction_costs.py` | `experiments/figures/transaction_cost_sensitivity.png` |
| Table 8 (honest Sharpe accounting) | Derived from Table 2 outputs — fields `sharpe_per_trade` and `sharpe_per_pair` in `experiments/results/verify_headline.json` | — |
| §5.9 live vs backtest reconciliation | `python experiments/run_live_reconciliation.py` | `experiments/results/reconciliation/*` |
| Table 9 (feature ablation, LOGO) | `python experiments/run_feature_ablation.py` | `experiments/results/ablation/*.json` |
| Table 10 + Fig. 11 (ensemble variants + weight sweep) | `python experiments/run_ensemble_sweep.py` | `experiments/results/ensemble/summary.json` + `experiments/figures/ensemble_weight_sweep.png` |
| Fig. 5 (SHAP bar plot) | `python experiments/run_shap_analysis.py` | `experiments/figures/shap_bar_plot.png` |
| Fig. 6 (equity curves) | `python experiments/run_backtest.py` | `experiments/figures/backtest_equity_curves.png` |
| Fig. 7 (bootstrap RMSE CI) | `python experiments/run_bootstrap_ci.py` | `experiments/figures/bootstrap_ci_rmse.png` |
| Fig. 8 (lookback sweep — Experiment 2) | `python experiments/run_experiment2_lookback.py` | `experiments/figures/experiment2_lookback_pnl.png` |
| Fig. 9 (threshold heatmap — Experiment 3) | `python experiments/run_experiment3_threshold.py` | `experiments/figures/experiment3_threshold_heatmap.png` |
| Fig. 10 (TFT VSN heatmap) | `python experiments/run_tft.py && python experiments/extract_tft_heatmap.py` | `experiments/figures/tft_variable_importance.png` + `experiments/results/tft/vsn_importance.json` |
| All figures (re-render in IEEE style) | `python scripts/regenerate_figures.py` | `experiments/figures/*.png` + `experiments/results/data_scaling/pnl_at_2pp_vs_data.png` |

## Paper integrity check

```bash
bash scripts/check_paper.sh
```

Exits 0 when the paper passes all POL-04 through POL-10 grep validators (abstract ≤ 250 words, no duplicate table numbers, no dead cross-references, no residual placeholders).

## Live paper-trading system

The autonomous system deployed on BU SCC is documented separately; see `docs/SCC_DEPLOYMENT.md` if present or the §4.4 Live System Architecture section of `PAPER_DRAFT.md`.

## Project structure

```
src/                          # Model code, feature pipeline, matching, live system
experiments/                  # Runner scripts for each paper table/figure
experiments/audit/            # Phase 18 audit scripts (Sharpe, leakage, costs, etc.)
experiments/results/          # JSON outputs from experiment runs
experiments/results/canonical/         # Canonical headline.json (single source of truth for paper numbers)
experiments/results/canonical_purged/  # Leakage-free re-run on pair-stratified split (Phase 18)
experiments/results/audit/             # Per-tier audit JSON outputs + paper_numbers.csv trace
experiments/figures/          # PNG figures (IEEE style, 300 DPI)
scripts/                      # Ops scripts (data scaling, figure regeneration, paper checks)
tests/                        # pytest suite (incl. tests/audit/ for Phase 18 audit fixtures)
data/raw/                     # Raw API dumps (Kalshi + Polymarket)
data/processed/               # Aligned feature dataframes
data/processed/purged_split/  # Pair-stratified train/test split (Phase 18, leakage-free)
```

## Code authorship & AI attribution

Three distinct categories of contribution:

1. **Research decisions** — purely human (Ian Sabia, Alvin Jang). The central research question, the four-tier complexity comparison, the hyperparameters we swept, the evaluation regimes (single-split, walk-forward, data-scaling, transaction-cost sensitivity, live deployment, audit), the matching-pipeline rules, the reward function for PPO, and the interpretation of every result are ours.

2. **Model implementations (`src/models/`)** — mostly human with AI assistance. The model wrappers around `scikit-learn`, `xgboost`, `pytorch`, `pytorch-forecasting`, and `stable_baselines3` are predominantly hand-written by the human authors. AI (Anthropic Claude via Claude Code) assisted alongside us as a coding helper for syntax, refactoring, and debugging, but did not drive the implementation.

3. **Supporting infrastructure (everything else)** — mostly AI-generated under human direction. The data ingestion adapters, market matching pipeline, feature engineering, evaluation utilities, live trading system, audit pipeline, tests, and operational scripts were generated by Claude Code from human-specified requirements, with every change reviewed, edited, and tested by the human authors. This is the bulk of the codebase by line count and is the layer that made the one-month timeline achievable.

Each package's `__init__.py` carries a short docstring identifying its authorship category. No code was copied from online sources; all third-party libraries are used per their published licenses (Apache 2.0, BSD-3, MIT, NCSA).

## Audit + reproducibility

Every numeric claim in `PAPER_DRAFT.md` is traced to a canonical file in `experiments/results/audit/paper_numbers.csv`. Re-running `bash scripts/check_paper.sh` validates 26 regression checks against the paper text. The leakage-free recompute that produced the final headline numbers is in `experiments/audit/audit_sharpe_purged.py`; see `AUDIT_REPORT.md` for the full methodology.
