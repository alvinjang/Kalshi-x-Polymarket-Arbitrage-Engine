"""Phase 13 ensemble sweep experiment runner (ENSM-02, ENSM-03, ENSM-04, ENSM-06).

Evaluates 4 ensemble variants, runs a concordance filter audit on each, and
produces the 11-point LR-weight sensitivity sweep for the LR+XGB pair.
Outputs are consumed by Plan 13-03's §5.11 paper section.

Pipeline (mirrors ``experiments/verify_headline.py``):
    1. Load ``data/processed/{train,test}.parquet`` and apply ``build()``.
    2. Derive flat and sequence feature views (per verify_headline.py lines 77-86).
    3. Evaluate 4 variants with per-member feature routing (RESEARCH.md Pitfall 2).
    4. Concordance audit per variant (filtered / unfiltered / rejected P&L).
    5. 11-point LR-weight sweep (0.0 → 1.0 step 0.1), filtered and unfiltered.
    6. Save ``experiments/results/ensemble/summary.json`` and
       ``experiments/figures/ensemble_weight_sweep.png``.

ENSM-05 safety: this script MUST NOT import from ``src/live/strategy.py`` and
MUST NOT modify it. Verification: ``git diff src/live/strategy.py`` is empty
after running.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.verify_headline import build, feature_cols, simulate_pnl
from src.evaluation.results_store import save_results
from src.models.base import BasePredictor
from src.models.ensemble import EnsemblePredictor
from src.models.linear_regression import LinearRegressionPredictor
from src.models.lstm import LSTMPredictor
from src.models.xgboost_model import XGBoostPredictor
from src.utils.seed import set_all_seeds

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("experiments/results/ensemble")
FIGURES_DIR = Path("experiments/figures")
FEE = 0.02
TARGET = "spread_change_target"
SEED = 42

# Member types that consume the sequence feature view (group_id present).
_SEQUENCE_MEMBER_TYPES = {"GRUPredictor", "LSTMPredictor"}


def _needs_seq(member: BasePredictor) -> bool:
    return type(member).__name__ in _SEQUENCE_MEMBER_TYPES


def fit_mixed_ensemble(
    members: Sequence[BasePredictor],
    X_flat: pd.DataFrame,
    X_seq: pd.DataFrame,
    y: np.ndarray,
) -> None:
    """Fit each member on its appropriate feature view.

    Sequence members (GRU/LSTM) receive ``X_seq`` (includes ``group_id``);
    flat members (LR/XGB) receive ``X_flat`` (excludes ``group_id``).
    Mitigates RESEARCH.md Pitfall 2 (``group_id`` as a latent feature).
    """
    for member in members:
        X_for_member = X_seq if _needs_seq(member) else X_flat
        member.fit(X_for_member, y)


def predict_mixed_members(
    members: Sequence[BasePredictor],
    X_flat: pd.DataFrame,
    X_seq: pd.DataFrame,
) -> List[np.ndarray]:
    """Collect per-member predictions with per-member feature routing."""
    out: List[np.ndarray] = []
    for member in members:
        X_for_member = X_seq if _needs_seq(member) else X_flat
        out.append(np.asarray(member.predict(X_for_member), dtype=float))
    return out


def combine_predictions(
    member_preds: Sequence[np.ndarray],
    weights: Sequence[float],
    concordance_mode: str,
) -> np.ndarray:
    """Weighted-average combine with optional strict concordance gate.

    Parallels ``EnsemblePredictor.predict`` but operates on pre-computed
    per-member prediction arrays so mixed-feature-routing variants can
    reuse the same math without piping a single DataFrame through the
    ensemble class.
    """
    stacked = np.stack(member_preds)  # (n_members, n_rows)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    weighted = (w[:, None] * stacked).sum(axis=0)
    if concordance_mode == "strict":
        signs = np.sign(stacked)
        agree = np.all(signs == signs[[0], :], axis=0)
        return np.where(agree, weighted, 0.0)
    return weighted


def concordance_audit(
    member_preds: Dict[str, np.ndarray],
    actuals: np.ndarray,
    fee: float = FEE,
) -> dict:
    """Compute filtered, unfiltered, and rejected P&L for a set of members.

    Implements the ENSM-03 audit schema: equal-weight average, all-members-
    agree concordance gate, counterfactual P&L on rejected trades, rejection
    rate, and the P4 flag (rejected trades net profitable).
    """
    stacked = np.stack(list(member_preds.values()))
    n_members = stacked.shape[0]
    weights = np.ones(n_members) / n_members
    avg_pred = (weights[:, None] * stacked).sum(axis=0)
    agree_mask = np.all(np.sign(stacked) == np.sign(stacked[[0]]), axis=0)

    unfiltered = simulate_pnl(avg_pred, actuals, fee)
    filtered = simulate_pnl(np.where(agree_mask, avg_pred, 0.0), actuals, fee)
    rejected = simulate_pnl(np.where(~agree_mask, avg_pred, 0.0), actuals, fee)

    denom = max(unfiltered["num_trades"], 1)
    rejection_rate = 1.0 - filtered["num_trades"] / denom
    flag = rejected["pnl"] > 0
    if flag:
        print(
            f"WARNING: concordance filter is rejecting profitable trades "
            f"(rejected P&L = ${rejected['pnl']:+.2f}) — P4 concordance trap active"
        )

    return {
        "filtered": filtered,
        "unfiltered": unfiltered,
        "rejected": rejected,
        "rejection_rate": round(float(rejection_rate), 4),
        "flag_rejected_profitable": bool(flag),
    }


def _variant_record(
    name: str,
    members: Sequence[BasePredictor],
    weights: Sequence[float],
    concordance_mode: str,
    audit: dict,
) -> dict:
    return {
        "name": name,
        "concordance_mode": concordance_mode,
        "members": [type(m).__name__ for m in members],
        "weights": [float(w) for w in weights],
        "filtered": audit["filtered"],
        "unfiltered": audit["unfiltered"],
        "rejected": audit["rejected"],
        "rejection_rate": audit["rejection_rate"],
        "flag_rejected_profitable": audit["flag_rejected_profitable"],
    }


def _print_audit_table(variants: Sequence[dict]) -> None:
    print("\n## Concordance Audit Table\n")
    header = (
        "| Variant | # trades (filtered) | # trades (unfiltered) | "
        "Rejection rate | P&L (filtered) | P&L (unfiltered) | "
        "P&L (rejected) | P4 flag |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for v in variants:
        f_trades = v["filtered"]["num_trades"]
        u_trades = v["unfiltered"]["num_trades"]
        rej = v["rejection_rate"]
        f_pnl = v["filtered"]["pnl"]
        u_pnl = v["unfiltered"]["pnl"]
        r_pnl = v["rejected"]["pnl"]
        flag = "WARN" if v["flag_rejected_profitable"] else "ok"
        print(
            f"| {v['name']} | {f_trades} | {u_trades} | {rej:.2%} | "
            f"${f_pnl:+.2f} | ${u_pnl:+.2f} | ${r_pnl:+.2f} | {flag} |"
        )


def _run_variant_a(X_flat_tr, X_flat_te, y_tr, y_te) -> dict:
    """(a) LR alone — concordance 'none'."""
    set_all_seeds(SEED)
    lr = LinearRegressionPredictor()
    lr.fit(X_flat_tr, y_tr)
    preds = lr.predict(X_flat_te)
    audit = concordance_audit({"lr": preds}, y_te, fee=FEE)
    return _variant_record(
        "(a) LR alone", [lr], [1.0], "none", audit
    )


def _run_variant_b(X_flat_tr, X_flat_te, y_tr, y_te) -> dict:
    """(b) LR + XGB equal-weight, concordance strict."""
    set_all_seeds(SEED)
    lr = LinearRegressionPredictor()
    xgb = XGBoostPredictor()
    lr.fit(X_flat_tr, y_tr)
    xgb.fit(X_flat_tr, y_tr)
    preds = {"lr": lr.predict(X_flat_te), "xgb": xgb.predict(X_flat_te)}
    audit = concordance_audit(preds, y_te, fee=FEE)
    return _variant_record(
        "(b) LR + XGB equal-weight", [lr, xgb], [0.5, 0.5], "strict", audit
    )


def _run_variant_c(X_flat_tr, X_flat_te, X_seq_tr, X_seq_te, y_tr, y_te) -> dict:
    """(c) LR + LSTM equal-weight, per-member routing, concordance strict.

    LR fits on X_flat (no group_id); LSTM fits on X_seq (includes group_id).
    """
    set_all_seeds(SEED)
    members: List[BasePredictor] = [LinearRegressionPredictor(), LSTMPredictor()]
    fit_mixed_ensemble(members, X_flat_tr, X_seq_tr, y_tr)
    test_preds = predict_mixed_members(members, X_flat_te, X_seq_te)
    audit = concordance_audit(
        {"lr": test_preds[0], "lstm": test_preds[1]}, y_te, fee=FEE
    )
    record = _variant_record(
        "(c) LR + LSTM equal-weight", members, [0.5, 0.5], "strict", audit
    )
    # Attach per-member P&L for the sanity cross-check with variant (a).
    record["member_lr_pnl"] = simulate_pnl(test_preds[0], y_te, fee=FEE)["pnl"]
    return record


def _run_variant_d(X_flat_tr, X_flat_te, X_seq_tr, X_seq_te, y_tr, y_te) -> dict:
    """(d) LR + XGB + LSTM, equal thirds, per-member routing, concordance strict."""
    set_all_seeds(SEED)
    members: List[BasePredictor] = [
        LinearRegressionPredictor(),
        XGBoostPredictor(),
        LSTMPredictor(),
    ]
    fit_mixed_ensemble(members, X_flat_tr, X_seq_tr, y_tr)
    test_preds = predict_mixed_members(members, X_flat_te, X_seq_te)
    audit = concordance_audit(
        {"lr": test_preds[0], "xgb": test_preds[1], "lstm": test_preds[2]},
        y_te,
        fee=FEE,
    )
    third = 1.0 / 3.0
    return _variant_record(
        "(d) LR + XGB + LSTM strict",
        members,
        [third, third, third],
        "strict",
        audit,
    )


def _run_weight_sweep(X_flat_tr, X_flat_te, y_tr, y_te) -> List[dict]:
    """11-point LR-weight sweep on LR+XGB, filtered and unfiltered in parallel."""
    sweep: List[dict] = []
    weights = np.round(np.arange(0.0, 1.01, 0.1), 2)
    for w in weights:
        lr_w = float(w)
        xgb_w = float(1.0 - w)

        set_all_seeds(SEED)
        ens_filtered = EnsemblePredictor(
            [LinearRegressionPredictor(), XGBoostPredictor()],
            weights=[lr_w, xgb_w],
            concordance_mode="strict",
        )
        ens_filtered.fit(X_flat_tr, y_tr)
        preds_f = ens_filtered.predict(X_flat_te)

        set_all_seeds(SEED)
        ens_unfiltered = EnsemblePredictor(
            [LinearRegressionPredictor(), XGBoostPredictor()],
            weights=[lr_w, xgb_w],
            concordance_mode="none",
        )
        ens_unfiltered.fit(X_flat_tr, y_tr)
        preds_u = ens_unfiltered.predict(X_flat_te)

        pnl_f = simulate_pnl(preds_f, y_te, fee=FEE)
        pnl_u = simulate_pnl(preds_u, y_te, fee=FEE)

        sweep.append(
            {
                "lr_weight": lr_w,
                "pnl_filtered": pnl_f["pnl"],
                "pnl_unfiltered": pnl_u["pnl"],
                "trades_filtered": pnl_f["num_trades"],
                "trades_unfiltered": pnl_u["num_trades"],
            }
        )
        print(
            f"  weight LR={lr_w:.1f} XGB={xgb_w:.1f}  "
            f"P&L filtered=${pnl_f['pnl']:+.2f}  "
            f"P&L unfiltered=${pnl_u['pnl']:+.2f}"
        )
    return sweep


def _save_weight_sweep_figure(sweep: Sequence[dict], out_path: Path) -> None:
    xs = [pt["lr_weight"] for pt in sweep]
    ys_f = [pt["pnl_filtered"] for pt in sweep]
    ys_u = [pt["pnl_unfiltered"] for pt in sweep]

    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "ieee", "no-latex"])
    except Exception:
        pass  # fall back to default matplotlib style

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys_f, marker="o", label="Concordance filtered")
    ax.plot(xs, ys_u, marker="s", label="No filter")
    ax.axhline(0.0, linestyle="--", linewidth=0.8, color="0.4")
    ax.set_xlabel("LR weight (XGB weight = 1 - LR weight)")
    ax.set_ylabel("P&L (USD)")
    ax.set_title("Ensemble Weight Sensitivity (LR + XGB)")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> int:
    set_all_seeds(SEED)

    train = build(pd.read_parquet(DATA_DIR / "train.parquet"))
    test = build(pd.read_parquet(DATA_DIR / "test.parquet"))

    feats = feature_cols(train)
    nonzero = [c for c in feats if train[c].std() > 1e-10]
    seq_cols = nonzero + ["group_id"]

    X_train_flat = train[nonzero]
    X_test_flat = test[nonzero]
    X_train_seq = train[seq_cols]
    X_test_seq = test[seq_cols]

    y_train = train[TARGET].to_numpy()
    y_test = test[TARGET].to_numpy()

    print(
        f"Train: {len(train):,} rows, Test: {len(test):,} rows, "
        f"Flat features: {len(nonzero)}, Seq features: {len(seq_cols)}"
    )

    print("\n=== Running 4 ensemble variants ===")
    variants: List[dict] = []
    print("\n[Variant a] LR alone")
    variants.append(_run_variant_a(X_train_flat, X_test_flat, y_train, y_test))
    print("\n[Variant b] LR + XGB equal-weight")
    variants.append(_run_variant_b(X_train_flat, X_test_flat, y_train, y_test))
    print("\n[Variant c] LR + LSTM equal-weight (per-member feature routing)")
    variants.append(
        _run_variant_c(
            X_train_flat, X_test_flat, X_train_seq, X_test_seq, y_train, y_test
        )
    )
    print("\n[Variant d] LR + XGB + LSTM strict (per-member feature routing)")
    variants.append(
        _run_variant_d(
            X_train_flat, X_test_flat, X_train_seq, X_test_seq, y_train, y_test
        )
    )

    # Sanity cross-check (RESEARCH.md Pitfall 2 / P4 guard): LR behavior
    # must not change between variant (a) and variant (c).
    pnl_a = variants[0]["filtered"]["pnl"]
    pnl_c_lr = variants[2].get("member_lr_pnl", float("nan"))
    print(
        f"\nSanity cross-check: LR-solo P&L (variant a) = ${pnl_a:+.2f}  "
        f"LR-member P&L (variant c) = ${pnl_c_lr:+.2f}"
    )
    assert abs(pnl_a - pnl_c_lr) < 2.0, (
        "LR behavior changed between variant (a) and variant (c) — "
        "feature routing bug suspected (group_id leakage?)"
    )

    _print_audit_table(variants)

    print("\n=== Running 11-point weight sweep (LR + XGB) ===")
    sweep = _run_weight_sweep(X_train_flat, X_test_flat, y_train, y_test)

    figure_path = FIGURES_DIR / "ensemble_weight_sweep.png"
    _save_weight_sweep_figure(sweep, figure_path)
    print(f"\nSaved weight-sweep figure to {figure_path}")

    summary = {"variants": variants, "weight_sweep": sweep}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"Saved summary to {summary_path}")

    # Emit a per-variant results file too (downstream comparison compatibility).
    for v in variants:
        save_results(
            model_name=v["name"],
            metrics={
                "filtered": v["filtered"],
                "unfiltered": v["unfiltered"],
                "rejected": v["rejected"],
                "rejection_rate": v["rejection_rate"],
                "flag_rejected_profitable": v["flag_rejected_profitable"],
            },
            output_dir=RESULTS_DIR,
            extra={
                "members": v["members"],
                "weights": v["weights"],
                "concordance_mode": v["concordance_mode"],
            },
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
