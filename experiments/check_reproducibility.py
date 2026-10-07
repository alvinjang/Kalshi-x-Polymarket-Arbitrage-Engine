"""Run verify_headline.py twice and assert results are reproducible within 1%.

Satisfies ENV-04: reproducibility verification for all Table 2 models.
Also satisfies ENV-05: reconciles verify_headline output against tier1/*.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_verify_headline() -> dict:
    """Run verify_headline.py and return its JSON output."""
    result = subprocess.run(
        [sys.executable, "-m", "experiments.verify_headline"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(Path("experiments/results/verify_headline.json").read_text())


def check_reproducibility() -> bool:
    """Run verify_headline twice and compare all metrics within 1%."""
    print("=== Run 1 ===")
    run1 = run_verify_headline()
    print("=== Run 2 ===")
    run2 = run_verify_headline()

    all_pass = True
    for model in run1["results"]:
        if model not in run2["results"]:
            print(f"  FAIL: {model} missing from run 2")
            all_pass = False
            continue
        for metric in ["rmse", "pnl", "win_rate", "sharpe_per_trade", "directional_accuracy"]:
            v1 = run1["results"][model].get(metric, 0)
            v2 = run2["results"][model].get(metric, 0)
            if v1 == 0 and v2 == 0:
                print(f"  {model}.{metric}: SKIP (both zero)")
                continue
            denom = max(abs(v1), abs(v2), 1e-10)
            pct_diff = abs(v1 - v2) / denom * 100
            status = "PASS" if pct_diff <= 1.0 else "FAIL"
            print(f"  {model}.{metric}: {status} ({pct_diff:.4f}% diff) [{v1} vs {v2}]")
            if pct_diff > 1.0:
                all_pass = False

    return all_pass


def check_tier1_reconciliation() -> bool:
    """Verify tier1/*.json files match verify_headline.json within 1%."""
    vh_path = Path("experiments/results/verify_headline.json")
    if not vh_path.exists():
        print("ERROR: verify_headline.json not found")
        return False

    vh = json.loads(vh_path.read_text())

    # Map tier1 filenames to verify_headline model names
    tier1_map = {
        "xgboost.json": "xgboost",
        "linear_regression.json": "linear_regression",
        "naive_spread_closes.json": "naive",
        "volume_higher_volume_correct.json": "volume",
    }

    tier1_dir = Path("experiments/results/tier1")
    all_pass = True

    for filename, vh_name in tier1_map.items():
        tier1_path = tier1_dir / filename
        if not tier1_path.exists():
            print(f"  SKIP: {filename} not found")
            continue

        tier1_data = json.loads(tier1_path.read_text())
        vh_model = vh["results"].get(vh_name, {})

        if not vh_model:
            print(f"  SKIP: {vh_name} not in verify_headline results")
            continue

        # Check n_features match (must be 51 after regeneration)
        t1_features = tier1_data.get("n_features", tier1_data.get("features", "?"))
        vh_features = vh.get("features", "?")
        print(f"  {filename}: n_features={t1_features} (verify_headline={vh_features})")

        if t1_features != vh_features:
            print(f"    FAIL: feature count mismatch ({t1_features} vs {vh_features})")
            all_pass = False

        # Compare RMSE only (within 5% tolerance).
        # P&L and win_rate are not compared because run_baselines.py uses a
        # different profit simulation (no explicit fee deduction in actuals*sign)
        # vs verify_headline.py's inline simulate_pnl (subtracts 2pp fee).
        # This discrepancy is documented and acceptable per the plan.
        t1_metrics = tier1_data.get("metrics", tier1_data.get("results", {}).get(vh_name, {}))
        t1_rmse = t1_metrics.get("rmse", None)
        vh_rmse = vh_model.get("rmse", None)
        if t1_rmse is not None and vh_rmse is not None:
            denom = max(abs(t1_rmse), abs(vh_rmse), 1e-10)
            pct = abs(t1_rmse - vh_rmse) / denom * 100
            tolerance = 5.0  # 5% tolerance (plan spec)
            status = "PASS" if pct <= tolerance else "FAIL"
            print(f"    rmse: {status} ({pct:.2f}%) [{t1_rmse:.5f} vs {vh_rmse:.5f}]")
            if pct > tolerance:
                all_pass = False

    return all_pass


def main() -> int:
    print("=" * 60)
    print("ENV-04: Reproducibility Check (two runs, <1% tolerance)")
    print("=" * 60)
    repro_ok = check_reproducibility()

    print()
    print("=" * 60)
    print("ENV-05: Tier1 JSON Reconciliation")
    print("=" * 60)
    recon_ok = check_tier1_reconciliation()

    print()
    if repro_ok and recon_ok:
        print("ALL CHECKS PASSED")
        return 0
    else:
        if not repro_ok:
            print("FAIL: Reproducibility check failed (>1% diff between runs)")
        if not recon_ok:
            print("FAIL: Tier1 reconciliation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
