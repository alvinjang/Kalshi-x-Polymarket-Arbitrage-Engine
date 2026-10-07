#!/usr/bin/env python3
"""Pre-flight check for Oracle VM deployment.

Verifies all deployment artifacts exist before deploying to the VM.
Run this locally before running setup_oracle.sh.

Usage:
    python scripts/preflight_check.py
"""
from pathlib import Path
import sys


REQUIRED_FILES = [
    # Pre-trained models
    "models/deployed/linear_regression.pkl",
    "models/deployed/xgboost.pkl",
    "models/deployed/feature_columns.json",
    # Live data
    "data/live/active_matches.json",
    "data/live/pair_classifications.json",
    # Deployment scripts
    "scripts/setup_oracle.sh",
    "scripts/oracle_keepalive.sh",
    "scripts/oracle_autocommit.sh",
    # Live trading code
    "src/live/trading_cycle.py",
    "src/live/strategy.py",
    "src/live/collector.py",
    "src/live/contract_classifier.py",
    "src/live/position_manager.py",
    # Supporting code
    "src/models/base.py",
    "src/features/engineering.py",
    "src/features/schemas.py",
]


def main() -> int:
    """Check all required files and report status."""
    print("DS340 Deployment Pre-flight Check")
    print("=" * 50)

    missing = []
    found = 0

    for filepath in REQUIRED_FILES:
        path = Path(filepath)
        if path.exists():
            print(f"  OK      {filepath}")
            found += 1
        else:
            print(f"  MISSING {filepath}")
            missing.append(filepath)

    print("=" * 50)
    print(f"Result: {found}/{len(REQUIRED_FILES)} files present")

    if missing:
        print(f"\n{len(missing)} missing file(s):")
        for f in missing:
            print(f"  - {f}")
        print("\nDeploy blocked until all files are present.")
        return 1
    else:
        print("\nAll files present. Ready to deploy:")
        print("  ssh opc@129.80.4.112 'bash -s' < scripts/setup_oracle.sh")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
