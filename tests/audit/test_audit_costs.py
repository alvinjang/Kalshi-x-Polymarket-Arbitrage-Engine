"""Proves Tier 3 audit detects when canonical results are computed at zero fee.

Two assertions:
    1. confirm_simulate_profit_fee_handling() returns fee_charged=0 with
       paper_claim_mismatch=True and §5.1 in paper_section. This is the
       load-bearing PAPER_DRAFT.md §5.1 prose-vs-code mismatch the audit
       MUST surface.
    2. kalshi_taker_fee_per_contract() validates the 2026 Kalshi fee formula
       at boundary points (C=0/0.01/0.10/0.50/0.99/1.0).

Plus an alias matching VALIDATION.md naming convention (test_audit_costs_catches_zero_fee).
"""
from experiments.audit.audit_costs import (
    confirm_simulate_profit_fee_handling, kalshi_taker_fee_per_contract,
)


def test_simulate_profit_fee_audit_flags_zero_fee_mismatch():
    sp = confirm_simulate_profit_fee_handling()
    assert sp["fee_charged"] == 0.0
    assert sp["paper_claim_mismatch"] is True
    assert "§5.1" in sp["paper_section"]


def test_kalshi_fee_formula_matches_2026_schedule():
    # At C = 0.50, max fee is 1.75c
    assert abs(kalshi_taker_fee_per_contract(0.50) - 0.0175) < 1e-9
    # At C = 0.10, fee is 0.0063
    assert abs(kalshi_taker_fee_per_contract(0.10) - 0.0063) < 1e-9
    # At extremes, fee is near zero
    assert kalshi_taker_fee_per_contract(0.01) < 0.001
    assert kalshi_taker_fee_per_contract(0.99) < 0.001
    # Boundary
    assert kalshi_taker_fee_per_contract(0.0) == 0.0
    assert kalshi_taker_fee_per_contract(1.0) == 0.0


# Alias matching VALIDATION.md naming convention ("zero-fee fixture MUST be flagged")
def test_audit_costs_catches_zero_fee():
    """Alias delegating to simulate_profit fee audit test."""
    test_simulate_profit_fee_audit_flags_zero_fee_mismatch()
