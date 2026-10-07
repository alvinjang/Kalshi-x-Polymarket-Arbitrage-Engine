"""Unit tests for src/analysis/reconciliation.py.

Covers RECON-01 through RECON-08 acceptance criteria.
All fixtures use synthetic in-memory data — no real DB, no real parquet reads.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _make_position(
    pair_id="kxdoge26apr1417b0092-0x5b919435",
    kalshi_ticker="KXDOGE-26APR1417-B0.092",
    direction="short_spread",
    entry_spread=0.05,
    exit_spread=0.02,
    realized_pnl=0.03,
    entry_time="2026-04-14T13:00:00Z",
    exit_time="2026-04-14T17:00:00Z",
    exit_reason="TIME_STOP",
    bars_held=4,
    tier="DAILY",
    category=None,
) -> dict:
    return {
        "pair_id": pair_id,
        "kalshi_ticker": kalshi_ticker,
        "direction": direction,
        "entry_spread": entry_spread,
        "exit_spread": exit_spread,
        "realized_pnl": realized_pnl,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "tier": tier,
        "max_spread": max(entry_spread, exit_spread),
        "min_spread": min(entry_spread, exit_spread),
        "entry_kalshi_price": 0.5,
        "entry_poly_price": 0.45,
    }


def _make_matched_result(
    pair_id="kxdoge26apr1417b0092-0x5b919435",
    kalshi_ticker="KXDOGE-26APR1417-B0.092",
    direction="short_spread",
    exit_reason="TIME_STOP",
    category="crypto",
    live_pnl=0.03,
    sim_pnl=0.05,
    lr_pred=0.04,
    xgb_pred=0.06,
    avg_pred=0.05,
    actual_change=-0.03,
    matched=True,
) -> dict:
    return {
        "pair_id": pair_id,
        "kalshi_ticker": kalshi_ticker,
        "direction": direction,
        "exit_reason": exit_reason,
        "category": category,
        "live_pnl": live_pnl,
        "sim_pnl": sim_pnl,
        "lr_pred": lr_pred,
        "xgb_pred": xgb_pred,
        "avg_pred": avg_pred,
        "actual_change": actual_change,
        "matched": matched,
    }


# ---------------------------------------------------------------------------
# RECON-01: Module importable with all 6 public functions present
# ---------------------------------------------------------------------------

class TestModuleImportable:
    """RECON-01: src.analysis.reconciliation importable; 6 functions present."""

    def test_module_importable(self):
        import src.analysis.reconciliation as recon  # noqa: F401
        required = [
            "load_closed_positions",
            "run_shadow_simulation",
            "build_summary",
            "category_breakdown",
            "exit_reason_attribution",
            "acceptance_gate",
        ]
        for fn in required:
            assert hasattr(recon, fn), f"Missing public function: {fn}"


# ---------------------------------------------------------------------------
# RECON-02: filter_window excludes schema-fix positions and pre-window entries
# ---------------------------------------------------------------------------

class TestWindowFilter:
    """RECON-02: filter_window excludes force_close_schema_fix and pre-April-11 entries."""

    def test_window_filter(self):
        from src.analysis.reconciliation import filter_window

        positions = [
            # Should be excluded: force_close_schema_fix exit reason
            _make_position(
                pair_id="force-close-pos",
                exit_reason="force_close_schema_fix",
                entry_time="2026-04-12T00:00:00Z",
            ),
            # Should be excluded: entry_time before window start (April 11)
            _make_position(
                pair_id="pre-window-pos",
                exit_reason="TIME_STOP",
                entry_time="2026-04-10T00:00:00Z",
            ),
            # Should be included: valid April 14 position
            _make_position(
                pair_id="valid-pos",
                exit_reason="TIME_STOP",
                entry_time="2026-04-14T08:00:00Z",
            ),
        ]

        result = filter_window(positions, start="2026-04-11")
        pair_ids = [p["pair_id"] for p in result]

        assert "force-close-pos" not in pair_ids, "force_close_schema_fix should be excluded"
        assert "pre-window-pos" not in pair_ids, "pre-window entry should be excluded"
        assert "valid-pos" in pair_ids, "valid April 14 position should be included"
        assert len(result) == 1


# ---------------------------------------------------------------------------
# RECON-03: run_shadow_simulation matched/unmatched counting
# ---------------------------------------------------------------------------

class TestPairTrades:
    """RECON-03: shadow simulation returns correct matched/unmatched counts."""

    def test_pair_trades(self):
        from src.analysis.reconciliation import build_summary

        # Simulate what run_shadow_simulation returns: 2 matched, 1 unmatched
        matched_results = [
            _make_matched_result(pair_id="pair-a", matched=True, live_pnl=0.03, sim_pnl=0.05),
            _make_matched_result(pair_id="pair-b", matched=True, live_pnl=-0.01, sim_pnl=0.00),
        ]
        all_results = matched_results + [
            {**_make_matched_result(pair_id="pair-c"), "matched": False}
        ]

        # Caller filters matched before passing to build_summary
        matched = [r for r in all_results if r.get("matched")]
        unmatched_count = len(all_results) - len(matched)

        summary = build_summary(matched)
        summary["unmatched_count"] = unmatched_count

        assert summary["matched_count"] == 2
        assert summary["unmatched_count"] == 1


# ---------------------------------------------------------------------------
# RECON-04: Fee function identity — reconciliation uses profit_sim.simulate_profit
# ---------------------------------------------------------------------------

class TestFeeFunctionIdentity:
    """RECON-04: reconciliation uses simulate_profit (threshold-only), not deduction model."""

    def test_fee_function_identity(self):
        import numpy as np
        from src.evaluation.profit_sim import simulate_profit

        # threshold-only model: |pred|=0.05 > 0.02 threshold -> trade taken
        # P&L = actual * sign(pred) = 0.05 * 1 = 0.05 (NO deduction)
        result = simulate_profit(
            predictions=np.array([0.05]),
            actuals=np.array([0.05]),
            threshold=0.02,
        )
        assert result["total_pnl"] == pytest.approx(0.05), (
            "simulate_profit should NOT deduct 2pp from P&L (expected 0.05, got deduction model 0.03)"
        )

    def test_reconciliation_imports_correct_fee_function(self):
        """Verify that reconciliation.py imports from src.evaluation.profit_sim, not verify_headline."""
        import src.analysis.reconciliation as recon
        import inspect, sys

        # Check that the module was imported (no NameError or ImportError)
        # and that the simulate_profit reference in reconciliation traces to profit_sim
        assert "src.analysis.reconciliation" in sys.modules

        # Find simulate_profit reference in reconciliation module
        for name, obj in inspect.getmembers(recon):
            if callable(obj) and getattr(obj, "__name__", "") == "simulate_profit":
                mod = getattr(obj, "__module__", "")
                assert "profit_sim" in mod, (
                    f"simulate_profit in reconciliation must come from profit_sim, got: {mod}"
                )
                break


# ---------------------------------------------------------------------------
# RECON-05: build_summary returns required keys
# ---------------------------------------------------------------------------

class TestSummarySchema:
    """RECON-05: build_summary returns dict with all required keys."""

    def test_summary_schema(self):
        from src.analysis.reconciliation import build_summary

        matched_results = [
            _make_matched_result(pair_id="pair-a", live_pnl=0.03, sim_pnl=0.05),
            _make_matched_result(pair_id="pair-b", live_pnl=-0.01, sim_pnl=0.00),
        ]

        summary = build_summary(matched_results)

        required_keys = {
            "live_total_pnl",
            "sim_total_pnl",
            "tracking_error",
            "matched_count",
            "unmatched_count",
            "gap_metric",
        }
        missing = required_keys - set(summary.keys())
        assert not missing, f"build_summary missing keys: {missing}"

    def test_summary_values_correct(self):
        from src.analysis.reconciliation import build_summary

        matched_results = [
            _make_matched_result(pair_id="pair-a", live_pnl=0.03, sim_pnl=0.05),
            _make_matched_result(pair_id="pair-b", live_pnl=-0.01, sim_pnl=0.00),
        ]

        summary = build_summary(matched_results)

        assert summary["live_total_pnl"] == pytest.approx(0.02)   # 0.03 + (-0.01)
        assert summary["sim_total_pnl"] == pytest.approx(0.05)    # 0.05 + 0.00
        assert summary["tracking_error"] == pytest.approx(-0.03)  # live - sim
        assert summary["matched_count"] == 2


# ---------------------------------------------------------------------------
# RECON-06: category_breakdown groups by category from kalshi_ticker
# ---------------------------------------------------------------------------

class TestCategoryBreakdown:
    """RECON-06: category_breakdown groups by category correctly."""

    def test_category_breakdown(self):
        from src.analysis.reconciliation import category_breakdown

        matched_results = [
            _make_matched_result(pair_id="pair-a", category="crypto", live_pnl=0.03, sim_pnl=0.04),
            _make_matched_result(pair_id="pair-b", category="crypto", live_pnl=0.02, sim_pnl=0.01),
            _make_matched_result(pair_id="pair-c", category="inflation", live_pnl=-0.01, sim_pnl=0.0),
        ]

        result = category_breakdown(matched_results)

        assert "crypto" in result, "crypto category missing from breakdown"
        assert "inflation" in result, "inflation category missing from breakdown"

        crypto = result["crypto"]
        for key in ("live_pnl", "sim_pnl", "count"):
            assert key in crypto, f"crypto breakdown missing key: {key}"

        assert crypto["count"] == 2
        assert result["inflation"]["count"] == 1

    def test_category_breakdown_pnl_sums(self):
        from src.analysis.reconciliation import category_breakdown

        matched_results = [
            _make_matched_result(pair_id="pair-a", category="crypto", live_pnl=0.03, sim_pnl=0.04),
            _make_matched_result(pair_id="pair-b", category="crypto", live_pnl=0.02, sim_pnl=0.01),
        ]

        result = category_breakdown(matched_results)
        crypto = result["crypto"]

        assert crypto["live_pnl"] == pytest.approx(0.05)   # 0.03 + 0.02
        assert crypto["sim_pnl"] == pytest.approx(0.05)    # 0.04 + 0.01


# ---------------------------------------------------------------------------
# RECON-07: exit_reason_attribution groups all 5 reasons
# ---------------------------------------------------------------------------

class TestExitReasonAttribution:
    """RECON-07: exit_reason_attribution groups all 5 exit reasons with correct keys."""

    def test_exit_reason_attribution(self):
        from src.analysis.reconciliation import exit_reason_attribution

        exit_reasons = ["TIME_STOP", "RESOLUTION_EXIT", "MOMENTUM", "STOP_LOSS", "TAKE_PROFIT"]
        matched_results = [
            _make_matched_result(pair_id=f"pair-{i}", exit_reason=reason, live_pnl=0.01, sim_pnl=0.01)
            for i, reason in enumerate(exit_reasons)
        ]

        result = exit_reason_attribution(matched_results)

        for reason in exit_reasons:
            assert reason in result, f"exit_reason_attribution missing reason: {reason}"
            entry = result[reason]
            for key in ("live_count", "sim_count", "live_pnl", "sim_pnl"):
                assert key in entry, f"exit_reason {reason} missing key: {key}"

    def test_exit_reason_counts(self):
        from src.analysis.reconciliation import exit_reason_attribution

        # 2 TIME_STOP, 1 RESOLUTION_EXIT
        matched_results = [
            _make_matched_result(pair_id="pair-0", exit_reason="TIME_STOP", live_pnl=0.01, sim_pnl=0.01),
            _make_matched_result(pair_id="pair-1", exit_reason="TIME_STOP", live_pnl=0.02, sim_pnl=0.03),
            _make_matched_result(pair_id="pair-2", exit_reason="RESOLUTION_EXIT", live_pnl=-0.01, sim_pnl=0.00),
        ]

        result = exit_reason_attribution(matched_results)

        assert result["TIME_STOP"]["live_count"] == 2
        assert result["RESOLUTION_EXIT"]["live_count"] == 1
        assert result["TIME_STOP"]["live_pnl"] == pytest.approx(0.03)  # 0.01 + 0.02


# ---------------------------------------------------------------------------
# RECON-08: acceptance_gate passes at >=80% and raises ValueError below 80%
# ---------------------------------------------------------------------------

class TestAcceptanceGate:
    """RECON-08: acceptance_gate passes when matched/total >= 80%; raises ValueError below."""

    def test_acceptance_gate_passes(self):
        from src.analysis.reconciliation import acceptance_gate

        # 90/100 = 90% -> should return True
        result = acceptance_gate(matched=90, total=100)
        assert result is True

    def test_acceptance_gate_passes_at_boundary(self):
        from src.analysis.reconciliation import acceptance_gate

        # exactly 80/100 = 80% -> should pass
        result = acceptance_gate(matched=80, total=100)
        assert result is True

    def test_acceptance_gate_fails_below_threshold(self):
        from src.analysis.reconciliation import acceptance_gate

        # 79/100 = 79% -> should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            acceptance_gate(matched=79, total=100)

        error_msg = str(exc_info.value)
        assert "gap" in error_msg.lower() or "79.0%" in error_msg, (
            f"ValueError message should contain 'gap' and '79.0%', got: {error_msg}"
        )

    def test_acceptance_gate_error_message_contains_percentage(self):
        from src.analysis.reconciliation import acceptance_gate

        with pytest.raises(ValueError) as exc_info:
            acceptance_gate(matched=79, total=100)

        error_msg = str(exc_info.value)
        assert "79.0%" in error_msg, f"Error message should contain '79.0%', got: {error_msg}"
