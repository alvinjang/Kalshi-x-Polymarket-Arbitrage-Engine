"""Tests for the two-leg arb order executor."""
from __future__ import annotations

import pytest

from src.live.order_executor import ArbExecutionResult, _infer_legs, execute_arb


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING", raising=False)
    monkeypatch.delenv("EMERGENCY_HALT", raising=False)


class FakeKalshi:
    def __init__(self, fail=False, fail_cancel=False):
        self.fail = fail
        self.fail_cancel = fail_cancel
        self.calls = []
        self.cancels = []

    def place_limit_order(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("simulated kalshi failure")
        return {"order": {"order_id": "k-1"}}

    def cancel_order(self, order_id):
        self.cancels.append(order_id)
        if self.fail_cancel:
            raise RuntimeError("simulated cancel failure")
        return {"cancelled": order_id}


class FakePoly:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def place_limit_order(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("simulated poly failure")
        return {"orderID": "p-1"}


def _args(tmp_path=None, spread=0.30, kalshi_yes=0.70, poly_yes=0.40):
    d = dict(
        pair_id="pair-test",
        kalshi_ticker="KXWTI-X",
        poly_yes_token_id="0xYES",
        poly_no_token_id="0xNO",
        kalshi_yes_price=kalshi_yes,
        poly_yes_price=poly_yes,
        spread=spread,
    )
    if tmp_path is not None:
        # Point risk gate at empty fixtures so we're isolated from the
        # real working-directory paper-trading state.
        d["history_path"] = str(tmp_path / "hist.jsonl")
        d["positions_db"] = str(tmp_path / "p.db")
    return d


def test_paper_mode_returns_noop_when_not_armed(tmp_path):
    res = execute_arb(**_args(tmp_path), kalshi_client=None, poly_client=None)
    assert res.permitted is True
    assert res.placed_kalshi is False
    assert res.placed_polymarket is False
    assert any("paper-mode" in n for n in res.notes)


def test_infer_legs_positive_spread(tmp_path):
    """spread>0 means Kalshi YES expensive → short Kalshi (buy NO), long Poly YES."""
    k_side, k_price, p_side, p_price = _infer_legs(
        spread=0.30, kalshi_yes_price=0.70, poly_yes_price=0.40
    )
    assert k_side == "no"
    assert k_price == pytest.approx(0.30)  # 1 - 0.70
    assert p_side == "yes_token"
    assert p_price == pytest.approx(0.40)


def test_infer_legs_negative_spread(tmp_path):
    """spread<0 means Poly YES expensive → long Kalshi YES, short Poly (buy NO)."""
    k_side, k_price, p_side, p_price = _infer_legs(
        spread=-0.20, kalshi_yes_price=0.30, poly_yes_price=0.50
    )
    assert k_side == "yes"
    assert k_price == pytest.approx(0.30)
    assert p_side == "no_token"
    assert p_price == pytest.approx(0.50)  # 1 - 0.50


def test_both_legs_placed_on_success(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_TRADING", "true")
    k, p = FakeKalshi(), FakePoly()
    res = execute_arb(**_args(tmp_path), kalshi_client=k, poly_client=p)
    assert res.permitted
    assert res.placed_kalshi and res.placed_polymarket
    assert res.kalshi_order_id == "k-1"
    assert res.polymarket_order_id == "p-1"
    assert len(k.calls) == 1 and len(p.calls) == 1


def test_kalshi_failure_short_circuits(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_TRADING", "true")
    k = FakeKalshi(fail=True)
    p = FakePoly()
    res = execute_arb(**_args(tmp_path), kalshi_client=k, poly_client=p)
    assert not res.placed_kalshi
    assert not res.placed_polymarket
    assert "kalshi placement failed" in res.reject_reason
    assert len(p.calls) == 0  # poly leg never attempted


def test_poly_failure_cancels_kalshi_leg(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_TRADING", "true")
    k = FakeKalshi()
    p = FakePoly(fail=True)
    res = execute_arb(**_args(tmp_path), kalshi_client=k, poly_client=p)
    assert res.placed_kalshi
    assert not res.placed_polymarket
    assert k.cancels == ["k-1"]
    assert "polymarket placement failed" in res.reject_reason
    assert any("cancelled kalshi leg" in n for n in res.notes)


def test_poly_failure_with_cancel_failure_flags_critical(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_TRADING", "true")
    k = FakeKalshi(fail_cancel=True)
    p = FakePoly(fail=True)
    res = execute_arb(**_args(tmp_path), kalshi_client=k, poly_client=p)
    assert res.placed_kalshi  # the kalshi leg is open and unhedged
    assert any("CRITICAL" in n for n in res.notes)


def test_kill_switch_blocks_before_orders(monkeypatch, tmp_path):
    """When daily loss limit is breached, no orders are placed."""
    import json
    from datetime import datetime, timezone

    monkeypatch.setenv("LIVE_TRADING", "true")
    # Write losses to the SAME path that _args() points the risk gate at,
    # so the gate sees them.
    hist_path = tmp_path / "hist.jsonl"
    today_iso = datetime.now(timezone.utc).isoformat()
    with open(hist_path, "w") as f:
        for _ in range(20):
            f.write(json.dumps({"realized_pnl": -1.0, "exit_time": today_iso}) + "\n")

    k, p = FakeKalshi(), FakePoly()
    res = execute_arb(**_args(tmp_path), kalshi_client=k, poly_client=p)
    assert not res.permitted
    assert "kill-switch" in res.reject_reason
    assert len(k.calls) == 0 and len(p.calls) == 0
