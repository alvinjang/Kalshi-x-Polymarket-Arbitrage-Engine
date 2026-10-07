"""Tests for the live-trading risk gate."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.live.risk_manager import (
    DEFAULT_DAILY_LOSS_LIMIT,
    DEFAULT_EXPOSURE_CAP_PER_EXCHANGE,
    DEFAULT_PER_TRADE_USD,
    check_pretrade,
    is_live_trading_armed,
    realized_loss_today,
)


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    """Make sure LIVE_TRADING/EMERGENCY_HALT do not leak across tests."""
    monkeypatch.delenv("LIVE_TRADING", raising=False)
    monkeypatch.delenv("EMERGENCY_HALT", raising=False)


def test_defaults_off(monkeypatch):
    assert is_live_trading_armed() is False


def test_armed_when_env_true(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    assert is_live_trading_armed() is True


def test_emergency_halt_overrides(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.setenv("EMERGENCY_HALT", "true")
    assert is_live_trading_armed() is False


def test_case_insensitive(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "TRUE")
    assert is_live_trading_armed() is True


def test_pretrade_blocked_when_not_armed(tmp_path):
    decision = check_pretrade(
        trade_size_usd=1.0, exchange="kalshi",
        history_path=tmp_path / "hist.jsonl",
        positions_db=tmp_path / "p.db",
    )
    assert not decision.permit
    assert "LIVE_TRADING" in decision.reason


def test_pretrade_blocks_oversize(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    decision = check_pretrade(
        trade_size_usd=DEFAULT_PER_TRADE_USD + 0.01,
        exchange="kalshi",
        history_path=tmp_path / "hist.jsonl",
        positions_db=tmp_path / "p.db",
    )
    assert not decision.permit
    assert "per_trade_cap" in decision.reason


def test_pretrade_permits_within_size(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    decision = check_pretrade(
        trade_size_usd=DEFAULT_PER_TRADE_USD,
        exchange="kalshi",
        history_path=tmp_path / "hist.jsonl",
        positions_db=tmp_path / "p.db",
    )
    assert decision.permit


def test_daily_loss_aggregation(tmp_path):
    hist = tmp_path / "hist.jsonl"
    now = datetime.now(timezone.utc)
    today_iso = now.isoformat()
    yesterday_iso = (now - timedelta(days=1)).isoformat()

    records = [
        # Today's records
        {"realized_pnl": -3.0, "exit_time": today_iso},
        {"realized_pnl": -2.5, "exit_time": today_iso},
        {"realized_pnl": +1.0, "exit_time": today_iso},  # wins don't subtract
        # Yesterday's records (must be excluded)
        {"realized_pnl": -100.0, "exit_time": yesterday_iso},
    ]
    with open(hist, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    loss = realized_loss_today(hist)
    assert loss == pytest.approx(5.5)


def test_kill_switch_blocks_when_daily_loss_exceeded(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    hist = tmp_path / "hist.jsonl"
    today_iso = datetime.now(timezone.utc).isoformat()
    with open(hist, "w") as f:
        for _ in range(20):
            f.write(json.dumps({"realized_pnl": -1.0, "exit_time": today_iso}) + "\n")

    decision = check_pretrade(
        trade_size_usd=DEFAULT_PER_TRADE_USD,
        exchange="kalshi",
        history_path=hist,
        positions_db=tmp_path / "p.db",
        daily_loss_limit=DEFAULT_DAILY_LOSS_LIMIT,
    )
    assert not decision.permit
    assert "daily loss kill-switch" in decision.reason


def test_exposure_cap_blocks_over_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")

    db = tmp_path / "p.db"
    with sqlite3.connect(db) as conn:
        conn.execute("""
            CREATE TABLE positions (
                pair_id TEXT, entry_kalshi_price REAL, entry_poly_price REAL
            )
        """)
        # 26 simultaneous positions @ $2/leg = $52 exposure > $50 cap
        for i in range(26):
            conn.execute(
                "INSERT INTO positions VALUES (?, ?, ?)",
                (f"p{i}", 1.0, 1.0),
            )

    decision = check_pretrade(
        trade_size_usd=DEFAULT_PER_TRADE_USD,
        exchange="kalshi",
        history_path=tmp_path / "hist.jsonl",
        positions_db=db,
        exposure_cap=DEFAULT_EXPOSURE_CAP_PER_EXCHANGE,
    )
    assert not decision.permit
    assert "exposure cap" in decision.reason
