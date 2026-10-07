"""Daily-rotation tests for paper_trades.jsonl.

The unrotated file grew to 81MB after ~4 days of collection, approaching
GitHub's 100MB hard limit. We rotate writes into per-UTC-day files:

  data/live/paper_trades.jsonl               # frozen archive, never written
  data/live/paper_trades_2026-04-11.jsonl    # today (being written to)
  data/live/paper_trades_2026-04-12.jsonl    # tomorrow

Dashboard and any other reader glob ``paper_trades*.jsonl`` so both the
archive and all daily files are included.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.live.paper_trader import PaperTrader


@pytest.fixture
def live_dir(tmp_path):
    d = tmp_path / "live"
    d.mkdir()
    return d


@pytest.fixture
def trader(tmp_path, live_dir):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return PaperTrader(data_dir=data_dir, live_dir=live_dir)


class TestDailyRotation:
    def test_write_path_includes_utc_date(self, trader, live_dir):
        """append_trades must write to paper_trades_YYYY-MM-DD.jsonl,
        where the date is today's UTC date."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        expected = live_dir / f"paper_trades_{today}.jsonl"

        trader.append_trades(
            [{"timestamp": 1, "pair_id": "live_0001", "model": "LR", "prediction": 0.1}]
        )

        assert expected.exists(), f"expected {expected.name} to be created"
        # Old static filename must NOT be created
        assert not (live_dir / "paper_trades.jsonl").exists()

    def test_does_not_touch_existing_static_archive(self, trader, live_dir):
        """If a legacy paper_trades.jsonl file exists (the 81MB archive),
        append_trades must not append to it. It writes only to the daily
        file."""
        archive = live_dir / "paper_trades.jsonl"
        archive.write_text('{"archived": true}\n')
        archive_size_before = archive.stat().st_size

        trader.append_trades([{"pair_id": "live_0002"}])

        assert archive.stat().st_size == archive_size_before, (
            "legacy archive must not grow"
        )

    def test_multiple_appends_go_to_same_daily_file(self, trader, live_dir):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily = live_dir / f"paper_trades_{today}.jsonl"

        trader.append_trades([{"pair_id": "live_0001"}])
        trader.append_trades([{"pair_id": "live_0002"}, {"pair_id": "live_0003"}])

        with open(daily) as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 3


class TestDashboardGlobs:
    """The dashboard must load trades from BOTH the legacy archive and
    all daily-rotated files."""

    def test_load_all_trade_logs_reads_archive_plus_daily(self, live_dir):
        from src.live.dashboard import load_all_trade_logs

        # Legacy archive
        (live_dir / "paper_trades.jsonl").write_text(
            json.dumps({"source": "archive", "pair_id": "live_A"}) + "\n"
        )
        # Daily file
        (live_dir / "paper_trades_2026-04-11.jsonl").write_text(
            json.dumps({"source": "daily", "pair_id": "live_B"}) + "\n"
        )
        # Another daily file
        (live_dir / "paper_trades_2026-04-12.jsonl").write_text(
            json.dumps({"source": "daily", "pair_id": "live_C"}) + "\n"
        )

        trades = load_all_trade_logs(live_dir)
        assert len(trades) == 3
        sources = {t["source"] for t in trades}
        assert sources == {"archive", "daily"}

    def test_load_all_trade_logs_handles_missing_dir(self, tmp_path):
        from src.live.dashboard import load_all_trade_logs

        missing = tmp_path / "nonexistent"
        # Must not raise
        trades = load_all_trade_logs(missing)
        assert trades == []

    def test_load_all_trade_logs_returns_empty_when_no_files(self, live_dir):
        from src.live.dashboard import load_all_trade_logs

        # live_dir exists but has no paper_trades*.jsonl files
        (live_dir / "bars.parquet").write_bytes(b"\x00\x01")
        trades = load_all_trade_logs(live_dir)
        assert trades == []
