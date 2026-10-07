"""Tests for the market_discovery upsert logic.

Network fetches (fetch_active_kalshi_markets, fetch_active_poly_markets)
and semantic matching (match_markets) are integration-heavy and not
tested here. The unit tests cover the pure upsert + dedup behaviour
that keeps active_matches.json stable across discovery runs.
"""
from __future__ import annotations

import pytest

from src.live.market_discovery import (
    _filter_poly_by_volume,
    _is_matchable_kalshi_market,
    _kalshi_mid,
    _poly_volume,
    make_match_key,
    upsert_active_matches,
)


class TestKalshiMid:
    def test_yes_bid_ask_dollars_strings(self):
        m = {"yes_bid_dollars": "0.4500", "yes_ask_dollars": "0.4600"}
        assert _kalshi_mid(m) == pytest.approx(0.455)

    def test_numeric_yes_bid_ask(self):
        m = {"yes_bid": 0.30, "yes_ask": 0.40}
        assert _kalshi_mid(m) == pytest.approx(0.35)

    def test_last_price_fallback(self):
        m = {"last_price_dollars": "0.7500"}
        assert _kalshi_mid(m) == pytest.approx(0.75)

    def test_empty_returns_zero(self):
        assert _kalshi_mid({}) == 0.0

    def test_zero_bid_ask_falls_back(self):
        # Fresh market with no quotes but a last trade
        m = {
            "yes_bid_dollars": "0.0000",
            "yes_ask_dollars": "0.0000",
            "last_price_dollars": "0.65",
        }
        assert _kalshi_mid(m) == pytest.approx(0.65)


class TestPolyVolume:
    """Polymarket markets expose volume through several different fields
    depending on market age/type. _poly_volume should take the max of
    whatever's present and return 0 when nothing is usable."""

    def test_reads_volume_field(self):
        assert _poly_volume({"volume": "12345.67"}) == 12345.67

    def test_reads_volumeNum_field(self):
        assert _poly_volume({"volumeNum": 50000}) == 50000.0

    def test_takes_max_across_fields(self):
        m = {"volume": 100, "volumeNum": 500, "volume24hr": 50}
        assert _poly_volume(m) == 500.0

    def test_missing_all_returns_zero(self):
        assert _poly_volume({}) == 0.0

    def test_malformed_strings_ignored(self):
        assert _poly_volume({"volume": "N/A"}) == 0.0

    def test_none_values_ignored(self):
        assert _poly_volume({"volume": None, "liquidity": 75}) == 75.0


class TestFilterPolyByVolume:
    def test_drops_low_volume(self):
        markets = [
            {"conditionId": "A", "volume": 50},      # below floor
            {"conditionId": "B", "volume": 10000},   # above floor
            {"conditionId": "C", "volume": 0},       # below floor
        ]
        kept = _filter_poly_by_volume(markets, min_volume=5000)
        assert [m["conditionId"] for m in kept] == ["B"]

    def test_keeps_all_when_floor_is_zero(self):
        markets = [
            {"conditionId": "A", "volume": 50},
            {"conditionId": "B", "volume": 100},
        ]
        kept = _filter_poly_by_volume(markets, min_volume=0)
        assert len(kept) == 2


class TestFetchKalshiNullSeriesHandling:
    """Regression: Kalshi's /series endpoint sometimes returns
    ``{"series": null}`` for empty categories (observed live on Climate).
    ``.get("series", [])`` then yields None and the downstream slice
    crashes with TypeError. The fetch must coalesce None -> []."""

    def test_null_series_payload_does_not_crash(self, monkeypatch):
        import src.live.market_discovery as md

        class FakeResp:
            def __init__(self, payload):
                self._payload = payload

            def raise_for_status(self):
                pass

            def json(self):
                return self._payload

        class FakeSession:
            def __init__(self):
                self.headers = {}
                self.call_count = 0

            def get(self, url, params=None, timeout=None):
                self.call_count += 1
                # Return {"series": null} for every category.
                return FakeResp({"series": None})

        class FakeRequests:
            Session = FakeSession

        monkeypatch.setitem(__import__("sys").modules, "requests", FakeRequests)

        # Must return [] without raising TypeError on the None series list.
        result = md.fetch_active_kalshi_markets(
            categories=("Economics", "Crypto"),
            max_series_per_category=10,
        )
        assert result == []


class TestFetchPolyPaginationDepth:
    """Regression: Polymarket's Gamma API returns markets in approximate
    volume-desc order. The commodity markets we need (WTI $X in April,
    Crude Oil (CL) settle buckets) live at offsets 5,000-28,000. With
    the old ``max_pages=10`` (5000 markets) these were completely
    invisible to the matcher even though the matcher already handled
    WTI text correctly — that's why active_matches.json had stale
    KXWTI pairs that upsert kept alive but discovery never refreshed.

    The default must reach at least offset ~15,000 where the high-volume
    WTI markets sit.
    """

    def test_default_max_pages_reaches_wti_offset(self):
        import inspect
        from src.live.market_discovery import fetch_active_poly_markets

        sig = inspect.signature(fetch_active_poly_markets)
        max_pages_default = sig.parameters["max_pages"].default
        page_size_default = sig.parameters["page_size"].default
        reachable = max_pages_default * page_size_default
        # WTI markets observed at offset 15,305+ on 2026-04-11
        assert reachable >= 16000, (
            f"default pagination only reaches offset {reachable}, "
            "but Polymarket WTI markets sit at 15,305+"
        )


class TestFetchKalshi429Retry:
    """Regression: Kalshi's /events endpoint rate-limits aggressively.

    Hitting it ~265 times back-to-back (Financials series list) used to
    trigger HTTP 429 on roughly half the series. The original fetch
    swallowed the exception at DEBUG level and silently dropped every
    market from the throttled series, producing non-deterministic output
    (different commodity series in active_matches.json on each run).

    The fix: retry 429s with exponential backoff, and log at WARNING
    level when retries are exhausted so the gap is visible."""

    def _make_fake_module(self, events_responses):
        """Build a fake requests module.

        ``events_responses`` is a dict {series_ticker: list-of-status-codes}.
        Each /events call for that series pops the next status code from
        the list — once exhausted, returns 200 with real data.
        """
        import src.live.market_discovery as md  # noqa: F401

        class HTTPError(Exception):
            pass

        class FakeResp:
            def __init__(self, status_code, payload):
                self.status_code = status_code
                self._payload = payload

            def raise_for_status(self):
                if self.status_code >= 400:
                    err = HTTPError(f"{self.status_code} Error")
                    err.response = self
                    raise err

            def json(self):
                return self._payload

        series_payload = {
            "series": [
                {"ticker": "KXWTIMIN"},
                {"ticker": "KXWTIMINM"},
            ]
        }

        def events_payload_for(ticker):
            return {
                "events": [
                    {
                        "event_ticker": f"{ticker}-EVT",
                        "markets": [
                            {
                                "ticker": f"{ticker}-26DEC31-T85",
                                "title": f"Will {ticker} hit $85?",
                            }
                        ],
                    }
                ]
            }

        class FakeSession:
            def __init__(self, _responses):
                self.headers = {}
                self._responses = {k: list(v) for k, v in _responses.items()}
                self.call_log = []

            def get(self, url, params=None, timeout=None):
                params = params or {}
                self.call_log.append((url, dict(params)))
                if "/series" in url:
                    return FakeResp(200, series_payload)
                # /events
                st = params.get("series_ticker")
                queue = self._responses.get(st, [])
                if queue:
                    code = queue.pop(0)
                    if code != 200:
                        return FakeResp(code, {})
                return FakeResp(200, events_payload_for(st))

        session_instance = {"s": None}

        def _session_factory():
            s = FakeSession(events_responses)
            session_instance["s"] = s
            return s

        fake_http_error = HTTPError

        class FakeExceptions:
            pass
        FakeExceptions.HTTPError = fake_http_error

        class FakeRequests:
            Session = _session_factory
            exceptions = FakeExceptions

        return FakeRequests, session_instance

    def test_429_on_events_is_retried_and_recovers(self, monkeypatch):
        """First /events call for KXWTIMIN returns 429, retry succeeds."""
        import src.live.market_discovery as md

        fake_requests, session_instance = self._make_fake_module(
            {"KXWTIMIN": [429], "KXWTIMINM": []}
        )
        monkeypatch.setitem(
            __import__("sys").modules, "requests", fake_requests
        )
        monkeypatch.setattr(md.time, "sleep", lambda *a, **k: None)

        markets = md.fetch_active_kalshi_markets(
            categories=("Financials",),
            max_series_per_category=10,
        )

        tickers = sorted(m["ticker"] for m in markets)
        # Both series must produce a market (no silent drop)
        assert tickers == ["KXWTIMIN-26DEC31-T85", "KXWTIMINM-26DEC31-T85"]

        # KXWTIMIN's /events should have been called at least twice
        # (first 429 + retry). The series list call + 2 events calls for
        # KXWTIMIN + 1 for KXWTIMINM + 1 series call = 4+ total.
        s = session_instance["s"]
        events_for_min = [
            c for c in s.call_log
            if "/events" in c[0] and c[1].get("series_ticker") == "KXWTIMIN"
        ]
        assert len(events_for_min) >= 2, (
            f"expected retry for KXWTIMIN, got {len(events_for_min)} call(s)"
        )

    def test_persistent_429_logs_warning_and_continues(
        self, monkeypatch, caplog
    ):
        """If /events returns 429 on every retry, skip the series with a
        WARNING (not silent DEBUG) so the gap is visible in logs."""
        import logging as _logging
        import src.live.market_discovery as md

        # Return 429 many times — more than any sane retry count.
        fake_requests, _ = self._make_fake_module(
            {"KXWTIMIN": [429] * 20, "KXWTIMINM": []}
        )
        monkeypatch.setitem(
            __import__("sys").modules, "requests", fake_requests
        )
        monkeypatch.setattr(md.time, "sleep", lambda *a, **k: None)

        with caplog.at_level(_logging.WARNING, logger="src.live.market_discovery"):
            markets = md.fetch_active_kalshi_markets(
                categories=("Financials",),
                max_series_per_category=10,
            )

        # KXWTIMINM still works
        tickers = {m["ticker"] for m in markets}
        assert "KXWTIMINM-26DEC31-T85" in tickers
        # KXWTIMIN was silently dropped — but we must have logged the failure
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "KXWTIMIN" in messages
        assert "429" in messages or "rate" in messages.lower()


class TestIsMatchableKalshiMarket:
    def test_accepts_simple_title(self):
        m = {
            "ticker": "KXWTI-26APR08-T107.99",
            "title": "Will the WTI front-month settle price be above $107.99?",
        }
        assert _is_matchable_kalshi_market(m) is True

    def test_rejects_kxmve_parlay(self):
        m = {
            "ticker": "KXMVECROSSCATEGORY-S12345-ABCDE",
            "title": "yes Denver,no Over 240.5,yes Xander Schauffele",
        }
        assert _is_matchable_kalshi_market(m) is False

    def test_rejects_empty_title(self):
        m = {"ticker": "KXWTI-26APR08-T107.99", "title": ""}
        assert _is_matchable_kalshi_market(m) is False

    def test_rejects_multi_leg_comma_title(self):
        m = {
            "ticker": "KXFOO-BAR",
            "title": "yes A,no B,yes C",
        }
        assert _is_matchable_kalshi_market(m) is False


class TestMakeMatchKey:
    def test_stable_key_same_inputs(self):
        m = {"kalshi_ticker": "KXWTI-26APR08-T107.99", "poly_id": "123456"}
        assert make_match_key(m) == make_match_key(m)

    def test_different_tickers_produce_different_keys(self):
        a = {"kalshi_ticker": "KXWTI-26APR08-T107.99", "poly_id": "123456"}
        b = {"kalshi_ticker": "KXWTI-26APR09-T107.99", "poly_id": "123456"}
        assert make_match_key(a) != make_match_key(b)

    def test_different_poly_ids_produce_different_keys(self):
        a = {"kalshi_ticker": "KXWTI-26APR08-T107.99", "poly_id": "111"}
        b = {"kalshi_ticker": "KXWTI-26APR08-T107.99", "poly_id": "222"}
        assert make_match_key(a) != make_match_key(b)


class TestUpsertActiveMatches:
    """Upsert invariants:

    1. Existing pair_ids NEVER change (collectors rely on array position).
    2. New matches are appended at the end with fresh pair_ids.
    3. Duplicates (same kalshi_ticker + poly_id) are collapsed to one entry
       and take updated price/volume/similarity fields from the new entry.
    4. Each match tracks a `discovered_at` timestamp. Existing entries
       keep their original discovery time; new entries get the fresh one.
    5. Each match tracks a `last_seen` timestamp that updates on every
       upsert where the match appears in the new batch.
    """

    def test_append_new_matches_to_empty(self):
        existing = []
        new = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.9, "spread": 0.1},
            {"kalshi_ticker": "KXB-2", "poly_id": "B", "similarity": 0.8, "spread": 0.2},
        ]
        merged, stats = upsert_active_matches(existing, new, now_ts=1000)
        assert len(merged) == 2
        assert merged[0]["kalshi_ticker"] == "KXA-1"
        assert merged[0]["discovered_at"] == 1000
        assert merged[0]["last_seen"] == 1000
        assert stats["added"] == 2
        assert stats["updated"] == 0

    def test_existing_matches_preserve_index(self):
        """Stability: existing pairs at index N must stay at index N."""
        existing = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.9, "discovered_at": 500},
            {"kalshi_ticker": "KXB-2", "poly_id": "B", "similarity": 0.8, "discovered_at": 500},
        ]
        # Fresh discovery: both still found + one brand new.
        new = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.91, "spread": 0.11},
            {"kalshi_ticker": "KXB-2", "poly_id": "B", "similarity": 0.82, "spread": 0.22},
            {"kalshi_ticker": "KXC-3", "poly_id": "C", "similarity": 0.95, "spread": 0.3},
        ]
        merged, stats = upsert_active_matches(existing, new, now_ts=2000)
        assert len(merged) == 3
        # Order preserved
        assert merged[0]["kalshi_ticker"] == "KXA-1"
        assert merged[1]["kalshi_ticker"] == "KXB-2"
        assert merged[2]["kalshi_ticker"] == "KXC-3"
        # Old discovered_at preserved
        assert merged[0]["discovered_at"] == 500
        # New one gets current ts
        assert merged[2]["discovered_at"] == 2000
        # last_seen updated on all three
        assert merged[0]["last_seen"] == 2000
        assert merged[2]["last_seen"] == 2000
        assert stats["added"] == 1
        assert stats["updated"] == 2

    def test_updates_prices_on_existing(self):
        """When an existing pair is rediscovered, numeric fields refresh."""
        existing = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.9,
             "kalshi_mid": 0.50, "poly_price": 0.40, "spread": 0.10,
             "discovered_at": 500},
        ]
        new = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.92,
             "kalshi_mid": 0.60, "poly_price": 0.35, "spread": 0.25},
        ]
        merged, _ = upsert_active_matches(existing, new, now_ts=1000)
        assert len(merged) == 1
        assert merged[0]["kalshi_mid"] == 0.60
        assert merged[0]["poly_price"] == 0.35
        assert merged[0]["spread"] == 0.25
        assert merged[0]["similarity"] == 0.92
        # Preserved
        assert merged[0]["discovered_at"] == 500

    def test_missing_from_new_marks_as_not_seen(self):
        """Pairs not in the new batch keep their old last_seen, not the fresh one."""
        existing = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.9,
             "discovered_at": 500, "last_seen": 500},
            {"kalshi_ticker": "KXB-2", "poly_id": "B", "similarity": 0.8,
             "discovered_at": 500, "last_seen": 500},
        ]
        # Only KXA-1 was rediscovered this run
        new = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.9},
        ]
        merged, stats = upsert_active_matches(existing, new, now_ts=2000)
        assert len(merged) == 2
        # KXA-1 refreshed
        assert merged[0]["last_seen"] == 2000
        # KXB-2 stale — still at 500
        assert merged[1]["last_seen"] == 500
        assert stats["stale"] == 1

    def test_dedup_within_new_batch(self):
        """If the new batch has duplicates, only one entry is kept."""
        existing = []
        new = [
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.9},
            {"kalshi_ticker": "KXA-1", "poly_id": "A", "similarity": 0.85},  # dup
            {"kalshi_ticker": "KXB-2", "poly_id": "B", "similarity": 0.8},
        ]
        merged, stats = upsert_active_matches(existing, new, now_ts=1000)
        assert len(merged) == 2
        # Later duplicate wins (more recent data)
        assert merged[0]["similarity"] == 0.85


class TestUpsertEviction:
    """Task #30: stale pair eviction.

    Pairs that haven't been re-discovered for N seconds (default 7 days)
    should be TOMBSTONED in place so that:

      1. The collector naturally skips them (the existing
         missing_kalshi_ticker rule already rejects empty tickers).
      2. The array index is preserved so open positions referencing
         pair_id ``live_0100`` stay pointed at the same underlying pair.
      3. Discovery can revive an evicted pair later if the same
         (kalshi_ticker, poly_id) shows up again.

    This is necessary because 615 pairs from the pre-upsert-era (no
    last_seen field at all, plus legacy numeric poly_ids that don't
    resolve in Gamma) were persisting indefinitely despite being
    effectively dead.
    """

    TTL = 7 * 24 * 3600  # match the default

    def test_evicts_pair_with_zero_last_seen(self):
        """Legacy pairs from the pre-upsert era have last_seen=0 or missing."""
        existing = [
            # Missing last_seen entirely (ancient)
            {"kalshi_ticker": "KXOLD-1", "poly_id": "OLD", "similarity": 0.8},
            # Explicit last_seen=0
            {"kalshi_ticker": "KXOLD-2", "poly_id": "ZERO", "similarity": 0.8, "last_seen": 0},
        ]
        # now_ts is well past any reasonable TTL boundary
        merged, stats = upsert_active_matches(existing, [], now_ts=2_000_000)
        assert len(merged) == 2, "array length must stay stable (tombstone in place)"
        assert merged[0].get("evicted") is True
        assert merged[1].get("evicted") is True
        assert merged[0].get("kalshi_ticker") == ""  # so filter rejects
        assert merged[1].get("kalshi_ticker") == ""
        assert stats["evicted"] == 2

    def test_evicts_pair_older_than_ttl(self):
        """Pair last seen longer than TTL ago is tombstoned."""
        last_seen_old = 1_000_000
        now_ts = last_seen_old + self.TTL + 3600  # 1h past TTL
        existing = [
            {"kalshi_ticker": "KXSTALE-1", "poly_id": "S", "similarity": 0.8,
             "last_seen": last_seen_old, "discovered_at": last_seen_old},
        ]
        merged, stats = upsert_active_matches(existing, [], now_ts=now_ts)
        assert len(merged) == 1
        assert merged[0].get("evicted") is True
        assert stats["evicted"] == 1

    def test_keeps_pair_within_ttl(self):
        """Pair last seen within TTL is preserved even if not in new batch."""
        last_seen = 1_000_000
        now_ts = last_seen + self.TTL - 3600  # 1h before TTL boundary
        existing = [
            {"kalshi_ticker": "KXOK-1", "poly_id": "OK", "similarity": 0.8,
             "last_seen": last_seen, "discovered_at": last_seen},
        ]
        merged, stats = upsert_active_matches(existing, [], now_ts=now_ts)
        assert len(merged) == 1
        assert merged[0].get("evicted") is not True
        assert merged[0]["kalshi_ticker"] == "KXOK-1"
        assert stats["evicted"] == 0

    def test_newly_seen_pair_never_evicted(self):
        """If a pair is in new_matches, its last_seen is refreshed first so
        it can never be evicted in the same run."""
        last_seen_ancient = 0  # would normally be evicted
        existing = [
            {"kalshi_ticker": "KXREV-1", "poly_id": "R", "similarity": 0.8,
             "last_seen": last_seen_ancient},
        ]
        new = [
            {"kalshi_ticker": "KXREV-1", "poly_id": "R", "similarity": 0.85},
        ]
        merged, stats = upsert_active_matches(existing, new, now_ts=2_000_000)
        assert len(merged) == 1
        assert merged[0].get("evicted") is not True
        assert merged[0]["last_seen"] == 2_000_000
        assert merged[0]["similarity"] == 0.85
        assert stats["evicted"] == 0

    def test_eviction_preserves_array_index(self):
        """CRITICAL: open positions reference pair_ids (live_0000, live_0001...)
        which are derived from the array index. Evicting entry 1 must NOT
        shift entry 2 down to index 1."""
        existing = [
            {"kalshi_ticker": "KXFRESH-0", "poly_id": "F0", "similarity": 0.9,
             "last_seen": 2_000_000},
            {"kalshi_ticker": "KXSTALE-1", "poly_id": "S1", "similarity": 0.8,
             "last_seen": 0},  # will be evicted
            {"kalshi_ticker": "KXFRESH-2", "poly_id": "F2", "similarity": 0.9,
             "last_seen": 2_000_000},
        ]
        merged, _ = upsert_active_matches(existing, [], now_ts=2_000_000)
        assert len(merged) == 3
        assert merged[0]["kalshi_ticker"] == "KXFRESH-0"
        assert merged[1].get("evicted") is True
        assert merged[2]["kalshi_ticker"] == "KXFRESH-2"

    def test_evicted_pair_records_original_identity(self):
        """For debugging: the tombstone should remember what it used to be."""
        existing = [
            {"kalshi_ticker": "KXWTI-OLD-1", "poly_id": "1712297",
             "similarity": 0.82, "kalshi_title": "Will WTI settle > $100?",
             "last_seen": 0},
        ]
        merged, _ = upsert_active_matches(existing, [], now_ts=2_000_000)
        assert merged[0].get("evicted") is True
        # Should preserve original identity so we can audit what was evicted
        assert merged[0].get("prev_kalshi_ticker") == "KXWTI-OLD-1"
        assert merged[0].get("prev_poly_id") == "1712297"
        assert merged[0].get("evicted_at") == 2_000_000

    def test_already_evicted_pair_stays_evicted(self):
        """Tombstones are sticky — they don't get re-checked each run."""
        existing = [
            {"evicted": True, "evicted_at": 1_000_000,
             "kalshi_ticker": "", "poly_id": "",
             "prev_kalshi_ticker": "KXOLD-1", "prev_poly_id": "X"},
        ]
        merged, stats = upsert_active_matches(existing, [], now_ts=2_000_000)
        assert len(merged) == 1
        assert merged[0].get("evicted") is True
        assert merged[0].get("evicted_at") == 1_000_000  # not overwritten
        assert stats["evicted"] == 0  # already evicted doesn't count again

    def test_protected_indices_skip_eviction(self):
        """Open positions hold pair_id references derived from array
        indices. If an index is listed in ``protected_indices``, even
        an obviously-stale pair at that index must NOT be tombstoned,
        so the collector can keep pricing the position until it closes."""
        existing = [
            {"kalshi_ticker": "KXSTALE-0", "poly_id": "S0", "similarity": 0.8,
             "last_seen": 0},
            {"kalshi_ticker": "KXSTALE-1", "poly_id": "S1", "similarity": 0.8,
             "last_seen": 0},
            {"kalshi_ticker": "KXSTALE-2", "poly_id": "S2", "similarity": 0.8,
             "last_seen": 0},
        ]
        merged, stats = upsert_active_matches(
            existing,
            [],
            now_ts=2_000_000,
            protected_indices={1},  # index 1 has an open position
        )
        assert len(merged) == 3
        assert merged[0].get("evicted") is True
        # Index 1 must survive despite being stale
        assert merged[1].get("evicted") is not True
        assert merged[1]["kalshi_ticker"] == "KXSTALE-1"
        assert merged[2].get("evicted") is True
        assert stats["evicted"] == 2

    def test_revive_evicted_pair_on_rediscovery(self):
        """If an evicted pair's (kalshi_ticker, poly_id) matches a new
        discovery, the tombstone is reanimated — its array index is
        restored to an active pair. This lets us recover from false
        evictions without a permanent hole."""
        existing = [
            {"evicted": True, "evicted_at": 1_000_000,
             "kalshi_ticker": "", "poly_id": "",
             "prev_kalshi_ticker": "KXREV-1", "prev_poly_id": "R"},
        ]
        new = [
            {"kalshi_ticker": "KXREV-1", "poly_id": "R", "similarity": 0.85},
        ]
        merged, stats = upsert_active_matches(existing, new, now_ts=2_000_000)
        assert len(merged) == 1
        assert merged[0].get("evicted") is not True
        assert merged[0]["kalshi_ticker"] == "KXREV-1"
        assert merged[0]["last_seen"] == 2_000_000
