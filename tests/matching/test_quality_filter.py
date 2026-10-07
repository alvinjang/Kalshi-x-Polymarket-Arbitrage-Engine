"""Tests for quality_filter module."""
import pytest
from src.matching.quality_filter import (
    parse_date,
    extract_direction,
    extract_threshold,
    directions_compatible,
    thresholds_compatible,
    filter_candidates,
    filter_active_match,
    filter_active_matches,
)


class TestParseDate:
    def test_iso_with_timezone(self):
        d = parse_date("2026-03-06T13:29:00Z")
        assert d is not None
        assert d.year == 2026 and d.month == 3 and d.day == 6

    def test_date_only(self):
        d = parse_date("2026-01-15")
        assert d is not None
        assert d.year == 2026

    def test_empty(self):
        assert parse_date("") is None
        assert parse_date(None) is None


class TestExtractDirection:
    def test_above(self):
        assert extract_direction("Will unemployment be above 4.3%?") == "above"

    def test_below(self):
        assert extract_direction("Will BTC drop below $60,000?") == "below"

    def test_exactly(self):
        assert extract_direction("Will unemployment be exactly 4.1%?") == "exactly"

    def test_between(self):
        assert extract_direction("Will BTC be between $80k and $85k?") == "between"

    def test_none(self):
        assert extract_direction("GDP growth in 2025?") is None


class TestExtractThreshold:
    def test_percentage(self):
        assert extract_threshold("Will unemployment be above 4.3%?") == 4.3

    def test_dollar(self):
        assert extract_threshold("Will BTC be above $80,000?") == 80000

    def test_no_threshold(self):
        assert extract_threshold("GDP growth in 2025?") is None


class TestDirectionsCompatible:
    def test_same(self):
        assert directions_compatible("above", "above") is True

    def test_opposite(self):
        assert directions_compatible("above", "below") is False

    def test_above_exactly(self):
        assert directions_compatible("above", "exactly") is False

    def test_none(self):
        assert directions_compatible(None, "above") is True


class TestThresholdsCompatible:
    def test_same(self):
        assert thresholds_compatible(4.3, 4.3) is True

    def test_close_percentage(self):
        assert thresholds_compatible(4.3, 4.1) is True  # within 0.5

    def test_different(self):
        assert thresholds_compatible(4.3, 3.4) is False

    def test_close_dollar(self):
        assert thresholds_compatible(80000, 81000) is True  # within 5%

    def test_far_dollar(self):
        assert thresholds_compatible(80000, 60000) is False

    def test_none(self):
        assert thresholds_compatible(None, 4.3) is True


class TestFilterCandidates:
    def test_filters_low_confidence(self):
        candidates = [{"confidence_score": 0.3, "kalshi_question": "test", "polymarket_question": "test"}]
        assert len(filter_candidates(candidates)) == 0

    def test_filters_temporal_mismatch(self):
        candidates = [{
            "confidence_score": 0.9,
            "kalshi_question": "Will unemployment be above 4.3%?",
            "polymarket_question": "Will unemployment be above 4.3%?",
            "kalshi_resolution_date": "2026-01-15T00:00:00Z",
            "polymarket_resolution_date": "2022-01-15T00:00:00Z",
        }]
        assert len(filter_candidates(candidates)) == 0

    def test_filters_direction_mismatch(self):
        candidates = [{
            "confidence_score": 0.9,
            "kalshi_question": "Will unemployment be above 4.3%?",
            "polymarket_question": "Will unemployment be below 4.3%?",
            "kalshi_resolution_date": "2026-01-15T00:00:00Z",
            "polymarket_resolution_date": "2026-01-20T00:00:00Z",
        }]
        assert len(filter_candidates(candidates)) == 0

    def test_passes_good_pair(self):
        candidates = [{
            "confidence_score": 0.9,
            "kalshi_question": "Will unemployment be above 4.3%?",
            "polymarket_question": "Will U-3 unemployment be above 4.3%?",
            "kalshi_resolution_date": "2026-01-15T00:00:00Z",
            "polymarket_resolution_date": "2026-01-10T00:00:00Z",
        }]
        result = filter_candidates(candidates)
        assert len(result) == 1
        assert result[0]["resolution_gap_days"] == 5


# --------------------------------------------------------------------
# filter_active_match: operates on active_matches.json schema.
# Catches the specific garbage patterns found in live paper trading data.
# --------------------------------------------------------------------


class TestFilterActiveMatch:
    """Rules for the live-trading quality filter.

    Each test case is drawn from an actual pair in data/live/active_matches.json
    that was either profitable (should pass) or systematically losing money
    (should be rejected).
    """

    # ----- Good pairs that MUST pass -----

    def test_oil_near_expiry_passes(self):
        """WTI front-month settle vs WTI April price. Real winner."""
        match = {
            "kalshi_ticker": "KXWTIW-26APR10-T106.00",
            "kalshi_title": "Will the WTI front-month settle oil price be <106.00 on Apr 10, 2026?",
            "poly_title": "What will WTI Crude Oil (WTI) hit in April 2026?",
            "similarity": 0.92,
        }
        ok, reason = filter_active_match(match)
        assert ok is True, f"good oil pair was rejected: {reason}"
        assert reason is None

    def test_aoc_democratic_nominee_passes(self):
        """Both sides ask the same discrete question. Real good match."""
        match = {
            "kalshi_ticker": "KXPRESNOMD-28-AOC",
            "kalshi_title": "Will Alexandria Ocasio-Cortez be the Democratic Presidential nominee in 2028?",
            "poly_title": "Democratic Presidential Nominee 2028 - Will Alexandria Ocasio-Cortez win the 2028 Democratic presidential nomination?",
            "similarity": 0.97,
        }
        ok, reason = filter_active_match(match)
        assert ok is True, f"good AOC pair was rejected: {reason}"

    # ----- Bad pattern 1: NBA season-wins vs championship -----

    def test_rejects_nba_season_wins_vs_champion(self):
        """Kalshi KXNBAWINS-* is an O/U on season wins. Polymarket is a champion market.
        These converge only by coincidence and have been losing consistently."""
        match = {
            "kalshi_ticker": "KXNBAWINS-SAS-25-T35",
            "kalshi_title": "Will the San Antonio pro basketball team win at least 35 times this season?",
            "poly_title": "2026 NBA Champion - Will the San Antonio Spurs win the 2026 NBA Finals?",
            "similarity": 0.86,
        }
        ok, reason = filter_active_match(match)
        assert ok is False
        assert "nba" in reason.lower() or "season_wins_vs_champion" in reason

    def test_rejects_nba_wins_any_team(self):
        """Same pattern on a different team+threshold."""
        match = {
            "kalshi_ticker": "KXNBAWINS-SAS-25-T60",
            "kalshi_title": "Will the San Antonio pro basketball team win at least 60 times this season?",
            "poly_title": "2026 NBA Champion - Will the San Antonio Spurs win the 2026 NBA Finals?",
            "similarity": 0.88,
        }
        ok, reason = filter_active_match(match)
        assert ok is False

    # ----- Bad pattern 2: Fed decision year mismatch -----

    def test_rejects_fed_2027_vs_april_2026(self):
        """Kalshi ticker encodes '27APR' (April 2027) but Polymarket is April 2026."""
        match = {
            "kalshi_ticker": "KXFEDDECISION-27APR-H0",
            "kalshi_title": "Will the Federal Reserve Hike rates by 0bps at their April 2027 meeting?",
            "poly_title": "Fed decision in April? - Will there be no change in Fed interest rates after the April meeting?",
            "similarity": 0.89,
        }
        ok, reason = filter_active_match(match)
        assert ok is False
        assert "year" in reason.lower() or "date" in reason.lower()

    def test_rejects_fed_27jun_vs_april_2026(self):
        match = {
            "kalshi_ticker": "KXFEDDECISION-27JUN-H0",
            "kalshi_title": "Will the Federal Reserve Hike rates by 0bps at their June 2027 meeting?",
            "poly_title": "Fed decision in April? - Will there be no change in Fed interest rates after the April meeting?",
            "similarity": 0.87,
        }
        ok, reason = filter_active_match(match)
        assert ok is False

    # ----- Bad pattern 3: Cross-topic cabinet vs nomination -----

    def test_rejects_cabinet_visit_vs_presidential_nomination(self):
        """Kalshi is about a State Department visit. Polymarket is about a 2028 nomination."""
        match = {
            "kalshi_ticker": "KXSECSTATEVISIT-27-MEX",
            "kalshi_title": "Will the Secretary of State visit Mexico before 2027?",
            "poly_title": "Republican Presidential Nominee 2028 - Will Marco Rubio win the 2028 Republican presidential nomination?",
            "similarity": 0.82,
        }
        ok, reason = filter_active_match(match)
        assert ok is False

    # ----- Bad pattern 4: ticker-year vs title-year mismatch -----

    def test_rejects_ticker_year_vs_title_year_mismatch(self):
        """Brazil inflation case from real live data: Kalshi ticker encodes
        '25DEC' (December 2025, already past) but Polymarket title explicitly
        says '2026'. Filter should reject even though neither Kalshi title nor
        Polymarket title would trigger the title-only year rule (the Kalshi
        title may not mention the year at all, only the ticker does)."""
        match = {
            "kalshi_ticker": "KXBRAZILINF-25DEC-T3.25",
            "kalshi_title": "Will Brazil inflation be above 3.25%?",
            "poly_title": "Will inflation reach more than 5% in 2026?",
            "similarity": 0.80,
        }
        ok, reason = filter_active_match(match)
        assert ok is False
        assert "year" in reason.lower()

    def test_rejects_stale_ticker_past_year(self):
        """A Kalshi contract whose ticker-encoded year is already in the past
        is meaningless to trade — the contract has (or should have) resolved.
        Kalshi occasionally leaves past-dated markets status=open, and we
        should never match against them."""
        match = {
            "kalshi_ticker": "KXSOMETHING-24DEC-T3.25",  # December 2024
            "kalshi_title": "Will something happen in Dec 2024?",
            "poly_title": "Will something happen in 2026?",
            "similarity": 0.85,
        }
        ok, reason = filter_active_match(match)
        assert ok is False
        assert "stale" in reason.lower() or "past" in reason.lower()

    def test_accepts_same_year_contract(self):
        """Sanity: ticker and title both indicate same year -> accept."""
        match = {
            "kalshi_ticker": "KXWTI-26APR10-T106.00",
            "kalshi_title": "Will WTI be below $106 on April 10, 2026?",
            "poly_title": "What will WTI Crude Oil hit in April 2026?",
            "similarity": 0.82,
        }
        ok, reason = filter_active_match(match)
        assert ok is True, f"Rejected good oil pair: {reason}"

    # ----- Bad pattern 5: numeric threshold vs ranking -----

    def test_rejects_net_worth_threshold_vs_richest_ranking(self):
        """Real case from discovery dry-run: KXMUSKNW-26APR30-T600 is a
        Kalshi contract on Musk's net worth > $600B on April 30.
        Polymarket is asking 'Will Musk be richest person on December 31?'
        Completely different question types — threshold on a number vs
        ranking position — even though both mention Musk."""
        match = {
            "kalshi_ticker": "KXMUSKNW-26APR30-T600",
            "kalshi_title": "Will Elon Musk's net worth be above $600B on April 30, 2026?",
            "poly_title": "Will Elon Musk be richest person on December 31?",
            "similarity": 0.82,
        }
        ok, reason = filter_active_match(match)
        assert ok is False
        assert "threshold" in reason.lower() or "ranking" in reason.lower()

    def test_rejects_threshold_vs_wealthiest(self):
        match = {
            "kalshi_ticker": "KXBEZNW-26JUN30-T300",
            "kalshi_title": "Will Bezos net worth exceed $300B by June 30?",
            "poly_title": "Will Bezos be the wealthiest person in 2026?",
            "similarity": 0.80,
        }
        ok, reason = filter_active_match(match)
        assert ok is False

    def test_accepts_two_threshold_contracts(self):
        """If BOTH sides are threshold contracts on the same underlying,
        accept — the thresholds differ but it's still a tradable pair
        (different strikes on the same asset)."""
        match = {
            "kalshi_ticker": "KXBTC-26APR30-T100000",
            "kalshi_title": "Will Bitcoin be above $100,000 on April 30?",
            "poly_title": "Will Bitcoin reach $95,000 in April 2026?",
            "similarity": 0.85,
        }
        ok, reason = filter_active_match(match)
        assert ok is True

    # ----- Bad pattern 6: Fed month mismatch within same year -----

    def test_rejects_fed_december_vs_june_same_year(self):
        """KXFEDDECISION-26DEC-* (December 2026) matched against a
        Polymarket question about the June 2026 meeting. Same year so
        fed_year_mismatch doesn't fire — need month awareness."""
        match = {
            "kalshi_ticker": "KXFEDDECISION-26DEC-C25",
            "kalshi_title": "Will the Federal Reserve Cut rates by 25bps at their December 2026 meeting?",
            "poly_title": "Will the Fed decrease interest rates by 25 bps after the June meeting?",
            "similarity": 0.83,
        }
        ok, reason = filter_active_match(match)
        assert ok is False
        assert "fed" in reason.lower() and ("month" in reason.lower() or "mismatch" in reason.lower())

    def test_accepts_fed_same_month(self):
        """Regression: KXFEDDECISION-26JUN-* vs a 'June meeting' question
        is a LEGITIMATE match and must still pass."""
        match = {
            "kalshi_ticker": "KXFEDDECISION-26JUN-C25",
            "kalshi_title": "Will the Federal Reserve Cut rates by 25bps at their June 2026 meeting?",
            "poly_title": "Will the Fed decrease interest rates by 25 bps after the June meeting?",
            "similarity": 0.85,
        }
        ok, reason = filter_active_match(match)
        assert ok is True, f"Rejected same-month Fed pair: {reason}"

    # ----- Bad pattern 7: threshold contract vs policy/event market -----

    def test_rejects_argentina_inflation_vs_dollarize(self):
        """KXARMOMINF (Argentina monthly inflation, numeric threshold)
        matched against 'Will Argentina dollarize?'. They share the
        word 'Argentina' but are totally different question types —
        continuous statistic vs binary policy decision."""
        match = {
            "kalshi_ticker": "KXARMOMINF-26APR14-T2.5",
            "kalshi_title": "Will Argentina monthly inflation be above 2.5% in April 2026?",
            "poly_title": "Will Argentina dollarize by June 30, 2026?",
            "similarity": 0.75,
        }
        ok, reason = filter_active_match(match)
        assert ok is False
        # Accept either new reason wording
        assert any(k in reason.lower() for k in ("threshold", "policy", "event", "ranking"))

    def test_rejects_threshold_vs_adoption_event(self):
        match = {
            "kalshi_ticker": "KXBTC-26JUN30-T150000",
            "kalshi_title": "Will Bitcoin be above $150000 on June 30, 2026?",
            "poly_title": "Will any country adopt Bitcoin as legal tender in 2026?",
            "similarity": 0.78,
        }
        ok, reason = filter_active_match(match)
        assert ok is False

    # ----- Bad pattern 8: Low similarity safety net -----

    def test_rejects_low_similarity(self):
        """Anything under the similarity floor is rejected regardless of content."""
        match = {
            "kalshi_ticker": "KXFOO-26-BAR",
            "kalshi_title": "Will X happen?",
            "poly_title": "Will Y happen?",
            "similarity": 0.65,
        }
        ok, reason = filter_active_match(match)
        assert ok is False
        assert "similar" in reason.lower()

    # ----- Required fields -----

    def test_rejects_missing_ticker(self):
        match = {
            "kalshi_title": "something",
            "poly_title": "something else",
            "similarity": 0.9,
        }
        ok, reason = filter_active_match(match)
        assert ok is False
        assert "ticker" in reason.lower() or "required" in reason.lower()

    def test_rejects_missing_poly_title(self):
        match = {
            "kalshi_ticker": "KXFOO-26-BAR",
            "kalshi_title": "something",
            "similarity": 0.9,
        }
        ok, reason = filter_active_match(match)
        assert ok is False

    # ----- Bad pattern: AAA retail gas date/geography mismatch -----
    #
    # Observed on 2026-04-11 after the commodity discovery fix (task #29)
    # added 205 KXAAAGAS* pairs. The underlying commodity is actually the
    # same (US retail regular gas, $/gallon) but the pairs fail on two
    # structural axes:
    #
    #   a) Geography: Kalshi KXAAAGAS*CA/FL/NY/TX are state-specific
    #      contracts, Polymarket "Will gas hit $X by April 30?" is
    #      national. State and national averages don't converge.
    #
    #   b) Resolution date: Kalshi KXAAAGAS*MAX/MIN tickers encode
    #      a year-end DEC31 resolution, Polymarket resolves April 30.
    #      Same year, different months, the contracts don't overlap
    #      in time.
    #
    # The semantic matcher scores all of these at 0.71-0.79 because
    # both sides mention "gas", but they're structurally different
    # questions.

    def test_rejects_aaa_gas_state_vs_national(self):
        """KXAAAGASMAXCA is California; Poly market is US national."""
        match = {
            "kalshi_ticker": "KXAAAGASMAXCA-26DEC31-4.30",
            "kalshi_title": "Will average gas prices be above or below $4.30 by Dec 31, 2026?",
            "poly_title": "Will gas hit (Low) $3.95 by April 30?",
            "similarity": 0.763,
        }
        ok, reason = filter_active_match(match)
        assert ok is False
        assert "aaa_gas" in (reason or "").lower() or "geography" in (reason or "").lower() or "state" in (reason or "").lower()

    def test_rejects_aaa_gas_ny_vs_national(self):
        """New York variant of the same geography mismatch."""
        match = {
            "kalshi_ticker": "KXAAAGASMAXNY-26DEC31-3.10",
            "kalshi_title": "Will average gas prices be above or below $3.10 by Dec 31, 2026?",
            "poly_title": "Will gas hit (Low) $3.95 by April 30?",
            "similarity": 0.763,
        }
        ok, reason = filter_active_match(match)
        assert ok is False

    def test_rejects_aaa_gas_min_texas_vs_national(self):
        """Texas min variant."""
        match = {
            "kalshi_ticker": "KXAAAGASMINTX-26DEC31-2.30",
            "kalshi_title": "Will average gas prices be above or below $2.30 by Dec 31, 2026?",
            "poly_title": "Will gas hit (Low) $3.95 by April 30?",
            "similarity": 0.762,
        }
        ok, reason = filter_active_match(match)
        assert ok is False

    def test_rejects_aaa_gas_max_vs_april_national(self):
        """KXAAAGASMAX is annual max (resolves Dec 31); Poly resolves April 30.
        Different time windows → structurally different questions."""
        match = {
            "kalshi_ticker": "KXAAAGASMAX-26DEC31-4.00",
            "kalshi_title": "Will average gas prices be above $4.00 by Dec 31, 2026?",
            "poly_title": "Will gas hit (High) $4.50 by April 30?",
            "similarity": 0.772,
        }
        ok, reason = filter_active_match(match)
        assert ok is False
        assert "aaa_gas" in (reason or "").lower() or "month" in (reason or "").lower() or "date" in (reason or "").lower()

    def test_rejects_aaa_gas_eoy_november_vs_april(self):
        """KXAAAGASED November tickers vs April Poly."""
        match = {
            "kalshi_ticker": "KXAAAGASED-26NOV03-2.50",
            "kalshi_title": "Will average gas prices be above $2.50?",
            "poly_title": "Will gas hit (Low) $3.85 by April 30?",
            "similarity": 0.711,
        }
        ok, reason = filter_active_match(match)
        assert ok is False

    def test_aaa_gas_same_month_national_passes(self):
        """Weekly national AAA gas with matching month should still pass.
        The filter only enforces structural (date/geography) rules —
        legitimate same-month same-national pairs remain tradable
        even if strike prices differ."""
        match = {
            "kalshi_ticker": "KXAAAGASW-26APR13-4.000",
            "kalshi_title": "Will average gas prices be above $4.000?",
            "poly_title": "Will gas hit (High) $4.50 by April 30?",
            "similarity": 0.724,
        }
        ok, reason = filter_active_match(match)
        assert ok is True, f"same-month national AAA gas pair was rejected: {reason}"


class TestFilterActiveMatchesBatch:
    def test_batch_returns_passed_and_stats(self):
        matches = [
            # Good oil pair
            {
                "kalshi_ticker": "KXWTIW-26APR10-T106.00",
                "kalshi_title": "Will the WTI front-month settle oil price be <106.00 on Apr 10, 2026?",
                "poly_title": "What will WTI Crude Oil (WTI) hit in April 2026?",
                "similarity": 0.92,
            },
            # NBA garbage
            {
                "kalshi_ticker": "KXNBAWINS-SAS-25-T35",
                "kalshi_title": "Will the San Antonio pro basketball team win at least 35 times this season?",
                "poly_title": "2026 NBA Champion - Will the San Antonio Spurs win the 2026 NBA Finals?",
                "similarity": 0.86,
            },
            # Fed year mismatch
            {
                "kalshi_ticker": "KXFEDDECISION-27APR-H0",
                "kalshi_title": "Will the Federal Reserve Hike rates by 0bps at their April 2027 meeting?",
                "poly_title": "Fed decision in April? - Will there be no change in Fed interest rates after the April meeting?",
                "similarity": 0.89,
            },
        ]
        passed, stats = filter_active_matches(matches)
        assert len(passed) == 1
        assert passed[0]["kalshi_ticker"] == "KXWTIW-26APR10-T106.00"
        assert stats["total"] == 3
        assert stats["passed"] == 1
        assert stats["rejected"] == 2
        # Each rejection reason should appear in the reasons dict
        assert len(stats["reasons"]) >= 1
