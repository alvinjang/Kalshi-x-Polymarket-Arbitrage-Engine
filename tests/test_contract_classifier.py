"""Tests for contract classifier -- ticker parsing, tier classification, reclassification.

Covers 8+ Kalshi ticker formats, tier boundary logic, sports/politics defaults,
API fallback (mocked), and dynamic reclassification.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.live.contract_classifier import (
    BAR_INTERVALS,
    ContractClassifier,
    Tier,
)


# ---------------------------------------------------------------------------
# Tier enum tests
# ---------------------------------------------------------------------------


class TestTierEnum:
    """Tests for the Tier enum and its bar_interval_seconds property."""

    def test_daily_bar_interval(self):
        assert Tier.DAILY.bar_interval_seconds == 900  # 15 min

    def test_weekly_bar_interval(self):
        assert Tier.WEEKLY.bar_interval_seconds == 3600  # 1h

    def test_monthly_bar_interval(self):
        assert Tier.MONTHLY.bar_interval_seconds == 14400  # 4h

    def test_quarterly_bar_interval(self):
        assert Tier.QUARTERLY.bar_interval_seconds == 86400  # daily

    def test_unknown_bar_interval(self):
        assert Tier.UNKNOWN.bar_interval_seconds == 14400  # default 4h

    def test_bar_intervals_dict(self):
        """BAR_INTERVALS backward compat dict matches enum."""
        assert BAR_INTERVALS[Tier.DAILY] == 900
        assert BAR_INTERVALS[Tier.WEEKLY] == 3600
        assert BAR_INTERVALS[Tier.MONTHLY] == 14400
        assert BAR_INTERVALS[Tier.QUARTERLY] == 86400


# ---------------------------------------------------------------------------
# Ticker parsing: Format 1 -- full YYMMMDD
# ---------------------------------------------------------------------------


class TestParseResolutionDate:
    """Tests for parse_resolution_date across all 8+ ticker format categories."""

    def setup_method(self):
        self.classifier = ContractClassifier()

    # Format 1: YYMMMDD with threshold suffix (daily oil/BTC)
    def test_format1_wti_daily(self):
        result = self.classifier.parse_resolution_date("KXWTI-26APR08-T105.99")
        assert result is not None
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 8

    def test_format1_wtiw_weekly(self):
        result = self.classifier.parse_resolution_date("KXWTIW-26APR10-T118.99")
        assert result is not None
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 10

    def test_format1_wtiw_band(self):
        result = self.classifier.parse_resolution_date("KXWTIW-26APR10-B118.5")
        assert result is not None
        assert result.month == 4
        assert result.day == 10

    def test_format1_btcd_special(self):
        """BTCD format: KXBTCD-26MAR3122-T67099.99 -- day=31 embedded in '3122'."""
        result = self.classifier.parse_resolution_date(
            "KXBTCD-26MAR3122-T67099.99"
        )
        assert result is not None
        assert result.year == 2026
        assert result.month == 3
        assert result.day == 31

    # Format 2: YYMMMDD with alphanumeric suffix (monthly BTC)
    def test_format2_btcmaxmon(self):
        result = self.classifier.parse_resolution_date(
            "KXBTCMAXMON-BTC-26APR30-7500000"
        )
        assert result is not None
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 30

    def test_format2_btcminmon(self):
        result = self.classifier.parse_resolution_date(
            "KXBTCMINMON-BTC-26APR30-6500000"
        )
        assert result is not None
        assert result.month == 4
        assert result.day == 30

    # Format 3: YYMMMDD yearly
    def test_format3_btcmaxy(self):
        result = self.classifier.parse_resolution_date(
            "KXBTCMAXY-26DEC31-149999.99"
        )
        assert result is not None
        assert result.year == 2026
        assert result.month == 12
        assert result.day == 31

    def test_format3_btcminy(self):
        result = self.classifier.parse_resolution_date(
            "KXBTCMINY-27JAN01-55000.00"
        )
        assert result is not None
        assert result.year == 2027
        assert result.month == 1
        assert result.day == 1

    def test_format3_btcy(self):
        """KXBTCY-27JAN0100-T20000.00 -- '0100' means day=01."""
        result = self.classifier.parse_resolution_date(
            "KXBTCY-27JAN0100-T20000.00"
        )
        assert result is not None
        assert result.year == 2027
        assert result.month == 1
        assert result.day == 1

    def test_format3_wtimax(self):
        result = self.classifier.parse_resolution_date("KXWTIMAX-26DEC31-T125")
        assert result is not None
        assert result.year == 2026
        assert result.month == 12
        assert result.day == 31

    def test_format3_ratecut(self):
        result = self.classifier.parse_resolution_date("KXRATECUT-26DEC31")
        assert result is not None
        assert result.year == 2026
        assert result.month == 12
        assert result.day == 31

    def test_format3_venezuela(self):
        result = self.classifier.parse_resolution_date(
            "KXVENEZUELALEADER-26DEC31-MRUB"
        )
        assert result is not None
        assert result.year == 2026
        assert result.month == 12
        assert result.day == 31

    # Format 4: YYMM only (Fed meetings)
    def test_format4_feddecision(self):
        result = self.classifier.parse_resolution_date(
            "KXFEDDECISION-26APR-C25"
        )
        assert result is not None
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 30  # end of April

    def test_format4_fed(self):
        result = self.classifier.parse_resolution_date("KXFED-27APR-T0.00")
        assert result is not None
        assert result.year == 2027
        assert result.month == 4
        assert result.day == 30

    def test_format4_fedcombo(self):
        result = self.classifier.parse_resolution_date("KXFEDCOMBO-26APR-0-0")
        assert result is not None
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 30

    # Format 5: Year-only politics (needs API or defaults)
    def test_format5_pres_person(self):
        """KXPRESPERSON-28-GNEWS -> election day default (Nov 5, 2028)."""
        result = self.classifier.parse_resolution_date(
            "KXPRESPERSON-28-GNEWS"
        )
        assert result is not None
        assert result.year == 2028
        assert result.month == 11
        assert result.day == 5

    def test_format5_presnomd_needs_api(self):
        """KXPRESNOMD-28-AOC -> None (needs API lookup)."""
        result = self.classifier.parse_resolution_date("KXPRESNOMD-28-AOC")
        assert result is None

    def test_format5_presnomr_convention(self):
        """KXPRESNOMR-28-MR -> convention default (Aug 31, 2028)."""
        result = self.classifier.parse_resolution_date("KXPRESNOMR-28-MR")
        assert result is not None
        assert result.year == 2028
        assert result.month == 8
        assert result.day == 31

    def test_format5_vpresnomd_convention(self):
        """KXVPRESNOMD-28-AOC -> convention default (Aug 31, 2028)."""
        result = self.classifier.parse_resolution_date("KXVPRESNOMD-28-AOC")
        assert result is not None
        assert result.year == 2028
        assert result.month == 8
        assert result.day == 31

    def test_format5_vpresnomr_convention(self):
        """KXVPRESNOMR-28-MR -> convention default (Aug 31, 2028)."""
        result = self.classifier.parse_resolution_date("KXVPRESNOMR-28-MR")
        assert result is not None
        assert result.year == 2028
        assert result.month == 8
        assert result.day == 31

    def test_format5_pres_election_occur(self):
        """KXPRESELECTIONOCCUR-28 -> election day Nov 5."""
        result = self.classifier.parse_resolution_date(
            "KXPRESELECTIONOCCUR-28"
        )
        assert result is not None
        assert result.year == 2028
        assert result.month == 11
        assert result.day == 5

    def test_format5_presendorsemuskd(self):
        """KXPRESENDORSEMUSKD-28 -> election day Nov 5."""
        result = self.classifier.parse_resolution_date(
            "KXPRESENDORSEMUSKD-28"
        )
        assert result is not None
        assert result.year == 2028
        assert result.month == 11
        assert result.day == 5

    def test_format5_aocsenate(self):
        """KXAOCSENATE-28 -> election day Nov 5."""
        result = self.classifier.parse_resolution_date("KXAOCSENATE-28")
        assert result is not None
        assert result.year == 2028
        assert result.month == 11
        assert result.day == 5

    def test_format5_2028drun(self):
        """KX2028DRUN-28-CBOOK -> election day Nov 5."""
        result = self.classifier.parse_resolution_date("KX2028DRUN-28-CBOOK")
        assert result is not None
        assert result.year == 2028
        assert result.month == 11
        assert result.day == 5

    def test_format5_pres_party(self):
        """KXPRESPARTY-2028-D -> election day Nov 5."""
        result = self.classifier.parse_resolution_date("KXPRESPARTY-2028-D")
        assert result is not None
        assert result.year == 2028
        assert result.month == 11
        assert result.day == 5

    def test_format5_senate_pad(self):
        """KXSENATEPAD-28-BBOY -> election day Nov 5."""
        result = self.classifier.parse_resolution_date("KXSENATEPAD-28-BBOY")
        assert result is not None
        assert result.year == 2028
        assert result.month == 11
        assert result.day == 5

    def test_format5_senatewa(self):
        """SENATEWA-28-D -> election day Nov 5."""
        result = self.classifier.parse_resolution_date("SENATEWA-28-D")
        assert result is not None
        assert result.year == 2028
        assert result.month == 11
        assert result.day == 5

    # Format 6: Year-only sports
    def test_format6_nba(self):
        result = self.classifier.parse_resolution_date("KXNBA-26-SAS")
        assert result is not None
        assert result.year == 2026
        assert result.month == 6
        assert result.day == 30

    def test_format6_nba_east(self):
        result = self.classifier.parse_resolution_date("KXNBAEAST-26-CHA")
        assert result is not None
        assert result.month == 6
        assert result.day == 30

    def test_format6_nba_west(self):
        result = self.classifier.parse_resolution_date("KXNBAWEST-26-SAS")
        assert result is not None
        assert result.month == 6
        assert result.day == 30

    def test_format6_nba_wins(self):
        """KXNBAWINS-SAS-25-T20 -- year is 25, NBA season ends June."""
        result = self.classifier.parse_resolution_date("KXNBAWINS-SAS-25-T20")
        assert result is not None
        assert result.year == 2025
        assert result.month == 6
        assert result.day == 30

    def test_format6_nba_top_pick(self):
        result = self.classifier.parse_resolution_date("KXNBATOPPICK-26-GSW")
        assert result is not None
        assert result.month == 6
        assert result.day == 30

    def test_format6_mlb(self):
        result = self.classifier.parse_resolution_date("KXMLB-26-PHI")
        assert result is not None
        assert result.month == 10
        assert result.day == 31

    def test_format6_mlbnl(self):
        result = self.classifier.parse_resolution_date("KXMLBNL-26-PHI")
        assert result is not None
        assert result.month == 10
        assert result.day == 31

    def test_format6_world_cup(self):
        result = self.classifier.parse_resolution_date("KXMENWORLDCUP-26-US")
        assert result is not None
        assert result.year == 2026
        assert result.month == 7
        assert result.day == 19

    def test_format6_wc_group_win(self):
        result = self.classifier.parse_resolution_date(
            "KXWCGROUPWIN-26B-BIH"
        )
        assert result is not None
        assert result.month == 7
        assert result.day == 19

    def test_format6_wc_round(self):
        result = self.classifier.parse_resolution_date(
            "KXWCROUND-26FINAL-ARG"
        )
        assert result is not None
        assert result.month == 7
        assert result.day == 19

    def test_format6_wc_congo(self):
        """KXWCCONGO-26 -> World Cup default."""
        result = self.classifier.parse_resolution_date("KXWCCONGO-26")
        assert result is not None
        assert result.month == 7
        assert result.day == 19

    def test_format6_f1(self):
        result = self.classifier.parse_resolution_date("KXF1-26-CL")
        assert result is not None
        assert result.month == 12
        assert result.day == 31

    def test_format6_ucl(self):
        result = self.classifier.parse_resolution_date("KXUCL-26-ATM")
        assert result is not None
        assert result.month == 6
        assert result.day == 1

    def test_format6_uel(self):
        result = self.classifier.parse_resolution_date("KXUEL-26-FCP")
        assert result is not None
        assert result.month == 5
        assert result.day == 31

    def test_format6_laliga(self):
        result = self.classifier.parse_resolution_date("KXLALIGA-26-BAR")
        assert result is not None
        assert result.month == 5
        assert result.day == 31

    def test_format6_laliga_relegation(self):
        result = self.classifier.parse_resolution_date(
            "KXLALIGARELEGATION-26-SEV"
        )
        assert result is not None
        assert result.month == 5
        assert result.day == 31

    def test_format6_ligamx(self):
        result = self.classifier.parse_resolution_date("KXLIGAMX-26CLA-CDG")
        assert result is not None
        assert result.month == 5
        assert result.day == 31

    def test_format6_eurovision(self):
        result = self.classifier.parse_resolution_date("KXEUROVISION-26-AUS")
        assert result is not None
        assert result.month == 5
        assert result.day == 31

    def test_format6_eurovision_participants(self):
        result = self.classifier.parse_resolution_date(
            "KXEUROVISIONPARTICIPANTS-26-PRT"
        )
        assert result is not None
        assert result.month == 5
        assert result.day == 31

    def test_format6_eurovision_telev(self):
        result = self.classifier.parse_resolution_date(
            "KXEUROVISIONTELEV-26-GRE"
        )
        assert result is not None
        assert result.month == 5
        assert result.day == 31

    def test_format6_sb(self):
        result = self.classifier.parse_resolution_date("KXSB-27-NE")
        assert result is not None
        assert result.year == 2027
        assert result.month == 2
        assert result.day == 15

    def test_format6_copadelrey(self):
        result = self.classifier.parse_resolution_date("KXCOPADELREY-26-ATM")
        assert result is not None
        assert result.month == 4
        assert result.day == 30

    def test_format6_teams_in_nba_finals(self):
        result = self.classifier.parse_resolution_date(
            "KXTEAMSINNBAF-26-SASDET"
        )
        assert result is not None
        assert result.month == 6
        assert result.day == 30

    def test_format6_teams_in_nba_east_finals(self):
        result = self.classifier.parse_resolution_date(
            "KXTEAMSINNBAEF-26-DETBOS"
        )
        assert result is not None
        assert result.month == 6
        assert result.day == 30

    def test_format6_teams_in_ucl(self):
        result = self.classifier.parse_resolution_date(
            "KXTEAMSINUCL-26-RMAATM"
        )
        assert result is not None
        assert result.month == 6
        assert result.day == 1

    # Format 7: Split YY-MMMDD
    def test_format7_trumpchina_may15(self):
        result = self.classifier.parse_resolution_date(
            "KXTRUMPCHINA-26-MAY15"
        )
        assert result is not None
        assert result.year == 2026
        assert result.month == 5
        assert result.day == 15

    def test_format7_trumpchina_jun01(self):
        result = self.classifier.parse_resolution_date(
            "KXTRUMPCHINA-26-JUN01"
        )
        assert result is not None
        assert result.year == 2026
        assert result.month == 6
        assert result.day == 1

    def test_format7_trumpchina_jul01(self):
        result = self.classifier.parse_resolution_date(
            "KXTRUMPCHINA-26-JUL01"
        )
        assert result is not None
        assert result.month == 7
        assert result.day == 1

    # Additional YYMMMDD formats found in data
    def test_brazil_senate(self):
        result = self.classifier.parse_resolution_date(
            "KXBRAZILSENATE-26OCT04-PL"
        )
        assert result is not None
        assert result.month == 10
        assert result.day == 4

    def test_brazil_pres_1r(self):
        result = self.classifier.parse_resolution_date(
            "KXBRAZILPRES1R-26OCT04"
        )
        assert result is not None
        assert result.month == 10
        assert result.day == 4

    def test_america_party_2028(self):
        result = self.classifier.parse_resolution_date(
            "KXAMERICAPARTY2028-28SEP01"
        )
        assert result is not None
        assert result.year == 2028
        assert result.month == 9
        assert result.day == 1

    def test_declare_pres_first_d(self):
        result = self.classifier.parse_resolution_date(
            "KXDECLAREPRESFIRSTD-28NOV07-AOCA"
        )
        assert result is not None
        assert result.year == 2028
        assert result.month == 11
        assert result.day == 7

    def test_elect_iran(self):
        result = self.classifier.parse_resolution_date(
            "KXELECTIRAN-27JAN01"
        )
        assert result is not None
        assert result.year == 2027
        assert result.month == 1
        assert result.day == 1

    def test_iran_democracy(self):
        result = self.classifier.parse_resolution_date(
            "KXIRANDEMOCRACY-27MAR01-T6"
        )
        assert result is not None
        assert result.year == 2027
        assert result.month == 3
        assert result.day == 1

    def test_hungary_parli_party(self):
        result = self.classifier.parse_resolution_date(
            "KXHUNGARYPARLIPARTY-26APR12-TISZ"
        )
        assert result is not None
        assert result.month == 4
        assert result.day == 12

    def test_fidesz_seats(self):
        result = self.classifier.parse_resolution_date(
            "KXFIDESZKDNPSEATS-26APR12-A80"
        )
        assert result is not None
        assert result.month == 4
        assert result.day == 12

    def test_scot_parliament(self):
        result = self.classifier.parse_resolution_date(
            "KXSCOTPARLIAMENT-26MAY07-LAB"
        )
        assert result is not None
        assert result.month == 5
        assert result.day == 7

    def test_next_hungary_pm(self):
        result = self.classifier.parse_resolution_date(
            "KXNEXTHUNGARYPM-26MAY01-PMAG"
        )
        assert result is not None
        assert result.month == 5
        assert result.day == 1

    def test_senate_dem_lead(self):
        result = self.classifier.parse_resolution_date(
            "KXSENATEDEMLEAD-28JAN01-CMUR"
        )
        assert result is not None
        assert result.year == 2028
        assert result.month == 1
        assert result.day == 1

    def test_btc2026200(self):
        result = self.classifier.parse_resolution_date(
            "KXBTC2026200-27JAN01-200000"
        )
        assert result is not None
        assert result.year == 2027
        assert result.month == 1
        assert result.day == 1

    def test_eth_miny(self):
        result = self.classifier.parse_resolution_date(
            "KXETHMINY-27JAN01-750"
        )
        assert result is not None
        assert result.year == 2027
        assert result.month == 1
        assert result.day == 1

    def test_trump_countries(self):
        result = self.classifier.parse_resolution_date(
            "KXTRUMPCOUNTRIES-27JAN01-CHI"
        )
        assert result is not None
        assert result.year == 2027
        assert result.month == 1
        assert result.day == 1

    def test_ucl_game_with_date(self):
        result = self.classifier.parse_resolution_date(
            "KXUCLGAME-26APR14ATMBAR-ATM"
        )
        assert result is not None
        assert result.month == 4
        assert result.day == 14

    def test_laliga_game_with_date(self):
        result = self.classifier.parse_resolution_date(
            "KXLALIGAGAME-26APR04ATMBAR-BAR"
        )
        assert result is not None
        assert result.month == 4
        assert result.day == 4

    # KXBTCMAX100 -- special format: KXBTCMAX100-26-APR (month name, no day)
    def test_btcmax100_month_only(self):
        result = self.classifier.parse_resolution_date("KXBTCMAX100-26-APR")
        assert result is not None
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 30  # end of April

    def test_btcmax100_june(self):
        """KXBTCMAX100-26-JUNE -> note: JUNE not JUN."""
        result = self.classifier.parse_resolution_date("KXBTCMAX100-26-JUNE")
        assert result is not None
        assert result.month == 6
        assert result.day == 30

    # Format 8: Special/no date parseable
    def test_format8_fedchairconfirm(self):
        result = self.classifier.parse_resolution_date(
            "KXFEDCHAIRCONFIRM-JSHE"
        )
        assert result is None

    def test_format8_personpresmam(self):
        result = self.classifier.parse_resolution_date("KXPERSONPRESMAM-45")
        assert result is None

    # Primary-style tickers with embedded date (KXFLPRIMARY-23D26-JMOS)
    def test_fl_primary(self):
        """KXFLPRIMARY-23D26-JMOS -> hard to parse, treat as year-only politics."""
        result = self.classifier.parse_resolution_date(
            "KXFLPRIMARY-23D26-JMOS"
        )
        # This one has "26" as a year hint, should get election default
        assert result is not None
        assert result.month == 11
        assert result.day == 5

    def test_ga_primary(self):
        result = self.classifier.parse_resolution_date(
            "KXGAPRIMARY-12D26-CSMI"
        )
        assert result is not None
        assert result.month == 11
        assert result.day == 5

    def test_nc_primary(self):
        result = self.classifier.parse_resolution_date(
            "KXNCPRIMARY-03D26-RJR."
        )
        assert result is not None
        assert result.month == 11
        assert result.day == 5


# ---------------------------------------------------------------------------
# Tier classification tests
# ---------------------------------------------------------------------------


class TestClassifyContract:
    """Tests for classify_contract tier classification based on time remaining."""

    def setup_method(self):
        self.classifier = ContractClassifier()
        self.now = datetime(2026, 4, 5, 12, 0, 0)

    def test_daily_tier(self):
        """3 days remaining -> DAILY."""
        res_date = self.now + timedelta(days=3)
        tier = self.classifier.classify_contract(res_date, self.now)
        assert tier == Tier.DAILY

    def test_weekly_tier(self):
        """14 days remaining -> WEEKLY."""
        res_date = self.now + timedelta(days=14)
        tier = self.classifier.classify_contract(res_date, self.now)
        assert tier == Tier.WEEKLY

    def test_monthly_tier(self):
        """45 days remaining -> MONTHLY."""
        res_date = self.now + timedelta(days=45)
        tier = self.classifier.classify_contract(res_date, self.now)
        assert tier == Tier.MONTHLY

    def test_quarterly_tier(self):
        """200 days remaining -> QUARTERLY."""
        res_date = self.now + timedelta(days=200)
        tier = self.classifier.classify_contract(res_date, self.now)
        assert tier == Tier.QUARTERLY

    def test_unknown_tier_none_date(self):
        """None resolution date -> UNKNOWN."""
        tier = self.classifier.classify_contract(None, self.now)
        assert tier == Tier.UNKNOWN

    # Boundary tests
    def test_boundary_7_days_is_weekly(self):
        """Exactly 7 days -> WEEKLY (not DAILY)."""
        res_date = self.now + timedelta(days=7)
        tier = self.classifier.classify_contract(res_date, self.now)
        assert tier == Tier.WEEKLY

    def test_boundary_just_under_7_days(self):
        """6.99 days -> DAILY."""
        res_date = self.now + timedelta(days=6, hours=23, minutes=50)
        tier = self.classifier.classify_contract(res_date, self.now)
        assert tier == Tier.DAILY

    def test_boundary_30_days_is_monthly(self):
        """Exactly 30 days -> MONTHLY (not WEEKLY)."""
        res_date = self.now + timedelta(days=30)
        tier = self.classifier.classify_contract(res_date, self.now)
        assert tier == Tier.MONTHLY

    def test_boundary_90_days_is_quarterly(self):
        """Exactly 90 days -> QUARTERLY (not MONTHLY)."""
        res_date = self.now + timedelta(days=90)
        tier = self.classifier.classify_contract(res_date, self.now)
        assert tier == Tier.QUARTERLY

    def test_past_resolution(self):
        """Resolution already passed -> DAILY (0 or negative days)."""
        res_date = self.now - timedelta(days=1)
        tier = self.classifier.classify_contract(res_date, self.now)
        assert tier == Tier.DAILY

    def test_zero_days(self):
        """Same time -> DAILY."""
        tier = self.classifier.classify_contract(self.now, self.now)
        assert tier == Tier.DAILY


# ---------------------------------------------------------------------------
# Dynamic reclassification
# ---------------------------------------------------------------------------


class TestDynamicReclassification:
    """Test that the same contract gets reclassified as time passes."""

    def setup_method(self):
        self.classifier = ContractClassifier()

    def test_reclassification_quarterly_to_daily(self):
        """Same resolution date, different now times -> different tiers."""
        res_date = datetime(2026, 7, 1)

        # 200+ days out -> QUARTERLY
        now_far = datetime(2026, 1, 1)
        assert self.classifier.classify_contract(res_date, now_far) == Tier.QUARTERLY

        # 45 days out -> MONTHLY
        now_mid = datetime(2026, 5, 17)
        assert self.classifier.classify_contract(res_date, now_mid) == Tier.MONTHLY

        # 14 days out -> WEEKLY
        now_near = datetime(2026, 6, 17)
        assert self.classifier.classify_contract(res_date, now_near) == Tier.WEEKLY

        # 2 days out -> DAILY
        now_close = datetime(2026, 6, 29)
        assert self.classifier.classify_contract(res_date, now_close) == Tier.DAILY


# ---------------------------------------------------------------------------
# API fallback (mocked)
# ---------------------------------------------------------------------------


class TestAPIFallback:
    """Test API fallback for unparseable tickers with mocked HTTP."""

    def setup_method(self, tmp_path=None):
        # Use a non-existent cache path so no disk cache interferes
        import tempfile
        self._tmp = tempfile.mkdtemp()
        cache_path = Path(self._tmp) / "test_cache.json"
        self.classifier = ContractClassifier(cache_path=cache_path)

    @patch("src.live.contract_classifier.requests.get")
    def test_fetch_resolution_from_api_success(self, mock_get):
        """Successful API call returns parsed datetime."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "market": {"close_time": "2028-11-07T15:00:00Z"}
        }
        mock_get.return_value = mock_resp

        result = self.classifier.fetch_resolution_from_api("KXPRESNOMD-28-AOC")
        assert result is not None
        assert result.year == 2028
        assert result.month == 11
        assert result.day == 7

    @patch("src.live.contract_classifier.requests.get")
    def test_fetch_resolution_from_api_404(self, mock_get):
        """404 returns None."""
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        result = self.classifier.fetch_resolution_from_api(
            "KXFEDCHAIRCONFIRM-JSHE"
        )
        assert result is None

    @patch("src.live.contract_classifier.requests.get")
    def test_fetch_resolution_from_api_caches(self, mock_get):
        """Second call uses cache, not API."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "market": {"close_time": "2028-11-07T15:00:00Z"}
        }
        mock_get.return_value = mock_resp

        # First call hits API
        self.classifier.fetch_resolution_from_api("KXPRESNOMD-28-AOC")
        # Second call uses cache
        self.classifier.fetch_resolution_from_api("KXPRESNOMD-28-AOC")

        assert mock_get.call_count == 1

    @patch("src.live.contract_classifier.requests.get")
    def test_fetch_resolution_from_api_exception(self, mock_get):
        """Network error returns None."""
        import requests as req
        mock_get.side_effect = req.exceptions.RequestException("timeout")

        result = self.classifier.fetch_resolution_from_api("KXPRESNOMD-28-AOC")
        assert result is None


# ---------------------------------------------------------------------------
# classify_all_pairs
# ---------------------------------------------------------------------------


class TestClassifyAllPairs:
    """Test classify_all_pairs on a mock matches list."""

    def setup_method(self):
        import tempfile
        self._tmp = tempfile.mkdtemp()
        cache_path = Path(self._tmp) / "test_cache.json"
        self.classifier = ContractClassifier(cache_path=cache_path)

    def test_classify_all_basic(self):
        """Small mock list returns correct structure.

        Pair_ids are now content-addressed via make_pair_id — see
        src/live/pair_ids.py and memory/project_pair_id_schema_bug.md.
        """
        from src.live.pair_ids import make_pair_id

        matches = [
            {"kalshi_ticker": "KXWTI-26APR08-T105.99", "poly_id": "0xaaaaaaaa11", "some_field": "x"},
            {"kalshi_ticker": "KXNBA-26-SAS", "poly_id": "0xbbbbbbbb22", "some_field": "y"},
        ]
        now = datetime(2026, 4, 5, 12, 0, 0)
        result = self.classifier.classify_all_pairs(
            matches, now=now, use_api=False
        )

        assert len(result) == 2

        wti_id = make_pair_id("KXWTI-26APR08-T105.99", "0xaaaaaaaa11")
        nba_id = make_pair_id("KXNBA-26-SAS", "0xbbbbbbbb22")

        # WTI resolves Apr 8 -> ~3 days -> DAILY
        assert result[wti_id]["tier"] == "DAILY"
        assert result[wti_id]["bar_interval_seconds"] == 900
        assert result[wti_id]["kalshi_ticker"] == "KXWTI-26APR08-T105.99"

        # NBA resolves June 30 -> ~86 days -> MONTHLY
        assert result[nba_id]["tier"] == "MONTHLY"
        assert result[nba_id]["bar_interval_seconds"] == 14400

    def test_classify_all_with_unparseable(self):
        """Unparseable ticker without API -> UNKNOWN."""
        from src.live.pair_ids import make_pair_id

        matches = [
            {"kalshi_ticker": "KXFEDCHAIRCONFIRM-JSHE", "poly_id": "0x1234567890"},
        ]
        now = datetime(2026, 4, 5)
        result = self.classifier.classify_all_pairs(
            matches, now=now, use_api=False
        )
        pid = make_pair_id("KXFEDCHAIRCONFIRM-JSHE", "0x1234567890")
        assert result[pid]["tier"] == "UNKNOWN"
        assert result[pid]["resolution_date"] is None

    @patch("src.live.contract_classifier.requests.get")
    def test_classify_all_with_api(self, mock_get):
        """API fills in unparseable ticker."""
        from src.live.pair_ids import make_pair_id

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "market": {"close_time": "2026-05-01T15:00:00Z"}
        }
        mock_get.return_value = mock_resp

        matches = [
            {"kalshi_ticker": "KXPRESNOMD-28-AOC", "poly_id": "0xcccccccc33"},
        ]
        now = datetime(2026, 4, 5)
        result = self.classifier.classify_all_pairs(
            matches, now=now, use_api=True
        )
        pid = make_pair_id("KXPRESNOMD-28-AOC", "0xcccccccc33")
        # API returned May 1 2026 -> ~26 days -> WEEKLY
        assert result[pid]["tier"] == "WEEKLY"
        assert result[pid]["resolution_date"] is not None

    def test_classify_all_days_remaining(self):
        """days_remaining field is correct."""
        matches = [
            {"kalshi_ticker": "KXWTI-26APR08-T105.99", "poly_id": "0xaaaaaaaa11"},
        ]
        from src.live.pair_ids import make_pair_id

        now = datetime(2026, 4, 5, 12, 0, 0)
        result = self.classifier.classify_all_pairs(
            matches, now=now, use_api=False
        )
        pid = make_pair_id("KXWTI-26APR08-T105.99", "0xaaaaaaaa11")
        # Apr 8 23:59 - Apr 5 12:00 ~ 3.5 days
        days = result[pid]["days_remaining"]
        assert days is not None
        assert 3.0 < days < 4.0

    def test_dynamic_reclass_all_pairs(self):
        """Same matches at different times produce different tiers."""
        from src.live.pair_ids import make_pair_id

        matches = [
            {"kalshi_ticker": "KXWTI-26APR08-T105.99", "poly_id": "0xaaaaaaaa11"},
        ]
        pid = make_pair_id("KXWTI-26APR08-T105.99", "0xaaaaaaaa11")
        # 3 days out -> DAILY
        result1 = self.classifier.classify_all_pairs(
            matches, now=datetime(2026, 4, 5, 12, 0, 0), use_api=False
        )
        assert result1[pid]["tier"] == "DAILY"

        # 30+ days out -> MONTHLY (hypothetical now far in past)
        result2 = self.classifier.classify_all_pairs(
            matches, now=datetime(2026, 2, 1), use_api=False
        )
        assert result2[pid]["tier"] == "MONTHLY"
