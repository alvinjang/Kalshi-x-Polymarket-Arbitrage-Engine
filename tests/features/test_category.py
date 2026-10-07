"""Tests for derive_category — Kalshi ticker → asset-class category.

Task #25: category becomes a training feature so tree models (XGBoost)
can learn category-conditional rules and linear models (LR) get
category-specific intercepts. Answers the CLAUDE.md research question
at a new axis: *does a unified model outperform category-conditional
models?*
"""
from __future__ import annotations

import pytest

from src.features.category import (
    derive_category_from_pair_id,
    derive_category_from_ticker,
    CATEGORIES,
)


class TestDeriveCategoryFromTicker:
    """Taxonomy covers the real Kalshi ticker prefixes observed in
    data/processed/train.parquet plus the new pairs added by discovery
    on 2026-04-11 (1000 new pairs across BNB, DOGE, CPI, GDP, etc.)."""

    # --- Crypto ---

    def test_btc_variants(self):
        assert derive_category_from_ticker("KXBTC-26JAN0217B-T95000") == "crypto"
        assert derive_category_from_ticker("KXBTCD-26APR1717-B60000") == "crypto"
        assert derive_category_from_ticker("KXBTCMAX-26DEC31-T200000") == "crypto"
        assert derive_category_from_ticker("KXBTCMAXY-26DEC31-T149999") == "crypto"
        assert derive_category_from_ticker("KXBTCMAXMON-BTC-26APR30") == "crypto"

    def test_eth_variants(self):
        assert derive_category_from_ticker("KXETH-26MAR0617B-T2000") == "crypto"
        assert derive_category_from_ticker("KXETHD-26APR17-B3000") == "crypto"

    def test_bnb_doge_xrp_hype(self):
        assert derive_category_from_ticker("KXBNB-26APR1717-B562") == "crypto"
        assert derive_category_from_ticker("KXBNBD-26APR1114-T579.99") == "crypto"
        assert derive_category_from_ticker("KXDOGE-26APR1117-T0.0599") == "crypto"
        assert derive_category_from_ticker("KXDOGED-26APR14-T0.1") == "crypto"
        assert derive_category_from_ticker("KXXRP-26APR10-T2.5") == "crypto"
        assert derive_category_from_ticker("KXXRPMAXMON-26DEC-T3.0") == "crypto"
        assert derive_category_from_ticker("KXHYPE-26APR-T25") == "crypto"

    # --- Oil ---

    def test_oil_variants(self):
        assert derive_category_from_ticker("KXWTI-26APR08-T107.99") == "oil"
        assert derive_category_from_ticker("KXWTIW-26APR10-T106.00") == "oil"
        assert derive_category_from_ticker("KXWTIMAX-26DEC31-T125") == "oil"
        assert derive_category_from_ticker("KXMEXCUBOIL-26-JUL") == "oil"

    # --- Commodities (non-oil) ---

    def test_commodities(self):
        assert derive_category_from_ticker("KXSILVER-26APR-T30") == "commodities"
        assert derive_category_from_ticker("KXSILVERW-26APR10-T30") == "commodities"
        assert derive_category_from_ticker("KXSILVERMON-26APR30-T35") == "commodities"
        assert derive_category_from_ticker("KXNATGAS-26APR-T3") == "commodities"
        assert derive_category_from_ticker("KXNATGASW-26APR10-T3.5") == "commodities"
        assert derive_category_from_ticker("KXWHEAT-26APR-T600") == "commodities"
        assert derive_category_from_ticker("KXWHEATMON-26APR30-T620") == "commodities"
        assert derive_category_from_ticker("KXCORN-26APR-T450") == "commodities"
        assert derive_category_from_ticker("KXCORNW-26APR10-T460") == "commodities"
        assert derive_category_from_ticker("KXCOCOA-26APR-T8000") == "commodities"
        assert derive_category_from_ticker("KXCOCOAMON-26APR30-T7500") == "commodities"
        assert derive_category_from_ticker("KXNICKEL-26APR-T18000") == "commodities"
        assert derive_category_from_ticker("KXNICKELMON-26APR30-T18500") == "commodities"
        assert derive_category_from_ticker("KXLCATTLE-26APR-T180") == "commodities"
        assert derive_category_from_ticker("KXLCATTLEW-26APR10-T185") == "commodities"

    # --- Inflation (US + foreign) ---

    def test_inflation_us(self):
        assert derive_category_from_ticker("KXCPI-26MAY-T0.1") == "inflation"
        assert derive_category_from_ticker("KXCPIYOY-26APR-T4.1") == "inflation"
        assert derive_category_from_ticker("KXCPICORE-26APR-T0.3") == "inflation"
        assert derive_category_from_ticker("KXCPICOREYOY-26APR-T3.4") == "inflation"
        assert derive_category_from_ticker("KXSHELTERCPI-26MAY12-T424.0") == "inflation"

    def test_inflation_foreign(self):
        assert derive_category_from_ticker("KXUKCPIYOY-26APR22-T3.8") == "inflation"
        assert derive_category_from_ticker("KXCACPIYOY-26APR-T2.5") == "inflation"
        assert derive_category_from_ticker("KXBRAZILINF-26DEC-4.00") == "inflation"
        assert derive_category_from_ticker("KXARMOMINF-26APR14-T2.5") == "inflation"
        assert derive_category_from_ticker("KXJPMOMINF-26APR17-T-0.3") == "inflation"

    # --- GDP ---

    def test_gdp(self):
        assert derive_category_from_ticker("KXGDP-26Q1-T2.5") == "gdp"
        assert derive_category_from_ticker("KXEZGDPQOQF-26APR30-T0.6") == "gdp"
        assert derive_category_from_ticker("KXFRGDPQOQP-26APR-T0.5") == "gdp"
        assert derive_category_from_ticker("KXESGDPYOYF-26APR-T3.0") == "gdp"
        assert derive_category_from_ticker("KXDEGDPYOYF-26APR-T0.5") == "gdp"
        assert derive_category_from_ticker("KXITGDPQOQA-26APR-T0.3") == "gdp"

    # --- Fed rates ---

    def test_fed_and_rates(self):
        assert derive_category_from_ticker("KXFED-27APR-T1.00") == "fed_rates"
        assert derive_category_from_ticker("KXFEDDECISION-26DEC-C25") == "fed_rates"
        assert derive_category_from_ticker("KX3MTBILL-26JUN30-T3.50") == "fed_rates"

    # --- Employment ---

    def test_employment(self):
        assert derive_category_from_ticker("KXU3-25DEC-T4.5") == "employment"
        assert derive_category_from_ticker("KXU3-26JAN-T4.4") == "employment"
        assert derive_category_from_ticker("KXUE-26APR-T4.3") == "employment"
        assert derive_category_from_ticker("KXLAYOFFSYINFO-26-494000") == "employment"

    # --- Politics ---

    def test_politics_election(self):
        assert derive_category_from_ticker("KXPRESNOMD-28-AOC") == "politics_election"
        assert derive_category_from_ticker("KXPRESNOMR-28-MR") == "politics_election"

    def test_politics_policy(self):
        assert derive_category_from_ticker("KXTRUMPMEET-26JAN-XI") == "politics_policy"
        assert derive_category_from_ticker("KXRECOGPALESTINE-26-FRA") == "politics_policy"
        assert derive_category_from_ticker("KXLEADERSOUT-26-PUTIN") == "politics_policy"
        assert derive_category_from_ticker("KXSECSTATEVISIT-27-MEX") == "politics_policy"
        assert derive_category_from_ticker("KXGOVTSHUTLENGTH-26-14") == "politics_policy"
        assert derive_category_from_ticker("KXSOCDEMSEATS-26-30") == "politics_policy"
        assert derive_category_from_ticker("KXDENMARK2ND-26MAY-X") == "politics_policy"
        assert derive_category_from_ticker("KXJAPANHOUSE-28DEC-X") == "politics_policy"

    # --- Sports ---

    def test_sports(self):
        assert derive_category_from_ticker("KXNBAWINS-SAS-25-T35") == "sports"
        assert derive_category_from_ticker("KXNBAGAME-26APR10OKCDEN-DEN") == "sports"
        assert derive_category_from_ticker("KXNBATOTAL-26APR10LACPOR-240") == "sports"
        assert derive_category_from_ticker("KXNFLWINS-NE-25-T8") == "sports"
        assert derive_category_from_ticker("KXMLBWINS-NYY-26-T90") == "sports"
        assert derive_category_from_ticker("KXATPSETWINNER-26-X") == "sports"
        assert derive_category_from_ticker("KXPGATOP10-MAST26-XSCH") == "sports"

    # --- FX ---

    def test_fx(self):
        assert derive_category_from_ticker("KXINXY-26APR-T100") == "fx"
        assert derive_category_from_ticker("KXUSDIRR-26APR-T100000") == "fx"

    # --- Housing / wealth / gas / misc ---

    def test_housing(self):
        assert derive_category_from_ticker("KXSEAHOMEVAL-26MAR19-T722500") == "housing"

    def test_gas_prices(self):
        assert derive_category_from_ticker("KXAAAGASMAXCA-26DEC31-4.30") == "gas_prices"

    def test_wealth(self):
        assert derive_category_from_ticker("KXTRILLIONAIRE-30-MUSK") == "wealth"

    # --- Other / fallback ---

    def test_unknown_returns_other(self):
        assert derive_category_from_ticker("KXUNKNOWN-26APR-T1") == "other"
        assert derive_category_from_ticker("KXWEIRDTHING-26") == "other"

    def test_empty_and_invalid(self):
        assert derive_category_from_ticker("") == "other"
        assert derive_category_from_ticker(None) == "other"


class TestDeriveCategoryFromPairId:
    """Pair IDs in train.parquet are lowercase truncated tickers +
    poly-id hex suffix. Derivation must handle that format."""

    def test_lowercase_pair_id_with_hex_suffix(self):
        assert derive_category_from_pair_id("kxbtc26jan0217b-0x511f8c84") == "crypto"
        assert derive_category_from_pair_id("kxcpiyoy26jant2-0xca2af3d5") == "inflation"
        assert derive_category_from_pair_id("kxu325dect4.5-0x62069242") == "employment"
        assert derive_category_from_pair_id("kxeth26mar0617b-0x0575450c") == "crypto"
        assert derive_category_from_pair_id("kxcpicore26febt-0xc1691f09") == "inflation"

    def test_live_pair_id_not_supported(self):
        """live_0042-style IDs don't encode the ticker — return 'other'
        so the caller can look up via active_matches.json if needed."""
        assert derive_category_from_pair_id("live_0042") == "other"


class TestCategoriesConstant:
    def test_categories_includes_all_expected(self):
        expected = {
            "crypto", "oil", "commodities", "inflation", "gdp",
            "fed_rates", "employment", "politics_election",
            "politics_policy", "sports", "fx", "housing",
            "gas_prices", "wealth", "other",
        }
        assert expected.issubset(set(CATEGORIES))
