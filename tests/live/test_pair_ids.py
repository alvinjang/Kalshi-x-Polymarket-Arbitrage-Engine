"""Regression tests for content-addressed pair_id generation.

Motivated by the 2026-04-11 discovery that three code paths
(collector.py, strategy.py, pair_mapping.json) had disagreeing
live_NNNN schemes. See memory/project_pair_id_schema_bug.md.
"""
from __future__ import annotations

from src.live.pair_ids import make_pair_id


class TestMakePairId:
    def test_hex_condition_id_matches_train_format(self):
        """Format must match train.parquet: {lowercase_nodash}-{0x + 8hex}."""
        pid = make_pair_id("KXWTI-26APR08-T107.99", "0x43d5953daec805127ff71b")
        assert pid == "kxwti26apr08t10799-0x43d5953d"

    def test_kalshi_already_lowercase(self):
        pid = make_pair_id("kxwti-26apr08", "0x43d5953daec805")
        assert pid == "kxwti26apr08-0x43d5953d"

    def test_threshold_period_stripped(self):
        """T107.99 → t10799 to keep pair_ids safe for column names."""
        pid = make_pair_id("KXWTI-T107.99", "0xabcdef1234567890")
        assert pid == "kxwtit10799-0xabcdef12"

    def test_legacy_numeric_poly_id_preserved(self):
        """Old Polymarket numeric 'id' values stay as-is — stable enough."""
        pid = make_pair_id("KXWTI-26APR08-T107.99", "1712297")
        assert pid == "kxwti26apr08t10799-1712297"

    def test_numeric_poly_id_no_truncation(self):
        """Numeric IDs under 10 chars aren't truncated."""
        pid = make_pair_id("KXFOO", "559653")
        assert pid == "kxfoo-559653"

    def test_empty_kalshi_returns_empty(self):
        assert make_pair_id("", "0xdeadbeef") == ""

    def test_empty_poly_returns_empty(self):
        assert make_pair_id("KXFOO", "") == ""

    def test_none_inputs_return_empty(self):
        assert make_pair_id(None, None) == ""  # type: ignore[arg-type]
        assert make_pair_id("KXFOO", None) == ""  # type: ignore[arg-type]

    def test_whitespace_stripped(self):
        pid = make_pair_id("  KXWTI-26APR08  ", "  0xabcdef1234  ")
        assert pid == "kxwti26apr08-0xabcdef12"

    def test_stable_across_calls(self):
        """Same input → same output, always."""
        a = make_pair_id("KXBTC-26FEB06-T85000", "0xdeadbeef12345678")
        b = make_pair_id("KXBTC-26FEB06-T85000", "0xdeadbeef12345678")
        assert a == b

    def test_different_strikes_produce_different_ids(self):
        """The whole point: different markets get different ids."""
        a = make_pair_id("KXBTC-26FEB06-T85000", "0xdeadbeef12345678")
        b = make_pair_id("KXBTC-26FEB06-T90000", "0xdeadbeef12345678")
        assert a != b

    def test_different_poly_ids_produce_different_ids(self):
        a = make_pair_id("KXBTC-26FEB06", "0xdeadbeef")
        b = make_pair_id("KXBTC-26FEB06", "0xcafebabe")
        assert a != b

    def test_uppercase_hex_lowercased(self):
        """0xABCD and 0xabcd should produce the same pair_id."""
        a = make_pair_id("KXFOO", "0xABCDEF1234")
        b = make_pair_id("KXFOO", "0xabcdef1234")
        assert a == b

    def test_result_safe_for_filename(self):
        """Output must not contain shell/filesystem metacharacters."""
        import re
        pid = make_pair_id("KXWTI-26APR08-T107.99", "0x43d5953daec")
        # Only lowercase alphanum, hyphens, and x (for 0x prefix)
        assert re.match(r"^[a-z0-9-]+$", pid) is not None
