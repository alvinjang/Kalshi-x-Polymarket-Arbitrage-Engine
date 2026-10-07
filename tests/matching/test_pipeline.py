"""Tests for the full matching pipeline runner."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from src.matching.run_pipeline import load_metadata, report_scope_gate, SCOPE_GATE_THRESHOLD


class TestLoadMetadata:
    """Tests for load_metadata function."""

    def test_load_metadata_missing_file(self, tmp_path):
        """Should exit when _metadata.json is missing."""
        with pytest.raises(SystemExit):
            load_metadata(tmp_path)

    def test_load_metadata_valid(self, tmp_path):
        """Should return list of dicts from valid _metadata.json."""
        metadata = [
            {"market_id": "K1", "question": "Test?", "category": "Crypto"},
            {"market_id": "K2", "question": "Test2?", "category": "Economics"},
        ]
        (tmp_path / "_metadata.json").write_text(json.dumps(metadata))
        result = load_metadata(tmp_path)
        assert len(result) == 2
        assert result[0]["market_id"] == "K1"


class TestScopeGateReport:
    """Tests for report_scope_gate function."""

    def test_report_scope_gate_pass(self, capsys):
        """Should print PASS when count >= threshold."""
        pairs = [{"status": "accepted"} for _ in range(35)]
        report_scope_gate(pairs)
        captured = capsys.readouterr()
        assert "PASS" in captured.out
        assert "35" in captured.out

    def test_report_scope_gate_warning(self, capsys):
        """Should print WARNING when count < threshold."""
        pairs = [{"status": "accepted"} for _ in range(15)]
        report_scope_gate(pairs)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "15" in captured.out


class TestFullPipeline:
    """Integration test for full pipeline with --skip-curation --skip-enrichment."""

    def test_full_pipeline_skip_all(
        self, tmp_path, sample_kalshi_markets, sample_poly_markets
    ):
        """Pipeline with skip flags should produce output JSON with expected keys."""
        kalshi_dir = tmp_path / "kalshi"
        poly_dir = tmp_path / "polymarket"
        kalshi_dir.mkdir()
        poly_dir.mkdir()

        # Write metadata files
        (kalshi_dir / "_metadata.json").write_text(
            json.dumps(sample_kalshi_markets)
        )
        (poly_dir / "_metadata.json").write_text(
            json.dumps(sample_poly_markets)
        )

        output = tmp_path / "matched_pairs.json"

        # Patch sys.argv for argparse
        test_args = [
            "run_pipeline",
            "--kalshi-dir", str(kalshi_dir),
            "--poly-dir", str(poly_dir),
            "--output", str(output),
            "--skip-curation",
            "--skip-enrichment",
        ]

        with patch.object(sys, "argv", test_args):
            from src.matching.run_pipeline import main
            results = main()

        assert output.exists()
        data = json.loads(output.read_text())
        assert len(data) > 0

        # Check expected keys on first entry
        first = data[0]
        expected_keys = {
            "kalshi_market_id",
            "polymarket_market_id",
            "kalshi_question",
            "polymarket_question",
            "confidence_score",
            "status",
            "pair_id",
            "kalshi_settlement",
            "polymarket_settlement",
            "settlement_aligned",
        }
        assert expected_keys.issubset(set(first.keys())), (
            f"Missing keys: {expected_keys - set(first.keys())}"
        )
        assert first["status"] == "accepted"
