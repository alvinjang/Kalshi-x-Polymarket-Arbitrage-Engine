"""Automated test: TFT (MOD-07) deferral rationale is documented per CONTEXT.md D9.

Covers Validation Dimension 7 from 05-RESEARCH.md.
"""
from pathlib import Path

DEFERRALS_PATH = Path(".planning/phases/05-time-series-models/05-DEFERRALS.md")
SUMMARY_PATH = Path(".planning/phases/05-time-series-models/05-SUMMARY.md")


def _read(path: Path) -> str:
    assert path.exists(), f"{path} does not exist"
    return path.read_text()


def test_deferrals_file_exists():
    assert DEFERRALS_PATH.exists()


def test_deferrals_file_contains_mod07():
    text = _read(DEFERRALS_PATH)
    assert "MOD-07" in text or "TFT" in text


def test_deferrals_file_mentions_roadmap_clause():
    text = _read(DEFERRALS_PATH).lower()
    assert "roadmap" in text
    assert "success criterion" in text or "deferral clause" in text


def test_deferrals_file_contains_param_to_sample_argument():
    text = _read(DEFERRALS_PATH).lower()
    assert "param-to-sample ratio" in text


def test_deferrals_file_mentions_overfitting_argument():
    text = _read(DEFERRALS_PATH).lower()
    assert any(k in text for k in ("overfit", "overfitting", "transformer"))


def test_deferrals_file_mentions_gru_and_lstm_alternative():
    text = _read(DEFERRALS_PATH)
    assert "GRU" in text and "LSTM" in text


def test_deferrals_file_mentions_timeline_preservation():
    text = _read(DEFERRALS_PATH)
    assert "Phase 6" in text or "Phase 7" in text


def test_summary_file_exists():
    assert SUMMARY_PATH.exists()


def test_summary_file_references_deferrals():
    text = _read(SUMMARY_PATH)
    assert "05-DEFERRALS.md" in text or ("TFT" in text and "deferred" in text.lower())
