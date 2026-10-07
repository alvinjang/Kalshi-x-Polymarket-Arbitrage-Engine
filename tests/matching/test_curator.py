"""Tests for CLI-based curation interface."""
import copy
import json

import pytest

from src.matching.curator import review_candidates


class TestReviewCandidates:
    """Tests for review_candidates CLI interface."""

    def test_accept_sets_status(self, monkeypatch, tmp_path, sample_scored_candidates):
        """Pressing 'a' should set status to 'accepted'."""
        candidates = [copy.deepcopy(sample_scored_candidates[0])]
        output = tmp_path / "pairs.json"

        inputs = iter(["a", ""])  # accept, empty notes
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = review_candidates(candidates, output)
        assert len(result) == 1
        assert result[0]["status"] == "accepted"

    def test_reject_sets_status(self, monkeypatch, tmp_path, sample_scored_candidates):
        """Pressing 'r' should set status to 'rejected'."""
        candidates = [copy.deepcopy(sample_scored_candidates[0])]
        output = tmp_path / "pairs.json"

        inputs = iter(["r"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = review_candidates(candidates, output)
        assert len(result) == 1
        assert result[0]["status"] == "rejected"

    def test_flag_sets_status(self, monkeypatch, tmp_path, sample_scored_candidates):
        """Pressing 'f' should set status to 'flagged' with review notes."""
        candidates = [copy.deepcopy(sample_scored_candidates[0])]
        output = tmp_path / "pairs.json"

        inputs = iter(["f", "needs review"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = review_candidates(candidates, output)
        assert len(result) == 1
        assert result[0]["status"] == "flagged"
        assert result[0]["review_notes"] == "needs review"

    def test_auto_saves_after_each(self, monkeypatch, tmp_path, sample_scored_candidates):
        """After each decision, file should be written with current results."""
        candidates = copy.deepcopy(sample_scored_candidates[:2])
        output = tmp_path / "pairs.json"

        # Accept both: a, notes, a, notes
        inputs = iter(["a", "", "a", ""])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = review_candidates(candidates, output)
        assert len(result) == 2

        # Verify file was saved with both entries
        data = json.loads(output.read_text())
        assert len(data) == 2

    def test_quit_stops_early(self, monkeypatch, tmp_path, sample_scored_candidates):
        """Pressing 'q' should stop and return only completed reviews."""
        candidates = copy.deepcopy(sample_scored_candidates[:3])
        output = tmp_path / "pairs.json"

        # Accept first, then quit
        inputs = iter(["a", "", "q"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = review_candidates(candidates, output)
        assert len(result) == 1
        assert result[0]["status"] == "accepted"
