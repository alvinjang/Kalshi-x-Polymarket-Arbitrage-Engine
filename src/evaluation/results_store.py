"""Persist evaluation results as JSON so models can be compared later.

Used by ``experiments/run_baselines.py`` (Phase 4) and every downstream
model tier (Phase 5, 6, 7) to write a single per-model JSON file that
``load_all_results`` re-assembles into a comparison table.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path


def _slugify(model_name: str) -> str:
    """Filesystem-safe slug for a human-readable model name."""
    slug = model_name.lower().strip()
    # Replace any run of non-alphanumeric characters with a single underscore.
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    # Collapse leading/trailing underscores.
    slug = slug.strip("_")
    return slug or "unnamed"


def save_results(
    model_name: str,
    metrics: dict,
    output_dir: Path,
    extra: dict | None = None,
) -> Path:
    """Write a single model's evaluation results to JSON.

    Args:
        model_name: Human-readable model name (used as-is inside JSON, and
            slugified to produce the filename).
        metrics: Dict returned by ``BasePredictor.evaluate`` (regression +
            trading metrics).
        output_dir: Directory to write the JSON file into. Created if
            missing.
        extra: Optional extra fields merged into the top-level payload
            (e.g., hyperparameters, dataset hash).

    Returns:
        Path to the saved JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "model_name": model_name,
        "metrics": metrics,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)

    filename = f"{_slugify(model_name)}.json"
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def load_results(path: Path) -> dict:
    """Load a single results JSON file written by ``save_results``."""
    with open(Path(path)) as f:
        return json.load(f)


def load_all_results(results_dir: Path) -> list[dict]:
    """Load every .json file in ``results_dir``, sorted by ``model_name``.

    Args:
        results_dir: Directory containing per-model JSON files.

    Returns:
        List of result dicts sorted alphabetically by the ``model_name``
        field. Empty list if the directory has no ``.json`` files.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    all_results: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        all_results.append(load_results(path))

    all_results.sort(key=lambda r: r.get("model_name", ""))
    return all_results
