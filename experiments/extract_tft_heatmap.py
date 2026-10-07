"""Extract TFT Variable Selection Network (VSN) encoder importance weights.

Retrains TFT with seed=42 (same hyperparameters as run_tft.py) and extracts
VSN weights via interpret_output(). Saves horizontal bar chart to
experiments/figures/tft_variable_importance.png at 300 DPI.

If re-training or VSN extraction fails for any reason, a placeholder figure is
written so the artifact path always exists for downstream plan steps.

Usage:
    cd "/Users/iansabia/Desktop/DS340 Project"
    PYTHONPATH=. .venv/bin/python experiments/extract_tft_heatmap.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

FIGURES_DIR = Path("experiments/figures")
OUT_PATH = FIGURES_DIR / "tft_variable_importance.png"
VSN_JSON_PATH = Path("experiments/results/tft/vsn_importance.json")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _load_data():
    """Load train data with group_id re-attached (mirrors run_tft.py)."""
    from src.features.engineering import compute_derived_features
    from experiments.run_baselines import _build_split

    data_dir = Path("data/processed")
    raw = pd.read_parquet(data_dir / "train.parquet")
    group_ids = raw["group_id"].copy()
    derived = compute_derived_features(raw)
    derived["group_id"] = group_ids.values
    return _build_split(derived)


def _get_feature_cols(train: pd.DataFrame) -> list[str]:
    from experiments.run_baselines import _feature_columns

    feature_cols_all = _feature_columns(train)
    # Remove zero-variance columns (kalshi_kyle_lambda has std=0 in full train)
    nonzero_var_cols = [
        c for c in feature_cols_all if c != "group_id" and train[c].std() > 1e-10
    ]
    return nonzero_var_cols


def _train_tft(train: pd.DataFrame, feature_cols: list[str]):
    """Train a single TFT with seed=42 (same as run_tft.py seed[0])."""
    from src.utils.seed import set_all_seeds
    from src.models.tft import TFTPredictor

    set_all_seeds(42)

    target_col = "spread_change_target"
    X_train = train[feature_cols + ["group_id"]].copy()
    y_train = train[target_col].values

    model = TFTPredictor(
        hidden_size=8,
        attention_head_size=1,
        dropout=0.3,
        hidden_continuous_size=8,
        lstm_layers=1,
        max_encoder_length=6,
        learning_rate=1e-3,
        max_epochs=30,
        patience=5,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def _extract_vsn_weights(model, feature_cols: list[str]) -> tuple[list[str], np.ndarray]:
    """Extract encoder VSN importance weights from a fitted TFTPredictor.

    Returns (feature_names, importance_array) sorted descending.
    """
    import torch

    tft = model._model
    training_dataset = model._training_dataset
    train_loader = training_dataset.to_dataloader(
        train=True, batch_size=64, num_workers=0
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with torch.no_grad():
            raw_out = tft.predict(train_loader, mode="raw", return_x=False)
        interpretation = tft.interpret_output(raw_out, reduction="mean")

    encoder_importance = interpretation["encoder_variables"].detach().cpu().numpy()
    # encoder_variables shape: (n_features,) — one weight per time-varying real
    encoder_feature_names = tft.encoder_variables

    # Sort descending
    order = np.argsort(encoder_importance)[::-1]
    sorted_names = [encoder_feature_names[i] for i in order]
    sorted_weights = encoder_importance[order]

    return sorted_names, sorted_weights


def _save_vsn_json(names: list[str], weights) -> None:
    """Persist VSN weights to JSON for downstream re-plotting without retraining."""
    import json
    VSN_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": list(names),
        "weights": [float(w) for w in weights],
        "note": "TFT Variable Selection Network encoder importance (mean over training set).",
    }
    with VSN_JSON_PATH.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved VSN weights JSON: {VSN_JSON_PATH}  ({len(names)} features)")


def _save_heatmap(names: list[str], weights: np.ndarray, top_n: int = 15) -> None:
    """Plot horizontal bar chart of top-N encoder VSN importances."""
    _save_vsn_json(names, weights)
    if len(names) > top_n:
        names = names[:top_n]
        weights = weights[:top_n]

    # Reverse so highest bar is at top of chart
    names = list(reversed(names))
    weights = list(reversed(weights))

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    bars = ax.barh(range(len(names)), weights, color="#2b6cb0", alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("VSN Encoder Importance Weight (mean over training set)", fontsize=9)
    ax.set_title(
        "TFT Variable Selection Network — Encoder Feature Importances\n"
        "(hidden_size=8, seed=42, N=6,802 training rows — negative result: RMSE=0.3262 vs GRU=0.2928)",
        fontsize=9,
    )
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved VSN heatmap: {OUT_PATH}  (top {len(names)} features)")


def _save_placeholder(reason: str) -> None:
    """Write a placeholder figure when extraction is not possible."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.6,
        "TFT Variable Selection Network",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.45,
        "VSN weights unavailable",
        ha="center",
        va="center",
        fontsize=11,
        color="#c53030",
    )
    ax.text(
        0.5,
        0.32,
        f"Reason: {reason}",
        ha="center",
        va="center",
        fontsize=9,
        color="#555",
        wrap=True,
    )
    ax.text(
        0.5,
        0.18,
        "Note: TFT did not converge at N=6,802 (RMSE=0.3262 vs GRU=0.2928)\n"
        "Attention audit: entropy=2.656, max_weight=0.368 — NOT degenerate",
        ha="center",
        va="center",
        fontsize=8,
        color="#333",
    )
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved placeholder heatmap: {OUT_PATH}")
    print(f"Placeholder reason: {reason}")


def main() -> None:
    print("=== TFT VSN Heatmap Extraction ===")
    print(f"Output: {OUT_PATH}")

    try:
        print("Loading training data...")
        train = _load_data()
        feature_cols = _get_feature_cols(train)
        print(f"Features: {len(feature_cols)} non-zero-variance columns")
        print(f"Target column: spread_change_target  (shape {train.shape})")

        print("Re-training TFT (seed=42, max_epochs=30)...")
        model = _train_tft(train, feature_cols)
        print("Training complete.")

        print("Extracting VSN encoder weights...")
        names, weights = _extract_vsn_weights(model, feature_cols)
        print(f"Extracted {len(names)} feature importances.")
        print("Top-5 features:")
        for n, w in zip(names[:5], weights[:5]):
            print(f"  {n}: {w:.4f}")

        _save_heatmap(names, weights, top_n=15)

    except Exception as exc:
        import traceback as tb

        reason = f"{type(exc).__name__}: {exc}"
        print(f"VSN extraction failed: {reason}", file=sys.stderr)
        tb.print_exc()
        _save_placeholder(reason)


if __name__ == "__main__":
    main()
