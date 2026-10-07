"""Temporal Fusion Transformer predictor for spread-change prediction (Tier 2).

Wraps ``pytorch_forecasting.TemporalFusionTransformer`` behind the
``BasePredictor`` interface so TFT plugs directly into the shared evaluation
pipeline (regression metrics + profit simulation).

Hyperparameters are LOCKED per REQUIREMENTS.md TFT-02 and CONTEXT.md.
Do NOT tune during implementation.

**Spec deviation — GroupNormalizer transformation:**
``GroupNormalizer`` uses ``transformation=None`` (NOT ``transformation='softplus'``).
REQUIREMENTS.md TFT-02 originally said 'softplus' but research (11-RESEARCH.md
Pitfall 1) confirmed softplus maps negative spread-change targets near zero,
causing degenerate P&L predictions. ``transformation=None`` applies standard
z-normalization per pair, which is correct for signed spread-change targets.

Training protocol:
- 3 quantiles: QuantileLoss(quantiles=[0.1, 0.5, 0.9]) for 2x speedup over
  the default 7-quantile loss.
- Point predictions extracted via ``mode='prediction'`` which calls
  QuantileLoss.to_prediction() and returns the 0.5 quantile.
- accelerator='cpu' — MPS has a segfault bug (Phase 5 decisions).
- EarlyStopping from lightning.pytorch.callbacks (NOT sequence_utils or pytorch_lightning).
  IMPORTANT: pytorch_forecasting 1.7.0 inherits from lightning.pytorch, NOT pytorch_lightning.
  Using pytorch_lightning.Trainer raises TypeError; use lightning.pytorch.Trainer instead.

Exports:
    TFTPredictor -- Tier 2 Temporal Fusion Transformer for spread-change prediction
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch

import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping

from src.models.base import BasePredictor
from src.models.sequence_utils import (
    set_seed,
    fit_feature_scaler,
    apply_feature_scaler,
)


class TFTPredictor(BasePredictor):
    """TFT-based spread-change predictor (Tier 2 transformer baseline).

    Hyperparameters default to the values locked in REQUIREMENTS.md TFT-02:
      - hidden_size=8, attention_head_size=1, dropout=0.3
      - hidden_continuous_size=8, lstm_layers=1
      - max_encoder_length=6, max_prediction_length=1
      - learning_rate=1e-3
      - max_epochs=30, patience=5

    ``fit()`` and ``predict()`` require a ``group_id`` column in ``X``
    for pair-level normalization and time-series dataset construction.
    A ``ValueError`` is raised immediately if the column is missing.

    **GroupNormalizer deviation:** Uses ``transformation=None`` (standard
    z-normalization per pair) instead of 'softplus'. Spread changes are signed
    — softplus warps negative targets near zero, causing degenerate predictions.
    """

    def __init__(
        self,
        hidden_size: int = 8,
        attention_head_size: int = 1,
        dropout: float = 0.3,
        hidden_continuous_size: int = 8,
        lstm_layers: int = 1,
        max_encoder_length: int = 6,
        learning_rate: float = 1e-3,
        max_epochs: int = 30,
        patience: int = 5,
        random_state: int = 42,
    ) -> None:
        self._hidden_size = hidden_size
        self._attention_head_size = attention_head_size
        self._dropout = dropout
        self._hidden_continuous_size = hidden_continuous_size
        self._lstm_layers = lstm_layers
        self._max_encoder_length = max_encoder_length
        self._learning_rate = learning_rate
        self._max_epochs = max_epochs
        self._patience = patience
        self._random_state = random_state

        self._fitted = False
        self._model: TemporalFusionTransformer | None = None
        self._training_dataset: TimeSeriesDataSet | None = None
        self._scaler = None
        self._cached_train: dict | None = None

    @property
    def name(self) -> str:
        return "TFT"

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> "TFTPredictor":
        """Train the TFT on ``X_train`` / ``y_train``.

        Args:
            X_train: Feature DataFrame. **Must** contain a ``group_id``
                column for per-pair TimeSeriesDataSet construction.
            y_train: Target array of spread changes, shape ``(n,)``.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If ``group_id`` column is missing from *X_train*.
        """
        # --- group_id guard (MUST be first step) ---
        if "group_id" not in X_train.columns:
            raise ValueError(
                "TFTPredictor.fit requires 'group_id' column in X_train for "
                "per-pair TimeSeriesDataSet construction. "
                "Pass df[feature_cols + ['group_id']]."
            )

        set_seed(self._random_state)
        torch.set_num_threads(1)  # Apple Silicon workaround

        y_train = np.asarray(y_train, dtype=float)
        feature_cols = [c for c in X_train.columns if c != "group_id"]
        bool_cols = [c for c in feature_cols if X_train[c].dtype == bool]

        # Scale features (same helpers as GRU/LSTM)
        self._scaler = fit_feature_scaler(X_train[feature_cols], bool_cols)
        scaled_vals = apply_feature_scaler(X_train[feature_cols], self._scaler, bool_cols)

        # Build long-format DataFrame for TimeSeriesDataSet
        df = X_train[["group_id"]].copy()
        for i, col in enumerate(feature_cols):
            df[col] = scaled_vals[:, i]
        df["target"] = y_train
        # Per-group monotonic time_idx (Pitfall 2: NOT global time_idx)
        df["time_idx"] = df.groupby("group_id").cumcount()
        # Ensure group_id is a string category for TimeSeriesDataSet
        df["group_id"] = df["group_id"].astype(str)
        df = df.reset_index(drop=True)

        # Build training TimeSeriesDataSet
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._training_dataset = TimeSeriesDataSet(
                data=df,
                time_idx="time_idx",
                target="target",
                group_ids=["group_id"],
                max_encoder_length=self._max_encoder_length,
                max_prediction_length=1,
                static_categoricals=["group_id"],
                time_varying_unknown_reals=feature_cols,
                target_normalizer=GroupNormalizer(
                    groups=["group_id"],
                    transformation=None,  # DEVIATION: NOT 'softplus' — spread changes are signed
                ),
                allow_missing_timesteps=True,
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True,
            )

        # Validation split: last 10% of rows (chronological)
        val_cutoff = int(len(df) * 0.9)
        df_val = df.iloc[val_cutoff:].copy()

        # Need enough rows in val for at least one sequence
        if len(df_val) < self._max_encoder_length:
            # Fall back: replicate the training dataset as val (not ideal but avoids crash)
            df_val = df.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val_dataset = TimeSeriesDataSet.from_dataset(
                self._training_dataset,
                df_val,
                predict=False,
                stop_randomization=True,
            )

        train_loader = self._training_dataset.to_dataloader(
            train=True, batch_size=64, num_workers=0
        )
        val_loader = val_dataset.to_dataloader(
            train=False, batch_size=64, num_workers=0
        )

        # Instantiate model via from_dataset() factory (NOT direct __init__)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = TemporalFusionTransformer.from_dataset(
                self._training_dataset,
                hidden_size=self._hidden_size,
                attention_head_size=self._attention_head_size,
                dropout=self._dropout,
                hidden_continuous_size=self._hidden_continuous_size,
                lstm_layers=self._lstm_layers,
                loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
                learning_rate=self._learning_rate,
                log_interval=-1,
                log_val_interval=-1,
            )

        # Train with Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=self._max_epochs,
            gradient_clip_val=0.1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=self._patience,
                    mode="min",
                    verbose=False,
                )
            ],
            logger=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer.fit(self._model, train_loader, val_loader)

        # Cache training data for warm-up stitching during predict()
        self._cached_train = {
            "df": df,               # long-format with time_idx and scaled features
            "feature_cols": feature_cols,
            "bool_cols": bool_cols,
        }
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict spread changes for every row in ``X``.

        Uses warm-up stitching: cached training rows are prepended so the
        first test rows also get full-length encoder windows. Derives
        test time_idx by continuing from the end of train per group.

        Args:
            X: Feature DataFrame. **Must** contain a ``group_id`` column.

        Returns:
            1-D ndarray of predictions, shape ``(len(X),)``.

        Raises:
            RuntimeError: If model has not been fit.
            ValueError: If ``group_id`` column is missing.
        """
        if not self._fitted:
            raise RuntimeError(
                "TFTPredictor: must call fit() before predict()"
            )
        if "group_id" not in X.columns:
            raise ValueError(
                "TFTPredictor.predict requires 'group_id' column in X for "
                "per-pair TimeSeriesDataSet construction."
            )

        feature_cols = self._cached_train["feature_cols"]
        bool_cols = self._cached_train["bool_cols"]
        train_df = self._cached_train["df"]

        # Scale test features using the TRAIN-fitted scaler (no re-fit)
        X_scaled = X[["group_id"]].copy()
        X_scaled["group_id"] = X_scaled["group_id"].astype(str)
        scaled_vals = apply_feature_scaler(X[feature_cols], self._scaler, bool_cols)
        for i, col in enumerate(feature_cols):
            X_scaled[col] = scaled_vals[:, i]

        # Derive time_idx for test rows (continuing from end of train per group)
        max_train_idx = train_df.groupby("group_id")["time_idx"].max()
        X_scaled["time_idx"] = X_scaled.groupby("group_id").cumcount()
        for gid in X_scaled["group_id"].unique():
            gid_str = str(gid)
            if gid_str in max_train_idx.index:
                offset = int(max_train_idx[gid_str]) + 1
            else:
                offset = 0
            mask = X_scaled["group_id"] == gid_str
            X_scaled.loc[mask, "time_idx"] = (
                X_scaled.loc[mask, "time_idx"] + offset
            )

        # Placeholder target (overwritten during prediction)
        X_scaled["target"] = 0.0
        X_scaled = X_scaled.reset_index(drop=True)

        # Round-based batch prediction: process all groups simultaneously.
        #
        # TFT predict=True returns 1 prediction per group (the last encoder
        # window). To get one prediction per test row, we do K rounds where
        # K = max number of test rows across groups (~11). In round k, we
        # stitch [train_g, test_rows_g[0..k]] for EACH group and combine
        # into ONE multi-group dataset, calling predict once per round.
        #
        # This gives O(K) dataset builds instead of O(n_test) = O(1673),
        # making prediction ~150x faster.

        n_test = len(X_scaled)
        preds_by_original_idx: dict[int, float] = {}

        X_scaled_with_idx = X_scaled.copy()
        X_scaled_with_idx["_orig_idx"] = np.arange(n_test)

        # Pre-sort test rows by group for efficient slicing
        groups_data: dict[str, tuple[list[int], pd.DataFrame, pd.DataFrame]] = {}
        for gid in X_scaled["group_id"].unique():
            test_mask = X_scaled_with_idx["group_id"] == gid
            test_rows_g = X_scaled_with_idx[test_mask].copy().reset_index(drop=True)
            orig_indices = test_rows_g["_orig_idx"].tolist()

            train_mask = train_df["group_id"] == gid
            train_rows_g = train_df[train_mask].copy()

            groups_data[gid] = (orig_indices, train_rows_g, test_rows_g)

        # Max test rows across all groups
        max_k = max(len(v[0]) for v in groups_data.values())

        for k in range(max_k):
            # Groups that have a k-th test row
            active_groups = {
                gid: data for gid, data in groups_data.items()
                if k < len(data[0])
            }
            if not active_groups:
                break

            # Build combined stitched dataset for all active groups
            combined_parts = []
            gid_to_orig_idx: dict[str, int] = {}

            for gid, (orig_indices, train_rows_g, test_rows_g) in active_groups.items():
                # Stitch train + test[0..k] for this group
                test_subset = test_rows_g.iloc[: k + 1].drop(
                    columns=["_orig_idx"]
                )
                stitched_g = pd.concat(
                    [train_rows_g, test_subset], ignore_index=True
                )
                combined_parts.append(stitched_g)
                gid_to_orig_idx[gid] = orig_indices[k]

            combined_df = pd.concat(combined_parts, ignore_index=True)

            # Build prediction dataset for all active groups at once
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pred_dataset_k = TimeSeriesDataSet.from_dataset(
                        self._training_dataset,
                        combined_df,
                        predict=True,
                        stop_randomization=True,
                    )
                pred_loader_k = pred_dataset_k.to_dataloader(
                    train=False, batch_size=64, num_workers=0
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_preds_k = self._model.predict(
                        pred_loader_k, mode="prediction"
                    )
                preds_k = raw_preds_k.numpy().astype(float).ravel()

                # Map predictions back to original indices
                # pred_dataset returns predictions in group order (alphabetical)
                active_gids_sorted = sorted(active_groups.keys())
                for i, gid in enumerate(active_gids_sorted):
                    if i < len(preds_k):
                        preds_by_original_idx[gid_to_orig_idx[gid]] = preds_k[i]
                    else:
                        preds_by_original_idx[gid_to_orig_idx[gid]] = 0.0

            except Exception:
                # Fallback: predict 0.0 for all active groups in this round
                for gid, orig_idx in gid_to_orig_idx.items():
                    preds_by_original_idx[orig_idx] = 0.0

        # Reconstruct in original row order
        output = np.array(
            [preds_by_original_idx.get(i, 0.0) for i in range(n_test)],
            dtype=float,
        )
        return output

    # ------------------------------------------------------------------
    # Attention audit (TFT-05)
    # ------------------------------------------------------------------

    def _audit_attention(self) -> dict:
        """Compute attention entropy audit after fit().

        Runs a forward pass over the training data to extract Variable
        Selection Network (VSN) encoder weights, then computes Shannon
        entropy. Flags as degenerate if entropy < 0.5*log(n_features)
        or if any single variable weight > 0.8.

        Returns:
            Dict with keys: entropy, max_variable_weight,
                            threshold_entropy, is_degenerate.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._fitted:
            raise RuntimeError(
                "TFTPredictor: must call fit() before _audit_attention()"
            )

        train_loader = self._training_dataset.to_dataloader(
            train=False, batch_size=64, num_workers=0
        )
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_predictions = self._model.predict(
                train_loader, mode="raw", return_x=False
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interpretation = self._model.interpret_output(
                raw_predictions, reduction="mean"
            )

        enc_vars = interpretation["encoder_variables"].detach().cpu().numpy()
        probs = enc_vars / enc_vars.sum()
        probs_nz = probs[probs > 0]
        entropy = float(-np.sum(probs_nz * np.log(probs_nz)))
        max_weight = float(probs.max())
        n_features = len(enc_vars)
        threshold = 0.5 * np.log(n_features)
        is_degenerate = (entropy < threshold) or (max_weight > 0.8)

        if is_degenerate:
            print(
                f"[TFT] WARN: attention collapse — entropy={entropy:.3f} "
                f"(threshold={threshold:.3f}), max_weight={max_weight:.3f}"
            )
        return {
            "entropy": entropy,
            "max_variable_weight": max_weight,
            "threshold_entropy": float(threshold),
            "is_degenerate": is_degenerate,
        }
