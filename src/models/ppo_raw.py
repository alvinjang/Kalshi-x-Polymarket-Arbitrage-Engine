"""PPO-Raw predictor for spread-change trading (Tier 3 RL).

Wraps a Stable-Baselines3 PPO agent that trains directly on raw
microstructure features via ``SpreadTradingEnv``.  Inherits from
``BasePredictor`` so it plugs into the shared evaluation pipeline
(regression metrics + profit simulation).

This is the "Tier 3" model answering KG's question: does RL acting
directly on features add value over regression baselines?  Expected
outcome: 75% likely converges to "always hold" (0 trades), which is a
valid finding that directly answers the research question.

Hyperparameters are locked per CONTEXT.md decisions (PPO-Raw section).
Discrete actions are mapped to pseudo-predictions for evaluation
compatibility: hold(0)->0.0, long(1)->+0.03, short(2)->-0.03.

Exports:
    PPORawPredictor -- Tier 3 RL baseline for spread-change prediction
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.models.base import BasePredictor
from src.models.sequence_utils import (
    set_seed,
    fit_feature_scaler,
    apply_feature_scaler,
)
from src.models.trading_env import SpreadTradingEnv


# Action-to-prediction mapping (locked in CONTEXT.md):
#   hold (0) -> 0.0
#   long (1) -> +0.03  (above profit_sim threshold of 0.02)
#   short (2) -> -0.03
_ACTION_TO_PREDICTION = {0: 0.0, 1: 0.03, 2: -0.03}


class PPORawPredictor(BasePredictor):
    """PPO-based spread-change predictor acting on raw features (Tier 3).

    Trains an SB3 PPO agent on ``SpreadTradingEnv`` with locked
    hyperparameters.  At predict time, discrete actions are converted
    to pseudo-predictions (coarse 3-value output) so the model can be
    evaluated with the same regression + profit metrics as all tiers.

    Constructor Args:
        total_timesteps: Total environment steps for PPO training.
        policy_kwargs: Dict passed to SB3 ``MlpPolicy`` (net_arch).
        learning_rate: PPO optimizer learning rate.
        n_steps: Rollout buffer length per update.
        batch_size: Minibatch size for PPO updates.
        n_epochs: PPO epochs per update.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_range: PPO clipping range.
        ent_coef: Entropy bonus coefficient.
        lookback: Number of historical bars in the observation window.
        transaction_cost: Per-side cost used in the trading env reward.
        random_state: Seed for reproducibility.
    """

    # Expose at class level for direct testing
    _action_to_prediction = _ACTION_TO_PREDICTION

    def __init__(
        self,
        total_timesteps: int = 100_000,
        policy_kwargs: dict | None = None,
        learning_rate: float = 3e-4,
        n_steps: int = 256,
        batch_size: int = 64,
        n_epochs: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.005,
        lookback: int = 6,
        transaction_cost: float = 0.02,
        random_state: int = 42,
    ) -> None:
        self._total_timesteps = total_timesteps
        self._policy_kwargs = policy_kwargs or dict(net_arch=[64, 64])
        self._learning_rate = learning_rate
        self._n_steps = n_steps
        self._batch_size = batch_size
        self._n_epochs = n_epochs
        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._clip_range = clip_range
        self._ent_coef = ent_coef
        self._lookback = lookback
        self._transaction_cost = transaction_cost
        self._random_state = random_state

        self._fitted = False
        self._model: PPO | None = None
        self._scaler = None
        self._feature_cols: list[str] = []
        self._bool_cols: list[str] = []
        self._cached_train: dict | None = None

    @property
    def name(self) -> str:
        return "PPO-Raw"

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> "PPORawPredictor":
        """Train the PPO agent on spread trading episodes.

        Args:
            X_train: Feature DataFrame.  **Must** contain a ``group_id``
                column for pair-boundary episode construction.
            y_train: Target array of spread changes, shape ``(n,)``.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If ``group_id`` column is missing from *X_train*.
        """
        if "group_id" not in X_train.columns:
            raise ValueError(
                "PPORawPredictor.fit requires 'group_id' column in X_train "
                "for pair-boundary episode construction. "
                "Pass df[feature_cols + ['group_id']]."
            )

        set_seed(self._random_state)

        y_train = np.asarray(y_train, dtype=float)

        # Determine feature columns (everything except group_id)
        self._feature_cols = [c for c in X_train.columns if c != "group_id"]
        self._bool_cols = [
            c for c in self._feature_cols if X_train[c].dtype == bool
        ]

        # Fit scaler on training features
        self._scaler = fit_feature_scaler(
            X_train[self._feature_cols], self._bool_cols
        )

        # Create the trading environment
        env = SpreadTradingEnv(
            X_train=X_train,
            y_train=y_train,
            feature_cols=self._feature_cols,
            lookback=self._lookback,
            transaction_cost=self._transaction_cost,
            scaler=self._scaler,
        )

        # Create and train the SB3 PPO model
        self._model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=self._policy_kwargs,
            learning_rate=self._learning_rate,
            n_steps=self._n_steps,
            batch_size=self._batch_size,
            n_epochs=self._n_epochs,
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
            clip_range=self._clip_range,
            ent_coef=self._ent_coef,
            seed=self._random_state,
            verbose=0,
        )
        self._model.learn(total_timesteps=self._total_timesteps)

        # Cache training data for warm-up stitching during predict()
        X_scaled = apply_feature_scaler(
            X_train[self._feature_cols], self._scaler, self._bool_cols
        )
        group_ids = X_train["group_id"].to_numpy()
        self._cached_train = {
            "X_scaled": X_scaled,
            "group_ids": group_ids,
        }

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict spread changes for every row in ``X``.

        For each row, builds a lookback observation window (using warm-up
        stitching from cached train data when available), feeds it to the
        trained PPO policy, and maps the discrete action to a
        pseudo-prediction: hold->0.0, long->+0.03, short->-0.03.

        Args:
            X: Feature DataFrame.  **Must** contain a ``group_id``
                column.

        Returns:
            1-D ndarray of predictions, shape ``(len(X),)``.

        Raises:
            RuntimeError: If model has not been fit.
            ValueError: If ``group_id`` column is missing.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fit before predict")

        if "group_id" not in X.columns:
            raise ValueError(
                "PPORawPredictor.predict requires 'group_id' column in X "
                "for pair-boundary-respecting windowing."
            )

        # Scale test features using train-fitted scaler
        X_scaled = apply_feature_scaler(
            X[self._feature_cols], self._scaler, self._bool_cols
        )
        group_ids_test = X["group_id"].to_numpy()
        n_test = len(X)

        cached_X_scaled = self._cached_train["X_scaled"]
        cached_group_ids = self._cached_train["group_ids"]

        n_features = len(self._feature_cols)
        n_channels = n_features + 2  # features + position + time_fraction

        # Process by group, preserving original row order
        seen_gids: list[int] = []
        gid_test_indices: dict[int, list[int]] = {}
        for i, gid in enumerate(group_ids_test):
            gid = int(gid)
            if gid not in gid_test_indices:
                seen_gids.append(gid)
                gid_test_indices[gid] = []
            gid_test_indices[gid].append(i)

        predictions = np.zeros(n_test, dtype=float)

        for gid in seen_gids:
            test_indices = gid_test_indices[gid]
            test_rows = X_scaled[test_indices]  # (n_test_group, n_features)

            # Warm-up stitching: prepend cached train rows for this group
            train_mask = cached_group_ids == gid
            if train_mask.any():
                train_rows = cached_X_scaled[train_mask]
                stitched = np.vstack([train_rows, test_rows])
            else:
                stitched = test_rows

            n_available = len(stitched)

            # Pad if needed (repeat first row)
            if n_available < self._lookback:
                n_pad = self._lookback - n_available
                pad_rows = np.tile(stitched[0:1], (n_pad, 1))
                stitched = np.vstack([pad_rows, stitched])

            test_start_in_stitched = len(stitched) - len(test_rows)

            # Track position per pair for observation augmentation
            current_position = 0

            for local_idx, global_idx in enumerate(test_indices):
                end_pos = test_start_in_stitched + local_idx + 1
                start_pos = end_pos - self._lookback
                if start_pos < 0:
                    start_pos = 0
                window = stitched[start_pos:end_pos]

                # Safety pad if window shorter than lookback
                if len(window) < self._lookback:
                    n_pad = self._lookback - len(window)
                    pad = np.tile(window[0:1], (n_pad, 1))
                    window = np.vstack([pad, window])

                # Augment with position and time_fraction (match env obs)
                n_rows = window.shape[0]
                # Use total stitched length as reference for time fraction
                total_bars = len(stitched)
                time_fraction = (
                    (test_start_in_stitched + local_idx)
                    / max(total_bars - 1, 1)
                )
                position_col = np.full(
                    (n_rows, 1), current_position, dtype=np.float32
                )
                time_col = np.full(
                    (n_rows, 1), time_fraction, dtype=np.float32
                )
                augmented = np.concatenate(
                    [window, position_col, time_col], axis=1
                )  # (lookback, 33)

                obs = augmented.flatten().astype(np.float32)

                # Get deterministic action from trained policy
                action, _ = self._model.predict(obs, deterministic=True)
                action = int(action)

                # Map action to pseudo-prediction
                pred_val = _ACTION_TO_PREDICTION[action]
                predictions[global_idx] = pred_val

                # Update position for next step (mirrors env logic)
                # action: 0=hold(0), 1=long(+1), 2=short(-1)
                _action_to_position = {0: 0, 1: 1, 2: -1}
                current_position = _action_to_position[action]

        return predictions
