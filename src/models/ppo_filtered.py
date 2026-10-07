"""PPO-Filtered predictor combining RL with autoencoder anomaly detection (MOD-10).

Tier 4 in the complexity-vs-performance analysis.  The autoencoder flags
anomalous spread patterns, and PPO only earns/loses on flagged timestamps.
Non-flagged bars receive a small penalty (``non_flagged_reward = -0.01``)
to discourage the agent from holding during normal periods.

This answers the research question: "Does anomaly pre-filtering help RL?"

Architecture:
  - ``FilteredTradingEnv``: wraps ``SpreadTradingEnv`` with reward masking
  - ``PPOFilteredPredictor(BasePredictor)``: trains SB3 PPO on filtered env,
    predicts via action-to-pseudo-prediction mapping

Exports:
    PPOFilteredPredictor -- Tier 4 RL + anomaly filter predictor
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

from collections import OrderedDict

import gymnasium
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO

from src.models.autoencoder import AnomalyDetectorAutoencoder
from src.models.base import BasePredictor
from src.models.sequence_utils import (
    fit_feature_scaler,
    apply_feature_scaler,
    set_seed,
)
from src.models.trading_env import SpreadTradingEnv


# ---------------------------------------------------------------------------
# Filtered environment wrapper
# ---------------------------------------------------------------------------

class FilteredTradingEnv(gymnasium.Wrapper):
    """Wraps ``SpreadTradingEnv`` to mask reward using anomaly flags.

    On ``step()``, if the current bar is NOT flagged as anomalous,
    the reward is overridden to ``non_flagged_reward`` (default -0.01).
    Flagged bars receive the normal reward from the underlying env.

    This forces PPO to only earn/lose on timestamps the autoencoder
    considers anomalous (unusual spread patterns).

    Args:
        env: A ``SpreadTradingEnv`` instance.
        anomaly_flags_by_pair: Dict mapping ``group_id`` -> boolean array
            of per-bar anomaly flags (``True`` = anomalous).
        non_flagged_reward: Reward override for non-flagged bars.
    """

    def __init__(
        self,
        env: SpreadTradingEnv,
        anomaly_flags_by_pair: dict[int, np.ndarray],
        non_flagged_reward: float = -0.01,
    ) -> None:
        super().__init__(env)
        self._anomaly_flags_by_pair = anomaly_flags_by_pair
        self._non_flagged_reward = non_flagged_reward

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Step with reward masking for non-flagged bars."""
        # Current step index and pair info BEFORE stepping
        current_step = self.env._current_step
        current_pair = self.env._current_pair
        group_id = current_pair["group_id"]

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Check anomaly flag for this bar
        flags = self._anomaly_flags_by_pair.get(group_id)
        if flags is not None and current_step < len(flags):
            if not flags[current_step]:
                reward = self._non_flagged_reward

        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Action-to-prediction mapping
# ---------------------------------------------------------------------------

_ACTION_TO_PREDICTION = {0: 0.0, 1: 0.03, 2: -0.03}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PPOFilteredPredictor(BasePredictor):
    """PPO-Filtered spread predictor (Tier 4 RL + anomaly filter).

    Combines Stable-Baselines3 PPO with an autoencoder anomaly detector.
    During training, reward is masked on non-flagged bars (replaced with
    ``non_flagged_reward``).  During inference, the learned policy runs
    on all rows without filtering.

    Per CONTEXT.md locked decisions:
      - Same PPO config as PPO-Raw
      - Autoencoder binary flag masks reward
      - Non-flagged bars get reward = -0.01
      - Action mapping: hold(0)->0.0, long(1)->+0.03, short(2)->-0.03

    Args:
        anomaly_detector: Pre-trained ``AnomalyDetectorAutoencoder``.
            If ``None``, one is trained internally during ``fit()``.
        total_timesteps: Number of PPO training timesteps.
        policy_kwargs: SB3 policy architecture config.
        learning_rate: PPO learning rate.
        n_steps: Rollout buffer size per update.
        batch_size: Minibatch size for PPO updates.
        n_epochs: PPO optimization epochs per rollout.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_range: PPO clipping range.
        ent_coef: Entropy coefficient for exploration.
        lookback: Number of historical bars in observation window.
        transaction_cost: Per-side cost in reward calculation.
        non_flagged_reward: Reward override for non-flagged bars.
        random_state: Seed for reproducibility.
    """

    _action_to_prediction = _ACTION_TO_PREDICTION

    def __init__(
        self,
        anomaly_detector: AnomalyDetectorAutoencoder | None = None,
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
        non_flagged_reward: float = -0.01,
        random_state: int = 42,
    ) -> None:
        self._anomaly_detector = anomaly_detector
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
        self._non_flagged_reward = non_flagged_reward
        self._random_state = random_state

        self._fitted = False
        self._ppo_model: PPO | None = None
        self._scaler = None
        self._cached_train: dict | None = None
        self._feature_cols: list[str] = []
        self._bool_cols: list[str] = []

    @property
    def name(self) -> str:
        return "PPO-Filtered"

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self, X_train: pd.DataFrame, y_train: np.ndarray
    ) -> "PPOFilteredPredictor":
        """Train PPO on a filtered trading environment.

        Args:
            X_train: Feature DataFrame with ``group_id`` column.
            y_train: Target spread-change array, shape ``(n,)``.

        Returns:
            ``self`` for method chaining.

        Raises:
            ValueError: If ``group_id`` column missing from *X_train*.
        """
        if "group_id" not in X_train.columns:
            raise ValueError(
                "PPOFilteredPredictor.fit requires 'group_id' column in "
                "X_train for pair-boundary-respecting training."
            )

        set_seed(self._random_state)
        y_train = np.asarray(y_train, dtype=float)

        # Determine feature columns and bool columns
        feature_cols = [c for c in X_train.columns if c != "group_id"]
        bool_cols = [c for c in feature_cols if X_train[c].dtype == bool]
        self._feature_cols = feature_cols
        self._bool_cols = bool_cols

        # Train autoencoder if not provided
        if self._anomaly_detector is None:
            self._anomaly_detector = AnomalyDetectorAutoencoder(
                input_dim=len(feature_cols),
                max_epochs=5,
                patience=3,
                random_state=self._random_state,
            )
            self._anomaly_detector.fit(
                X_train[feature_cols], feature_cols
            )

        # Compute anomaly flags for training data
        all_flags = self._anomaly_detector.flag_anomalies(
            X_train[feature_cols]
        )

        # Build per-pair flag arrays
        group_ids = X_train["group_id"].to_numpy()
        seen: OrderedDict[int, None] = OrderedDict()
        for gid in group_ids:
            seen.setdefault(int(gid), None)

        anomaly_flags_by_pair: dict[int, np.ndarray] = {}
        for gid in seen:
            mask = group_ids == gid
            anomaly_flags_by_pair[gid] = all_flags[mask]

        # Fit scaler
        self._scaler = fit_feature_scaler(
            X_train[feature_cols], bool_cols
        )

        # Create the base SpreadTradingEnv
        base_env = SpreadTradingEnv(
            X_train=X_train,
            y_train=y_train,
            feature_cols=feature_cols,
            lookback=self._lookback,
            transaction_cost=self._transaction_cost,
            scaler=self._scaler,
        )

        # Wrap with reward masking
        filtered_env = FilteredTradingEnv(
            env=base_env,
            anomaly_flags_by_pair=anomaly_flags_by_pair,
            non_flagged_reward=self._non_flagged_reward,
        )

        # Train PPO
        self._ppo_model = PPO(
            "MlpPolicy",
            filtered_env,
            learning_rate=self._learning_rate,
            n_steps=self._n_steps,
            batch_size=self._batch_size,
            n_epochs=self._n_epochs,
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
            clip_range=self._clip_range,
            ent_coef=self._ent_coef,
            policy_kwargs=self._policy_kwargs,
            seed=self._random_state,
            verbose=0,
        )
        self._ppo_model.learn(total_timesteps=self._total_timesteps)

        # Cache training data for warm-up stitching during predict()
        X_scaled = apply_feature_scaler(
            X_train[feature_cols], self._scaler, bool_cols
        )
        self._cached_train = {
            "X_scaled": X_scaled,
            "group_ids": group_ids,
            "feature_cols": feature_cols,
            "bool_cols": bool_cols,
        }
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Run the learned policy on all rows (no anomaly filtering at inference).

        Uses warm-up stitching: cached training rows for each group_id
        are prepended so the first test rows get full lookback windows.

        Args:
            X: Feature DataFrame with ``group_id`` column.

        Returns:
            1-D ndarray of pseudo-predictions from the action mapping:
            ``{-0.03, 0.0, +0.03}``, shape ``(len(X),)``.

        Raises:
            RuntimeError: If model has not been fit.
        """
        if not self._fitted:
            raise RuntimeError(
                "PPOFilteredPredictor must be fit before predict"
            )

        feature_cols = self._cached_train["feature_cols"]
        bool_cols = self._cached_train["bool_cols"]
        cached_X_scaled = self._cached_train["X_scaled"]
        cached_group_ids = self._cached_train["group_ids"]

        # Scale test features using train-fitted scaler
        X_scaled = apply_feature_scaler(X[feature_cols], self._scaler, bool_cols)
        group_ids_test = X["group_id"].to_numpy()
        n_test = len(X)

        # Process by group for warm-up stitching
        seen_gids: list[int] = []
        gid_test_indices: dict[int, list[int]] = {}
        for i, gid in enumerate(group_ids_test):
            gid = int(gid)
            if gid not in gid_test_indices:
                seen_gids.append(gid)
                gid_test_indices[gid] = []
            gid_test_indices[gid].append(i)

        predictions = np.zeros(n_test, dtype=float)
        n_channels = len(feature_cols) + 2  # features + position + time_fraction

        for gid in seen_gids:
            test_indices = gid_test_indices[gid]
            test_rows = X_scaled[test_indices]

            # Warm-up stitching: prepend cached train rows
            train_mask = cached_group_ids == gid
            if train_mask.any():
                train_rows = cached_X_scaled[train_mask]
                stitched = np.vstack([train_rows, test_rows])
            else:
                stitched = test_rows

            n_available = len(stitched)

            # Pad if needed
            if n_available < self._lookback:
                n_pad = self._lookback - n_available
                pad_rows = np.tile(stitched[0:1], (n_pad, 1))
                stitched = np.vstack([pad_rows, stitched])

            test_start = len(stitched) - len(test_rows)

            # Run the policy deterministically on each step
            position = 0
            for local_idx, global_idx in enumerate(test_indices):
                end_pos = test_start + local_idx + 1
                start_pos = max(0, end_pos - self._lookback)
                window = stitched[start_pos:end_pos].astype(np.float32)

                # Pad window if shorter than lookback
                if len(window) < self._lookback:
                    n_pad = self._lookback - len(window)
                    pad = np.tile(window[0:1], (n_pad, 1))
                    window = np.vstack([pad, window])

                # Augment: position + time_fraction
                n_rows = window.shape[0]
                total_steps = len(stitched)
                current_step = test_start + local_idx
                time_fraction = current_step / max(total_steps - 1, 1)

                position_col = np.full(
                    (n_rows, 1), position, dtype=np.float32
                )
                time_col = np.full(
                    (n_rows, 1), time_fraction, dtype=np.float32
                )
                augmented = np.concatenate(
                    [window, position_col, time_col], axis=1
                )
                obs = augmented.flatten().astype(np.float32)

                action, _ = self._ppo_model.predict(
                    obs, deterministic=True
                )
                action = int(action)
                predictions[global_idx] = _ACTION_TO_PREDICTION[action]

                # Update position for next observation
                _action_to_position = {0: 0, 1: 1, 2: -1}
                position = _action_to_position[action]

        return predictions
