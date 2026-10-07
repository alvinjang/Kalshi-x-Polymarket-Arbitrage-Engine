"""Custom Gymnasium trading environment for PPO spread-arbitrage agents.

Provides ``SpreadTradingEnv``, a Gymnasium environment where an RL agent
trades the spread between Kalshi and Polymarket on matched prediction-market
pairs.  Both PPO-Raw (MOD-08) and PPO-Filtered (MOD-10) train on this env.

State space: flattened ``(198,)`` = 6 bars x 33 channels
    (31 scaled features + current_position + time_fraction).
Action space: ``Discrete(3)`` -- {0: hold, 1: long, 2: short}.
Reward: ``clip(position * actual_spread_change - 0.02 * |position_change|, -0.5, 0.5)``.

Episodes iterate over pairs; pairs with fewer than ``lookback`` bars are
skipped.

Exports:
    SpreadTradingEnv -- Gymnasium environment for spread trading
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

from collections import OrderedDict

import gymnasium
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler

from src.models.sequence_utils import fit_feature_scaler, apply_feature_scaler


# Action mapping: action_id -> position value
_ACTION_TO_POSITION = {0: 0, 1: 1, 2: -1}


class SpreadTradingEnv(gymnasium.Env):
    """Gymnasium environment for per-pair spread-change trading.

    The agent observes a rolling window of scaled features augmented with
    the current position and time fraction, then selects hold / long / short.
    Reward is the P&L from the position times the actual spread change,
    penalised by a transaction cost proportional to position change, and
    clipped to ``[-reward_clip, +reward_clip]``.

    Constructor Args:
        X_train: DataFrame with ``group_id`` column plus feature columns.
        y_train: 1-D array of spread-change targets aligned with ``X_train``.
        feature_cols: List of 31 feature column names (excludes ``group_id``).
        lookback: Number of historical bars in the observation window.
        transaction_cost: Per-side cost applied to ``|position_change|``.
        reward_clip: Symmetric clip bound for per-step reward.
        scaler: Pre-fitted ``StandardScaler``.  If ``None``, one is fitted
            on ``X_train[feature_cols]`` internally.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        feature_cols: list[str],
        lookback: int = 6,
        transaction_cost: float = 0.02,
        reward_clip: float = 0.5,
        scaler: StandardScaler | None = None,
    ) -> None:
        super().__init__()

        self._lookback = lookback
        self._transaction_cost = transaction_cost
        self._reward_clip = reward_clip
        self._feature_cols = list(feature_cols)
        n_channels = len(feature_cols) + 2  # features + position + time_fraction

        # --- Spaces ---
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback * n_channels,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # --- Fit / apply scaler ---
        bool_cols = [
            c for c in feature_cols if X_train[c].dtype == bool
        ]
        if scaler is None:
            self._scaler = fit_feature_scaler(X_train[feature_cols], bool_cols)
        else:
            self._scaler = scaler
        X_scaled = apply_feature_scaler(X_train[feature_cols], self._scaler, bool_cols)

        y_train = np.asarray(y_train, dtype=float)

        # --- Group data by pair (group_id) ---
        group_ids = X_train["group_id"].to_numpy()
        # Preserve first-occurrence order
        seen: OrderedDict[int, None] = OrderedDict()
        for gid in group_ids:
            seen.setdefault(int(gid), None)

        self._pairs: list[dict] = []
        for gid in seen:
            mask = group_ids == gid
            features = X_scaled[mask]  # (n_bars, 31)
            targets = y_train[mask]    # (n_bars,)
            n_bars = len(features)
            if n_bars < lookback:
                continue  # skip short pairs
            self._pairs.append({
                "features": features.astype(np.float32),
                "targets": targets.astype(np.float32),
                "n_bars": n_bars,
                "group_id": gid,
            })

        if len(self._pairs) == 0:
            raise ValueError(
                f"No pairs have >= {lookback} bars. Cannot create environment."
            )

        # --- Episode state ---
        self._pair_idx: int = -1  # incremented on first reset
        self._current_step: int = 0
        self.current_position: int = 0
        self._current_pair: dict | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Load the next pair and return the initial observation.

        Cycles through pairs in order, wrapping around at the end.

        Returns:
            ``(observation, info)`` where observation has shape ``(198,)``
            and info contains ``{"group_id": int, "pair_length": int}``.
        """
        super().reset(seed=seed)

        # Advance to next pair (cycle)
        self._pair_idx = (self._pair_idx + 1) % len(self._pairs)
        self._current_pair = self._pairs[self._pair_idx]

        # First valid step: after accumulating lookback window
        self._current_step = self._lookback - 1
        self.current_position = 0

        obs = self._get_observation()
        info = {
            "group_id": self._current_pair["group_id"],
            "pair_length": self._current_pair["n_bars"],
        }
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one trading step.

        Args:
            action: 0 = hold, 1 = long (+1), 2 = short (-1).

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
        """
        assert self._current_pair is not None, "Must call reset() before step()"

        new_position = _ACTION_TO_POSITION[int(action)]
        position_change = abs(new_position - self.current_position)

        # Reward: position * spread_change - cost * |position_change|
        spread_change = float(self._current_pair["targets"][self._current_step])
        raw_reward = (
            self.current_position * spread_change
            - self._transaction_cost * position_change
        )
        reward = float(np.clip(raw_reward, -self._reward_clip, self._reward_clip))

        # Update position
        self.current_position = new_position

        # Advance step
        self._current_step += 1
        terminated = self._current_step >= self._current_pair["n_bars"]
        truncated = False

        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_observation()

        info = {
            "group_id": self._current_pair["group_id"],
            "step": self._current_step,
            "position": self.current_position,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Build the flattened observation vector.

        Extracts the lookback window of scaled features, augments each row
        with ``current_position`` and ``time_fraction``, then flattens to
        ``(lookback * n_channels,)``.
        """
        pair = self._current_pair
        start = self._current_step - self._lookback + 1
        end = self._current_step + 1
        window = pair["features"][start:end]  # (lookback, 31)

        # Augment: position + time fraction
        n_rows = window.shape[0]
        time_fraction = self._current_step / max(pair["n_bars"] - 1, 1)

        position_col = np.full((n_rows, 1), self.current_position, dtype=np.float32)
        time_col = np.full((n_rows, 1), time_fraction, dtype=np.float32)

        augmented = np.concatenate([window, position_col, time_col], axis=1)  # (lookback, 33)
        return augmented.flatten().astype(np.float32)
