"""Tests for SpreadTradingEnv (Gymnasium environment for PPO training).

TDD RED phase: these tests define the SpreadTradingEnv contract before
any implementation exists.  All tests import from ``src.models.trading_env``,
which does not yet exist, so collection should fail with ``ImportError``.

Tests cover:
  - Gymnasium inheritance and spaces
  - Reset / step tuple contracts
  - Observation shape (198,) = 6 bars x 33 channels
  - Reward: transaction cost, clipping
  - Episode termination at end of pair
  - Short-pair skipping (<6 bars)
"""
from __future__ import annotations

import gymnasium
import numpy as np
import pandas as pd
import pytest

from src.models.trading_env import SpreadTradingEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_env_data(
    n_group0: int = 20,
    n_group1: int = 3,
    n_features: int = 31,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Create synthetic DataFrame + targets for env tests.

    Returns (X, y, feature_cols) where X has ``group_id`` plus
    ``n_features`` numeric columns.  Group 0 has ``n_group0`` rows,
    group 1 has ``n_group1`` rows.
    """
    rng = np.random.default_rng(seed)
    n_total = n_group0 + n_group1
    group_ids = np.array([0] * n_group0 + [1] * n_group1)

    feature_cols = [f"feat_{i}" for i in range(n_features)]
    data = {"group_id": group_ids}
    for col in feature_cols:
        data[col] = rng.standard_normal(n_total)

    X = pd.DataFrame(data)
    y = rng.standard_normal(n_total) * 0.05
    return X, y, feature_cols


@pytest.fixture
def env_data():
    """Standard env fixture: group 0 = 20 rows, group 1 = 3 rows."""
    return _make_env_data(n_group0=20, n_group1=3)


@pytest.fixture
def env(env_data):
    """Pre-built SpreadTradingEnv from the standard fixture."""
    X, y, feature_cols = env_data
    return SpreadTradingEnv(
        X_train=X,
        y_train=y,
        feature_cols=feature_cols,
        lookback=6,
        transaction_cost=0.02,
        reward_clip=0.5,
    )


# ---------------------------------------------------------------------------
# Gymnasium inheritance / spaces
# ---------------------------------------------------------------------------

class TestEnvInterface:
    def test_env_inherits_gymnasium_env(self, env):
        assert isinstance(env, gymnasium.Env)

    def test_action_space_is_discrete_3(self, env):
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        assert env.action_space.n == 3

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (198,)


# ---------------------------------------------------------------------------
# Reset contract
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_correct_obs_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == (198,)
        assert obs.dtype == np.float32

    def test_reset_position_is_zero(self, env):
        env.reset()
        # After reset, current_position should be 0 (no position)
        assert env.current_position == 0


# ---------------------------------------------------------------------------
# Step contract
# ---------------------------------------------------------------------------

class TestStep:
    def test_step_returns_valid_tuple(self, env):
        env.reset()
        result = env.step(0)  # hold
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (198,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_hold_action_no_transaction_cost(self, env):
        """Action 0 (hold) from position 0 -> no position change -> no cost."""
        env.reset()
        _obs, reward, _done, _trunc, _info = env.step(0)  # hold
        # position=0, actual_spread_change may be nonzero, but position*spread=0
        # transaction cost = 0.02 * |0 - 0| = 0
        # reward = 0*spread - 0 = 0 then clipped -> 0.0
        assert reward == pytest.approx(0.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class TestReward:
    def test_reward_includes_transaction_cost(self, env):
        """Going from position=0 to long (action=1) incurs -0.02 cost."""
        env.reset()
        # First step: go long (action=1)
        _obs, reward_long, _d, _t, _info = env.step(1)  # long
        # reward = 0 * spread_change - 0.02 * |1 - 0| = -0.02 + 0 = ...
        # position was 0 before step, so position*spread = 0*spread = 0
        # transaction = 0.02 * 1 = 0.02
        # reward = 0.0 - 0.02 = -0.02 (before clip)
        assert reward_long == pytest.approx(-0.02, abs=1e-8)

    def test_reward_clipped(self, env_data):
        """Reward is clipped to [-0.5, 0.5] range."""
        X, y, feature_cols = env_data
        # Make y values very large to test clipping
        y_extreme = np.ones_like(y) * 100.0
        env_clip = SpreadTradingEnv(
            X_train=X,
            y_train=y_extreme,
            feature_cols=feature_cols,
            lookback=6,
            transaction_cost=0.02,
            reward_clip=0.5,
        )
        env_clip.reset()
        # Go long (position becomes +1)
        env_clip.step(1)
        # Now step again holding long -- position=1, spread_change=100 -> huge reward -> clipped
        _obs, reward, _d, _t, _info = env_clip.step(1)
        assert -0.5 <= reward <= 0.5


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------

class TestEpisode:
    def test_episode_terminates_at_end_of_pair(self, env):
        """Done=True after stepping through all bars of a pair."""
        env.reset()
        terminated = False
        steps = 0
        max_steps = 200  # safety limit
        while not terminated and steps < max_steps:
            _obs, _reward, terminated, _trunc, _info = env.step(0)
            steps += 1
        assert terminated, "Episode should terminate when pair bars are exhausted"
        assert steps > 0

    def test_short_pair_skipped(self):
        """Pairs with <6 bars are skipped. If only short pairs exist, only valid pairs used."""
        # Create data: group 0 = 3 rows (<6 lookback), group 1 = 20 rows (valid)
        X, y, feature_cols = _make_env_data(n_group0=3, n_group1=20, seed=99)
        # Swap group order: group 0 is short, group 1 is long
        # _make_env_data already puts group 0 first with 3 rows, group 1 with 20 rows
        env = SpreadTradingEnv(
            X_train=X,
            y_train=y,
            feature_cols=feature_cols,
            lookback=6,
        )
        obs, _info = env.reset()
        # Should successfully reset (skipping group 0) and load group 1
        assert obs.shape == (198,)
        # Verify we can step through the episode
        terminated = False
        steps = 0
        while not terminated and steps < 200:
            _obs, _reward, terminated, _trunc, _info = env.step(0)
            steps += 1
        assert terminated
