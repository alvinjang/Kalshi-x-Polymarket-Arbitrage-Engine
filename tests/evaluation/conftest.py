"""Shared test fixtures for evaluation module tests."""
import numpy as np
import pytest


@pytest.fixture
def sample_y_true():
    """10-element ndarray of ground-truth spread changes.

    Mix of positive, negative, and near-zero values so directional
    accuracy and profit simulation have non-trivial inputs.
    """
    return np.array(
        [0.05, -0.03, 0.02, -0.01, 0.04, -0.02, 0.01, 0.00, 0.03, -0.04],
        dtype=float,
    )


@pytest.fixture
def sample_y_pred():
    """10-element ndarray of predicted spread changes matching sample_y_true."""
    return np.array(
        [0.04, -0.02, 0.03, -0.02, 0.05, -0.01, 0.00, 0.01, 0.02, -0.03],
        dtype=float,
    )
