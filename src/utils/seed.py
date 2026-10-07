"""Reproducibility seed utility for all experiment scripts.

Covers: Python random, numpy, torch (CPU+CUDA), CUDNN, DataLoader workers,
PYTHONHASHSEED, and torch.use_deterministic_algorithms.
"""
# AI-assisted authorship: written with Anthropic Claude (Sonnet 4.5 / Opus 4.6) as pair-programming assistant. All design decisions and interpretations are the authors'.
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_all_seeds(seed: int = 42) -> None:
    """Seed every RNG source for reproducible experiments.

    Must be called BEFORE any data loading, model creation, or training.
    Forces single-threaded CPU execution on Apple Silicon for determinism.
    """
    # 1. Python stdlib
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch
    torch.manual_seed(seed)  # Seeds CPU and CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 4. CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 5. Deterministic algorithms (warn instead of raise for compatibility)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # 6. Apple Silicon: single-threaded to avoid OpenMP/Accelerate segfault
    #    AND to ensure deterministic thread scheduling
    if not torch.cuda.is_available():
        torch.set_num_threads(1)


def worker_init_fn(worker_id: int) -> None:
    """Seed DataLoader workers for reproducible shuffling.

    Pass as worker_init_fn= argument to any DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
