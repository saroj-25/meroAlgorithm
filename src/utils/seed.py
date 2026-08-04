"""Global seeding so that every table in results/ is byte-reproducible."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> np.random.Generator:
    """Seed python/numpy (+torch if present) and return a numpy Generator."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # optional dependency
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    return np.random.default_rng(seed)
