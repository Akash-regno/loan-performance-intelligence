"""
src/utils/seed.py
-----------------
Global random seed setter. Must be called at the very start of every
entry-point script (training, inference, notebooks).

Usage:
    from src.utils.seed import set_global_seed
    set_global_seed()          # uses RANDOM_SEED from config
    set_global_seed(123)       # override for one-off experiment
"""

import os
import random
import numpy as np

RANDOM_SEED: int = 42


def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Seed Python random, NumPy, and any available deep-learning backends."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional: seed PyTorch if installed
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    # Optional: seed TensorFlow if installed
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


set_all_seeds = set_global_seed

