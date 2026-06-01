import os
import random

import numpy as np
import torch

from src.config.config import CONFIG, logger


def set_seed(seed=None):
    """Sets the seed for pseudo-random number generators to ensure reproducibility.

    Args:
        seed (int, optional): The seed value to use. Defaults to CONFIG training seed.
    """
    if seed is None:
        seed = CONFIG["training"]["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed, "all")
    except (ImportError, AttributeError):
        pass

    logger.info(f"Random seed set to {seed}")
