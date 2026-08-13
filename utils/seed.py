# 4. utils/seed.py
# Where it goes: Inside the utils/ folder.
# Why it exists: Neural networks start with random numbers (weights). 
# If we don't lock down the random seed, every time you run the code, 
# you will get a different result. This file ensures absolute mathematical reproducibility.

# utils/seed.py
"""Reproducibility engine for locking random seeds across all frameworks."""

import os
import random

import numpy as np
import torch

from utils.logger import get_logger

# Initialize our custom logger for this file
logger = get_logger(__name__)

def set_seed(seed: int = 42) -> None:
    """
    Sets the random seed across every single library that generates random numbers.
    """
    # 1. Lock the standard Python random library
    random.seed(seed)
    
    # 2. Lock Python's dictionary hashing (prevents dicts from shuffling in memory)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # 3. Lock NumPy (which we will use for evaluation metrics later)
    np.random.seed(seed)

    # 4. Lock PyTorch's CPU random number generator
    torch.manual_seed(seed)
    
    # 5. Lock PyTorch's Apple Silicon (MPS) random number generator
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # 6. Lock PyTorch's NVIDIA (CUDA) random number generator (just in case you move to a cloud GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Log a beautiful message letting us know the seed is locked
    logger.info("Global seed established successfully", seed=seed)
