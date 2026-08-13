# 5. tests/test_foundations.py

# Where it goes: Inside the tests/ folder.
# Why it exists: We must prove our code works. T
# he pytest library automatically looks 
# for files starting with test_ and runs them to verify our logic.

# tests/test_foundations.py
"""Unit tests to verify foundational seed determinism and environment configuration."""

import torch

from configs.base_config import env_config
from utils.seed import set_seed


def test_seed_reproducibility_cpu():
    """
    Test to verify that identical seeds produce bitwise-identical random tensors on the CPU.
    """
    # Set seed to 42 and generate a 10x10 grid of random numbers
    set_seed(42)
    tensor_a = torch.randn(10, 10)

    # Set seed to 42 AGAIN and generate another 10x10 grid
    set_seed(42)
    tensor_b = torch.randn(10, 10)

    # Use assert to prove they are identical. If they aren't, the test fails.
    assert torch.equal(tensor_a, tensor_b), "CPU Tensors were not identical!"

def test_environment_device():
    """
    Test to verify proper device assignment. It should detect MPS on your Mac.
    """
    # Assert that the device chosen by our config is a valid PyTorch device
    assert env_config.device in ["cuda", "cpu", "mps"]
    
    # Print the detected device so we can see it in the terminal
    print(f"\n✅ Successfully running PyTorch on device: {env_config.device}")
