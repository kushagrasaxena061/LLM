import sys
import os
from pathlib import Path

# Safely inject the project root into Python's path so direct execution works
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
import pytest
from model.config import GPTConfig, canonical_151m_config
from model.transformer import GPT

def test_canonical_reproducibility():
    # 1. Verify architectural configuration
    assert canonical_151m_config.d_model == 768
    assert canonical_151m_config.n_layers == 12
    assert canonical_151m_config.n_heads == 12
    assert canonical_151m_config.vocab_size == 50257
    assert canonical_151m_config.weight_tying is True
    
    # 2. Instantiate the model independently
    model = GPT(canonical_151m_config)
    
    # 3. Mathematically calculate parameters directly from tensors
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nActual Parameter Count: {total_params:,}")
    
    # 4. Strictly assert it equals exactly 151,862,784
    assert total_params == 151862784, f"CRITICAL: Architecture compromised. Expected 151862784 parameters, got {total_params}"

if __name__ == "__main__":
    test_canonical_reproducibility()
    print("✅ Direct instantiation and exact parameter count verified independently.")
