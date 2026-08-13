# tests/test_lora.py
"""Unit tests for LoRA fine-tuning components."""

import torch
import torch.nn as nn
from fine_tuning.lora import LoRALinear

def test_lora_parameter_freezing():
    """Verifies that base weights are frozen and LoRA parameters are trainable."""
    original = nn.Linear(64, 64)
    lora_layer = LoRALinear(original, rank=4, alpha=16)
    
    # Base weights must be frozen
    assert not lora_layer.weight.requires_grad, "Base weights were not frozen!"
    
    # LoRA parameters must require gradients
    assert lora_layer.lora_A.requires_grad, "LoRA matrix A is not trainable!"
    assert lora_layer.lora_B.requires_grad, "LoRA matrix B is not trainable!"
    
    # Verify forward pass shape
    x = torch.randn(2, 10, 64)
    out = lora_layer(x)
    assert out.shape == (2, 10, 64), f"Unexpected output shape: {out.shape}"
    
    print(f"\n✅ LoRA Layer verified. Trainable params in LoRA: {sum(p.numel() for p in lora_layer.parameters() if p.requires_grad):,}")
