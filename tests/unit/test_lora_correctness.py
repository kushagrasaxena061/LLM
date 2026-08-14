# tests/unit/test_lora_correctness.py
"""Unit tests to verify LoRA parameter freezing and gradient routing."""

import torch
from model.config import GPTConfig
from model.transformer import GPT
from fine_tuning.inject import inject_lora_to_model

def test_lora_freezes_base_weights():
    """
    CRITICAL LORA TEST:
    Proves that injecting LoRA successfully freezes the base model weights 
    and significantly reduces the trainable parameter count.
    """
    config = GPTConfig(vocab_size=100, context_length=16, d_model=32, n_layers=2, n_heads=2)
    base_model = GPT(config)
    
    # 1. Count parameters before LoRA
    total_params_before = sum(p.numel() for p in base_model.parameters())
    
    # 2. Inject LoRA
    lora_model = inject_lora_to_model(base_model, rank=4, alpha=16)
    
    # 3. Count parameters after LoRA
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total_params_after = sum(p.numel() for p in lora_model.parameters())
    
    # Assertions
    ratio = trainable_params / total_params_after
    
    assert total_params_after > total_params_before, "Total parameters should increase after adding LoRA matrices."
    assert ratio < 0.15, f"Trainable parameter ratio is too high: {ratio * 100:.2f}%. Base weights might not be frozen!"
    
    # Verify strict gradient isolation
    for name, param in lora_model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad is True, f"LoRA parameter {name} is incorrectly frozen!"
        elif "weight" in name and "lora" not in name and "norm" not in name:
            # Assuming standard linear weights are frozen
            assert param.requires_grad is False, f"Base parameter {name} was not frozen!"

    print(f"\n✅ LoRA Gradient Routing Verified!")
    print(f"   - Trainable Parameters: {trainable_params:,}")
    print(f"   - Total Parameters: {total_params_after:,}")
    print(f"   - Parameter Fraction: {ratio * 100:.2f}%")
