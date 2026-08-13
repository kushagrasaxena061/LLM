# tests/test_lora_integration.py
"""Integration tests for applying LoRA to the full GPT model."""

import torch
from model.config import GPTConfig
from model.transformer import GPT
from fine_tuning.inject import inject_lora_to_model

def test_full_model_lora_injection():
    """Verifies that injecting LoRA freezes the base model and leaves only LoRA weights trainable."""
    # 1. Initialize a small GPT model
    config = GPTConfig(
        vocab_size=260,
        context_length=32,
        d_model=64,
        n_layers=2,
        n_heads=2
    )
    model = GPT(config)
    
    # Count initial total parameters
    total_params_before = sum(p.numel() for p in model.parameters())
    
    # 2. Inject LoRA into attention projections
    model = inject_lora_to_model(model, rank=4, alpha=16)
    
    # 3. Audit parameter states
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            assert "lora_" in name, f"Unexpected trainable parameter found: {name}"
        else:
            frozen_params += param.numel()
            
    assert trainable_params > 0, "No trainable LoRA parameters found!"
    assert frozen_params > trainable_params, "Base parameters were not frozen properly!"
    
    print(f"\n✅ LoRA Integration Passed!")
    print(f"   - Frozen Base Parameters:   {frozen_params:,}")
    print(f"   - Trainable LoRA Parameters: {trainable_params:,}")
    print(f"   - LoRA Parameter Ratio:     {(trainable_params / total_params_before) * 100:.2f}% of total model")
