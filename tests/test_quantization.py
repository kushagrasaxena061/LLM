# tests/test_quantization.py
"""Unit tests for model quantization and compression metrics."""

import torch
from configs.base_config import env_config
from model.config import GPTConfig
from model.transformer import GPT
from quantization.quantize import quantize_model_to_int8, get_model_size_mb

def test_quantization_pipeline():
    """Verifies that model quantization runs, executes forward passes, and reduces size."""
    config = GPTConfig(
        vocab_size=260,
        context_length=32,
        d_model=64,
        n_layers=2,
        n_heads=2
    )
    model = GPT(config).to(env_config.device)
    
    # 1. Measure FP32 size
    fp32_size = get_model_size_mb(model)
    
    # 2. Quantize model to INT8
    quantized_model = quantize_model_to_int8(model)
    quantized_model.eval()
    
    # 3. Measure Quantized size
    int8_size = get_model_size_mb(quantized_model)
    
    # 4. Verify forward pass works post-quantization (pass targets to evaluate full sequence logits)
    idx = torch.randint(0, config.vocab_size, (1, 10), device=env_config.device)
    with torch.no_grad():
        logits, _ = quantized_model(idx, targets=idx)
        
    assert logits.shape == (1, 10, config.vocab_size), f"Quantized logit shape invalid: {logits.shape}"
    assert int8_size < fp32_size, "Quantization did not reduce model size!"
    
    print(f"\n✅ Quantization Test Passed!")
    print(f"   - Original FP32 Size: {fp32_size:.2f} MB")
    print(f"   - Quantized INT8 Size: {int8_size:.2f} MB")
    print(f"   - Compression Ratio: {fp32_size / int8_size:.2f}x")
