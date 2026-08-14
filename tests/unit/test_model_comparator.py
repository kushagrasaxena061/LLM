# tests/unit/test_model_comparator.py
"""Unit tests for model comparison profiler."""

from model.config import GPTConfig
from evaluation.model_comparator import ModelComparator

def test_model_comparator_profiling():
    config = GPTConfig(vocab_size=100, context_length=16, d_model=32, n_layers=2, n_heads=2)
    profile = ModelComparator.profile_configuration(config, device="cpu")
    
    assert "Base_FP32" in profile
    assert "LoRA_PEFT" in profile
    assert "INT8_Quantized" in profile
    assert profile["LoRA_PEFT"]["trainable_pct"] < 15.0
    assert profile["INT8_Quantized"]["size_mb"] <= profile["Base_FP32"]["size_mb"]
