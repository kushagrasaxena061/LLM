# tests/test_inference.py
"""Unit tests for inference and generation pipelines."""

import torch
from configs.base_config import env_config
from model.config import GPTConfig
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from inference.generate import generate_text

def test_generation_pipeline():
    """Verifies that the trained or untrained model can execute text generation without crashing."""
    config = GPTConfig(
        vocab_size=300,
        context_length=32,
        d_model=32,
        n_layers=2,
        n_heads=2
    )
    
    # FIX: Move the model weights to the target device (MPS/CUDA/CPU)
    model = GPT(config).to(env_config.device)
    
    # Train a tiny tokenizer for testing
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train("The quick brown fox jumps over the lazy dog.")
    
    # Run generation
    prompt = "The quick"
    output = generate_text(model, tokenizer, prompt, max_new_tokens=5, device=env_config.device, return_full_text=True)
    
    assert isinstance(output, str), "Generation did not return a string!"
    assert len(output) > len(prompt), "Generation failed to append new tokens!"
    print(f"\n✅ Generation Test Passed! Output: '{output}'")
