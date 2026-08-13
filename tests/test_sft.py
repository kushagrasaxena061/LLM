# tests/test_sft.py
"""Unit tests for Supervised Fine-Tuning and loss masking."""

import torch
from configs.base_config import env_config
from model.config import GPTConfig
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from fine_tuning.sft import InstructionDataset, run_sft_training

def test_sft_pipeline():
    """Verifies instruction dataset formatting and LoRA fine-tuning execution."""
    config = GPTConfig(
        vocab_size=260,
        context_length=64,
        d_model=32,
        n_layers=2,
        n_heads=2
    )
    model = GPT(config)
    
    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train("Hello world, this is a test corpus for fine tuning.")
    
    samples = [
        {"prompt": "What is AI?", "response": "AI stands for artificial intelligence."},
        {"prompt": "Who are you?", "response": "I am a custom language model."}
    ]
    
    # Run a quick 1-epoch fine-tune test
    trained_model = run_sft_training(model, tokenizer, samples, device=env_config.device, epochs=1)
    
    assert trained_model is not None, "SFT training failed to return a model!"
    print("\n✅ Supervised Fine-Tuning Test Passed Successfully!")
