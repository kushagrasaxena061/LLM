# tests/test_transformer.py
"""Unit tests to verify end-to-end model execution and parameter scaling."""

import torch

from model.config import GPTConfig
from model.transformer import GPT


def test_gpt_forward_pass():
    """Verifies that token IDs produce correct logit shapes and non-zero loss."""
    # 1. Setup a tiny model config for fast execution
    config = GPTConfig(
        vocab_size=260,
        context_length=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
    )
    model = GPT(config)

    # 2. Fake inputs and targets (Batch=2, SeqLen=16)
    idx = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    # 3. Forward pass with targets (Training Mode)
    logits, loss, _ = model(idx, targets)

    # Assert Logit shape: (Batch, SeqLen, vocab_size)
    assert logits.shape == (2, 16, config.vocab_size), f"Unexpected logit shape: {logits.shape}"
    assert loss is not None, "Loss was not calculated when targets were provided!"
    assert loss.item() > 0, "Loss must be positive!"

    print(f"\n✅ Training forward pass successful! Loss: {loss.item():.4f}")

    # 4. Forward pass without targets (Inference Mode)
    inference_logits, inference_loss, _ = model(idx)

    # In inference mode, logits should only be calculated for the last token position (Shape: 2, 1, vocab_size)
    assert inference_logits.shape == (2, 1, config.vocab_size), f"Unexpected inference logit shape: {inference_logits.shape}"
    assert inference_loss is None, "Inference loss should be None when targets are absent!"

    print("✅ Inference mode optimization verified!")


def test_parameter_count():
    """Verifies parameter counting utility."""
    config = GPTConfig(
        vocab_size=1000,
        context_length=128,
        d_model=128,
        n_layers=4,
        n_heads=4,
    )
    model = GPT(config)
    params = model.get_num_params()

    assert params > 0, "Model has 0 parameters!"
    print(f"✅ Parameter counter functional. Total Parameters for small config: {params:,}")
