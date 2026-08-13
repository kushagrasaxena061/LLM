# tests/test_dataset.py
"""Unit tests to verify data packing and shifting."""

import torch

from data.dataset import LLMDataset
from tokenizer.bpe import BPETokenizer


def test_dataset_x_y_shifting():
    """Verify that the target tensor Y is exactly the input tensor X shifted by 1."""
    # 1. Create a tiny dataset and train our tokenizer on it
    raw_text = "The quick brown fox jumps over the lazy dog."
    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train(raw_text)
    
    # 2. Create the dataset with a context length of 5 tokens
    context_length = 5
    dataset = LLMDataset(raw_text, tokenizer, context_length)
    
    # 3. Fetch the very first item
    x, y = dataset[0]
    
    # Ensure our shapes are correct: both should be a 1D vector of length 5
    assert x.shape == (context_length,), f"Expected shape ({context_length},), got {x.shape}"
    assert y.shape == (context_length,), f"Expected shape ({context_length},), got {y.shape}"
    
    # 4. The crucial test: The 2nd element of X MUST be the 1st element of Y
    assert torch.equal(x[1:], y[:-1]), "Data Pipeline Error: Y is not X shifted by one!"
    
    print("\n--- Data Pipeline Test ---")
    print(f"X (Input Tensor):  {x.tolist()}")
    print(f"Y (Target Tensor): {y.tolist()}")
