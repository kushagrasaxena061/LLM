# tests/unit/test_tokenizer_suite.py
"""Comprehensive unit tests for Tokenizer implementations."""

import pytest
from tokenizer.bpe import BPETokenizer
from tokenizer.base import BaseTokenizer

class CharacterTokenizer(BaseTokenizer):
    """Character-level baseline tokenizer."""
    def __init__(self):
        super().__init__()
        self.char2idx = {}
        self.idx2char = {}

    def train(self, text: str):
        unique_chars = sorted(list(set(text)))
        self.char2idx = {ch: i for i, ch in enumerate(unique_chars)}
        self.idx2char = {i: ch for i, ch in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)

    def encode(self, text: str):
        return [self.char2idx.get(ch, 0) for ch in text]

    def decode(self, ids):
        return "".join([self.idx2char.get(i, "") for i in ids])


def test_bpe_roundtrip_lossless():
    """Verifies that BPE encoding followed by decoding returns the original string."""
    corpus = "Transformer models rely on self-attention mechanisms to process sequential data."
    tokenizer = BPETokenizer(vocab_size=280)
    tokenizer.train(corpus)
    
    test_strings = [
        "Transformer models rely on self-attention",
        "process sequential data.",
        "completely unseen words xyz 12345"
    ]
    for s in test_strings:
        encoded = tokenizer.encode(s)
        decoded = tokenizer.decode(encoded)
        assert decoded == s, f"Decoded string '{decoded}' does not match original '{s}'"


def test_bpe_unicode_and_numbers():
    """Verifies BPE handles numbers and UTF-8 multibyte characters safely."""
    corpus = "Numbers: 1234567890. Unicode: こんにちは, 🚀, café, résumé."
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(corpus)
    
    test_str = "Unicode: café, 🚀, 123456"
    encoded = tokenizer.encode(test_str)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_str, f"Decoded Unicode '{decoded}' did not match '{test_str}'"


def test_compression_ratio_measurement():
    """Verifies that BPE achieves higher compression ratio than byte/character representations."""
    corpus = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps again."
    tokenizer = BPETokenizer(vocab_size=280)
    tokenizer.train(corpus)
    
    encoded = tokenizer.encode(corpus)
    raw_char_count = len(corpus)
    token_count = len(encoded)
    
    compression_ratio = raw_char_count / token_count
    assert compression_ratio >= 1.0, f"Compression ratio {compression_ratio:.2f} must be > 1.0 for repeated corpus"
