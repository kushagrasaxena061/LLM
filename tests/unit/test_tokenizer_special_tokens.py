# tests/unit/test_tokenizer_special_tokens.py
"""Unit tests verifying BPE special tokens and UTF-8 recovery."""

from tokenizer.bpe import BPETokenizer

def test_bpe_special_tokens_roundtrip():
    tokenizer = BPETokenizer(vocab_size=300)
    prompt = "<|im_start|>system\nYou are a helpful AI.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>"
    
    ids = tokenizer.encode(prompt)
    decoded = tokenizer.decode(ids)
    
    assert ids[0] == tokenizer.special_tokens["<|im_start|>"]
    assert decoded == prompt
