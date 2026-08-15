"""Exhaustive Lossless Encode/Decode Tests."""
import pytest
from tokenizer.bpe import BPETokenizer

@pytest.fixture(scope="module")
def tokenizer():
    tok = BPETokenizer(vocab_size=50257)
    return tok

def test_english_punctuation_whitespace(tokenizer):
    text = "Hello, World! \n\tThis is an English test with punctuation.   "
    assert tokenizer.decode(tokenizer.encode(text)) == text

def test_numbers_json_python(tokenizer):
    text = '{"id": 12345, "val": [1, 2.5]}\ndef func(x):\n    return x + 1'
    assert tokenizer.decode(tokenizer.encode(text)) == text

def test_markdown_urls(tokenizer):
    text = "# Title\n[Link](https://example.com/api?q=1&b=2) **Bold** `code`"
    assert tokenizer.decode(tokenizer.encode(text)) == text

def test_unicode_emojis_hindi(tokenizer):
    text = "Unicode 🌍 emojis! Hindi: नमस्ते. Arabic: مرحبا. Kanji: 漢字."
    assert tokenizer.decode(tokenizer.encode(text)) == text

def test_strict_utf8_no_replacement_characters(tokenizer):
    text = " fractured 🌍 emojis testing"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert "\ufffd" not in decoded, "CRITICAL: Replacement character detected!"
    assert decoded == text

def test_special_tokens():
    tok = BPETokenizer()
    assert tok.encode("<|im_start|>")[0] == tok.special_tokens["<|im_start|>"]

def test_save_load_persistence(tmp_path):
    tok1 = BPETokenizer(vocab_size=1000)
    tok1.train("test")
    path = str(tmp_path / "tok.json")
    tok1.save(path)
    
    tok2 = BPETokenizer(vocab_size=1000)
    tok2.load(path)
    
    assert tok1.vocab_size == tok2.vocab_size
    assert tok1.special_tokens == tok2.special_tokens
    assert tok1.vocab == tok2.vocab
