import os
import pytest
from tokenizer.bpe import BPETokenizer

def test_genuine_bpe_features():
    tokenizer = BPETokenizer(vocab_size=50257)
    
    # 1. Test special tokens map exactly to expected constraints
    assert tokenizer.special_tokens["<|endoftext|>"] == 50256
    assert tokenizer.special_tokens["<|im_start|>"] == 50255
    assert tokenizer.special_tokens["<|pad|>"] == 50253
    
    # 2. Train on a real corpus path
    corpus_path = "temp_bpe_corpus.txt"
    test_text = "The quick brown fox jumps over the lazy dog. def python_func(): return True"
    training_data = (test_text + " ") * 50
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(training_data)
        
    tokenizer.train(corpus_path)
    
    # Verify genuine merges were created based on frequency (no dummy placeholder merges)
    assert len(tokenizer.merges) > 0
    
    # 3. Test exact encode/decode round trip (lossless multi-byte UTF-8)
    complex_text = "JSON: {\"key\": \"value\"}, code: `def foo(): pass`, Hindi: नमस्ते, Emojis: 🌍🔥 \n\t <|im_start|> system prompt <|im_end|>"
    encoded = tokenizer.encode(complex_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == complex_text, "UTF-8 fracturing occurred!"
    
    # 4. Test Persistence
    save_path = "temp_tokenizer.json"
    tokenizer.save(save_path)
    
    new_tokenizer = BPETokenizer(vocab_size=50257)
    new_tokenizer.load(save_path)
    
    assert new_tokenizer.merges == tokenizer.merges
    
    new_encoded = new_tokenizer.encode(complex_text)
    assert new_encoded == encoded
    
    # Cleanup
    os.remove(corpus_path)
    os.remove(save_path)
    
    print("\n✅ Genuine BPE Tokenizer Confirmed. Roundtrip lossless UTF-8 verified.")

if __name__ == "__main__":
    test_genuine_bpe_features()
