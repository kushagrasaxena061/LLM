# 3. tests/test_tokenizer.py

# Why it exists: To prove mathematically that 
# if we encode a string into numbers, and decode those numbers, 
# we get the exact original string back (a lossless transformation).

# tests/test_tokenizer.py
"""Unit tests for tokenizer functionality."""


from tokenizer.character import CharacterTokenizer

# A tiny dummy dataset to train our tokenizer on
DUMMY_CORPUS = "Hello World! This is an LLM engineering project."

def test_character_tokenizer_lossless():
    """Tests if encoding and then decoding returns the exact original text."""
    # 1. Initialize the tokenizer with our corpus
    tokenizer = CharacterTokenizer(DUMMY_CORPUS)
    
    # 2. Pick a string to test (must only contain characters from the corpus)
    test_string = "Hello LLM!"
    
    # 3. Encode the string into integers
    encoded_ids = tokenizer.encode(test_string)
    
    # 4. Decode the integers back into a string
    decoded_string = tokenizer.decode(encoded_ids)
    
    # 5. Assert that the process was completely lossless
    assert test_string == decoded_string, f"Expected '{test_string}', got '{decoded_string}'"
    
    # Print out the engineering data so we can visually inspect it
    print(f"\nOriginal: '{test_string}'")
    print(f"Encoded:  {encoded_ids}")
    print(f"Decoded:  '{decoded_string}'")
    print(f"Vocab Size: {tokenizer.vocab_size}")


# Add this import at the top of your test_tokenizer.py file:
from tokenizer.bpe import BPETokenizer


# Add this new test at the bottom of test_tokenizer.py:
def test_bpe_tokenizer():
    """Tests the BPE tokenizer for compression and lossless decoding."""
    # A longer corpus so the tokenizer can find repeating patterns
    training_corpus = "aaabdaaabac" * 10 
    
    # We start with 256 base bytes. Let's add 4 merges, making vocab = 260
    tokenizer = BPETokenizer(vocab_size=270)
    tokenizer.train(training_corpus)
    
    test_string = "aaabdaaabac"
    
    # Encode using our trained BPE
    encoded_ids = tokenizer.encode(test_string)
    decoded_string = tokenizer.decode(encoded_ids)
    
    # 1. Assert Lossless
    assert test_string == decoded_string, "BPE Decoding was not lossless!"
    
    # 2. Assert Compression
    # The raw string is 11 bytes. Our encoded version should be shorter due to merges.
    assert len(encoded_ids) < len(test_string), "BPE did not compress the text!"
    
    compression_ratio = len(test_string) / len(encoded_ids)
    
    print("\n--- BPE Tokenizer Results ---")
    print(f"Original String:   '{test_string}' (Length: {len(test_string)})")
    print(f"BPE Encoded Array: {encoded_ids} (Length: {len(encoded_ids)})")
    print(f"Compression Ratio: {compression_ratio:.2f}x")
    print(f"Learned Merges:    {tokenizer.merges}")
