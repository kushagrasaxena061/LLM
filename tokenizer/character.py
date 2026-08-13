# 2. tokenizer/character.py

# Why it exists: This is the simplest possible tokenizer. 
# Every unique letter/symbol becomes a token. 
# It is terrible for performance (compression ratio is exactly 1.0) 
# because the model has to predict text letter-by-letter, 
# but it is mathematically perfect for learning how the encoding/decoding mapping works.

# tokenizer/character.py
"""A simple character-level tokenizer for foundational understanding."""

from tokenizer.base import BaseTokenizer
from utils.logger import get_logger

logger = get_logger(__name__)

class CharacterTokenizer(BaseTokenizer):
    def __init__(self, dataset_text: str):
        """
        Initializes the tokenizer by reading a dataset and finding every unique character.
        
        Args:
            dataset_text (str): The raw text used to build the vocabulary.
        """
        # 1. Find all unique characters in the text and sort them for consistency
        chars = sorted(list(set(dataset_text)))
        
        # 2. Create the Vocabulary mappings
        # stoi (String TO Integer): e.g., {'a': 0, 'b': 1, 'c': 2}
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        
        # itos (Integer TO String): e.g., {0: 'a', 1: 'b', 2: 'c'}
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        
        # Save the size of the vocabulary
        self._vocab_size = len(chars)
        
        logger.info(
            "CharacterTokenizer initialized", 
            vocab_size=self._vocab_size,
            sample_chars=chars[:10] # Show first 10 chars for debugging
        )

    def encode(self, text: str) -> list[int]:
        """
        Translates text to integers. 
        Loops through every character, looks it up in the stoi dictionary, and returns a list.
        """
        try:
            return [self.stoi[c] for c in text]
        except KeyError as e:
            # Security/Robustness: What happens if someone inputs a character we haven't seen?
            raise ValueError(f"Character {e} not found in vocabulary! You must handle Unknown (UNK) tokens.")

    def decode(self, ids: list[int]) -> str:
        """
        Translates integers back to text.
        Loops through every ID, looks it up in the itos dictionary, and joins them into a string.
        """
        return ''.join([self.itos[i] for i in ids])

    @property
    def vocab_size(self) -> int:
        return self._vocab_size
