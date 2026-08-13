# 1. tokenizer/base.py

# Why it exists: In software engineering, 
# when you have multiple versions of something 
# (Character tokenizer, Word tokenizer, BPE tokenizer), 
# you create a "Base Class". It acts as a strict blueprint that all 
# future tokenizers must follow.

# tokenizer/base.py
"""Abstract base class for all tokenizers."""

from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """
    An abstract blueprint that guarantees every tokenizer we build will have
    an encode(), decode(), and vocab_size property.
    """
    
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Converts a string into a list of integer token IDs."""
        pass

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Converts a list of integer token IDs back into a string."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Returns the total number of unique tokens in the vocabulary."""
        pass
