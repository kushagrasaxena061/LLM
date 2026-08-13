# tokenizer/bpe.py
"""Byte-Pair Encoding (BPE) Tokenizer implementation."""

from tokenizer.base import BaseTokenizer
from utils.logger import get_logger

logger = get_logger(__name__)

def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """
    Counts the frequencies of adjacent pairs of tokens.
    Example: [1, 2, 1, 2, 3] -> {(1, 2): 2, (2, 1): 1, (2, 3): 1}
    """
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """
    Replaces all consecutive occurrences of `pair` in `ids` with the new token `idx`.
    Example: merge([1, 2, 3, 1, 2], (1, 2), 4) -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # If we find the pair, replace it with the new index
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2 # Skip the next token because we merged it
        else:
            newids.append(ids[i])
            i += 1
    return newids

class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size: int):
        """
        Initializes the BPE tokenizer.
        Args:
            vocab_size: The target number of total tokens. Must be >= 256.
        """
        if vocab_size < 256:
            raise ValueError("Vocab size must be at least 256 to cover all UTF-8 bytes.")
            
        self.target_vocab_size = vocab_size
        self.num_merges = vocab_size - 256
        
        # The dictionary that stores our learned merges: e.g., (101, 102) -> 256
        self.merges: dict[tuple[int, int], int] = {}
        
        # The vocabulary mapping token IDs to their actual byte sequences
        # Initialize with the standard 256 UTF-8 bytes
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        
    def train(self, text: str):
        """
        Trains the tokenizer on a corpus of text, learning the most common pairs.
        """
        logger.info(f"Training BPE Tokenizer for {self.num_merges} merges...")
        
        # 1. Convert raw string into a list of UTF-8 byte integers (0-255)
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        # 2. Iteratively merge the most common pairs
        for i in range(self.num_merges):
            stats = get_stats(ids)
            if not stats:
                break # No more pairs to merge
                
            # Find the pair with the highest count
            best_pair = max(stats, key=stats.get)
            
            # Create a new token ID (starting at 256, then 257, etc.)
            new_id = 256 + i
            
            # Record the merge
            self.merges[best_pair] = new_id
            
            # Record the new byte sequence in our vocabulary
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Execute the merge on our training data
            ids = merge(ids, best_pair, new_id)
            
        logger.info("BPE Training complete", final_vocab_size=self.vocab_size)

    def encode(self, text: str) -> list[int]:
        """Translates text into BPE token IDs."""
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        # We must apply the merges in the exact same order we learned them
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find the pair in our current text that was merged earliest during training
            # If none of the pairs in the text are in our learned merges, we are done
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            
            if pair not in self.merges:
                break # Nothing else to merge
                
            # Apply the merge
            ids = merge(ids, pair, self.merges[pair])
            
        return ids

    def decode(self, ids: list[int]) -> str:
        """Translates BPE token IDs back into text."""
        # 1. Look up the bytes for each ID
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        # 2. Decode the bytes back to a UTF-8 string, replacing bad characters if necessary
        return text_bytes.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
