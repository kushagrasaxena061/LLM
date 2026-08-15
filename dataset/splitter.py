"""Deterministic Train/Validation Splitter for Pretraining Corpus."""
import hashlib
from typing import List, Tuple, Any

def deterministic_hash_split(
    items: List[Any], 
    val_ratio: float = 0.1, 
    salt: str = "minigpt_151m"
) -> Tuple[List[Any], List[Any]]:
    """
    Deterministically partitions items (chunks, file paths, or document IDs)
    into training and validation sets using SHA-256 hashing.
    """
    train_split = []
    val_split = []
    
    for item in items:
        item_bytes = str(item).encode("utf-8") + salt.encode("utf-8")
        hash_hex = hashlib.sha256(item_bytes).hexdigest()
        
        # Convert first 8 hex characters to normalized float in [0.0, 1.0)
        hash_val = int(hash_hex[:8], 16) / 0xffffffff
        
        if hash_val < val_ratio:
            val_split.append(item)
        else:
            train_split.append(item)
            
    return train_split, val_split
