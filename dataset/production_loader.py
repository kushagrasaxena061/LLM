"""Scalable, Memory-Efficient Streaming & Packed Pretraining Dataset Pipeline."""

import os
from typing import Iterator, List, Tuple
import torch
from torch.utils.data import IterableDataset, Dataset

class ShardedTextStreamingDataset(IterableDataset):
    """
    Memory-efficient streaming dataset for large-scale pretraining.
    Reads file shards line-by-line without loading the entire corpus into RAM.
    Packs tokens into fixed-length blocks for causal language modeling.
    """
    def __init__(
        self,
        shard_paths: List[str],
        tokenizer,
        context_length: int = 2048,
        shuffle_shards: bool = True,
        seed: int = 42
    ):
        self.shard_paths = sorted(shard_paths)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.shuffle_shards = shuffle_shards
        self.seed = seed

        if self.shuffle_shards:
            import random
            rng = random.Random(self.seed)
            rng.shuffle(self.shard_paths)

    def _stream_shards(self) -> Iterator[int]:
        """Streams raw token IDs from file shards line by line."""
        for shard_path in self.shard_paths:
            with open(shard_path, "r", encoding="utf-8", errors="ignore") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    for tid in self.tokenizer.encode(chunk):
                        yield tid

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Packs tokens into fixed-length causal language modeling sequences."""
        buffer = []
        target_length = self.context_length + 1

        for token_id in self._stream_shards():
            buffer.append(token_id)
            if len(buffer) >= target_length:
                chunk = buffer[:target_length]
                buffer = buffer[self.context_length:]

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

class MemoryMappedDataset(Dataset):
    """
    Compatible wrapper for smaller legacy files or unit testing while maintaining fixed-length packing.
    """
    def __init__(self, texts: List[str], tokenizer, context_length: int = 2048):
        self.samples = []
        target_length = context_length + 1
        
        all_tokens = []
        for text in texts:
            all_tokens.extend(tokenizer.encode(text))
            
        for i in range(0, len(all_tokens) - target_length + 1, context_length):
            chunk = all_tokens[i:i + target_length]
            if len(chunk) == target_length:
                self.samples.append((
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long)
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]
