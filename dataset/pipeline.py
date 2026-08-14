import torch
from torch.utils.data import Dataset
from typing import List

class StreamingContextDataset(Dataset):
    """Production dataset pipeline with sliding window context truncation."""
    def __init__(self, token_ids: List[int], context_length: int):
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self):
        return max(0, len(self.token_ids) - self.context_length)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.token_ids[idx : idx + self.context_length], dtype=torch.long)
        y = torch.tensor(self.token_ids[idx + 1 : idx + 1 + self.context_length], dtype=torch.long)
        return x, y
