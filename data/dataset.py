# data/dataset.py
"""PyTorch Dataset pipeline for Next-Token Prediction."""

import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer.bpe import BPETokenizer


class LLMDataset(Dataset):
    def __init__(self, text_data: str, tokenizer: BPETokenizer, context_length: int):
        """
        Initializes the dataset by tokenizing the entire text and preparing it for chunking.
        
        Args:
            text_data (str): The raw string of the entire corpus (e.g., Shakespeare text).
            tokenizer (BPETokenizer): The tokenizer used to encode the text.
            context_length (int): The maximum number of tokens the model can see at once.
        """
        self.context_length = context_length
        
        # 1. Encode the entire text into a giant 1D array of integers
        # In a real enterprise system, we would stream this from disk. 
        # For a 1MB file, RAM is fine.
        self.data = tokenizer.encode(text_data)
        
        # 2. Convert the standard Python list into a highly optimized PyTorch Tensor
        self.data = torch.tensor(self.data, dtype=torch.long)
        
    def __len__(self) -> int:
        """
        Returns the total number of possible chunks we can extract.
        We subtract context_length so we don't read past the end of the data array.
        """
        return len(self.data) - self.context_length
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fetches a single (X, Y) pair from the data for training.
        
        Args:
            idx (int): The random starting position chosen by the PyTorch DataLoader.
            
        Returns:
            Tuple containing:
            - X: The input sequence of length `context_length`
            - Y: The target sequence of length `context_length` (shifted right by 1)
        """
        # X is the chunk from idx to idx + context_length
        x = self.data[idx : idx + self.context_length]
        
        # Y is the exact same size chunk, but shifted forward by exactly 1 token
        y = self.data[idx + 1 : idx + self.context_length + 1]
        
        return x, y

def create_dataloader(
    text_path: str, 
    tokenizer: BPETokenizer, 
    context_length: int, 
    batch_size: int, 
    shuffle: bool = True
) -> DataLoader:
    """Helper function to load the text file and create a PyTorch DataLoader."""
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    dataset = LLMDataset(text, tokenizer, context_length)
    
    # DataLoader automatically handles batching our 1D chunks into 2D matrices
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
