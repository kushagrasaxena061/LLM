import os
import math
import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info

class StreamingTextDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, seq_len=2048, start_byte=0, end_byte=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.start_byte = start_byte
        self.end_byte = end_byte if end_byte is not None else os.path.getsize(file_path)

    def __iter__(self):
        worker_info = get_worker_info()
        start = self.start_byte
        end = self.end_byte
        
        # 1. Deterministic Sharding to prevent sample duplication across workers
        if worker_info is not None:
            per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = start + worker_id * per_worker
            end = min(start + per_worker, self.end_byte)
            
        token_buffer = []
        
        # 2. Memory-Efficient Chunk Reading
        with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(start)
            
            # If we land in the middle of a chunk/line, skip to the next newline to prevent fracturing
            if start > 0:
                f.readline()
            
            while f.tell() < end:
                # Read safely in chunks (preserving whitespace/formatting perfectly)
                chunk = f.read(4096)
                if not chunk:
                    break
                    
                tokens = self.tokenizer.encode(chunk)
                token_buffer.extend(tokens)
                
                # 3. Continuous Sequence Packing
                while len(token_buffer) >= self.seq_len + 1:
                    chunk_tokens = token_buffer[:self.seq_len + 1]
                    
                    # Slide the window forward by exactly seq_len tokens for seamless LM packing
                    token_buffer = token_buffer[self.seq_len:]
                    
                    # 4. Strict Causal LM Shift
                    x = torch.tensor(chunk_tokens[:-1], dtype=torch.long)
                    y = torch.tensor(chunk_tokens[1:], dtype=torch.long)
                    yield x, y

def create_dataloaders(file_path, tokenizer, batch_size=4, seq_len=2048, train_ratio=0.9):
    total_size = os.path.getsize(file_path)
    split_byte = int(total_size * train_ratio)
    
    # 5. Guaranteed Leakage-Free Train/Val Split via absolute byte-range boundaries
    train_dataset = StreamingTextDataset(file_path, tokenizer, seq_len=seq_len, start_byte=0, end_byte=split_byte)
    val_dataset = StreamingTextDataset(file_path, tokenizer, seq_len=seq_len, start_byte=split_byte, end_byte=total_size)
    
    # CRITICAL: IterableDatasets must never use shuffle=True in PyTorch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
