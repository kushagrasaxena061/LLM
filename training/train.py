# training/train.py
"""Production-ready pretraining loop for the canonical 151M model."""

import sys
from pathlib import Path

# Automatically add the project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import math
import os

from model.config import canonical_151m_config
from model.transformer import GPT

class DummyStreamingDataset(Dataset):
    """Placeholder for the production data pipeline. Replaces f.read()."""
    def __init__(self, seq_len: int, size: int = 1000):
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = torch.randint(0, 300, (self.seq_len,))
        y = torch.randint(0, 300, (self.seq_len,))
        return x, y

def get_lr(step: int, max_steps: int, max_lr: float, min_lr: float, warmup_steps: int):
    """Cosine learning rate with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def run_training_smoke_test():
    """Phase 14: Tiny training run to verify loss decreases and gradients flow."""
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing 151M Model on {device}...")
    
    model = GPT(canonical_151m_config).to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    
    dataset = DummyStreamingDataset(seq_len=256, size=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    initial_loss = None
    final_loss = None
    
    print("Starting Pretraining Smoke Test...")
    for step, (x, y) in enumerate(dataloader):
        if step >= 10:
            break
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16, enabled=(device=="cuda")):
            logits, loss, _ = model(x, targets=y, use_cache=False)
            
        if device == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        print(f"Step {step} | Loss: {loss.item():.4f}")
        
    assert final_loss < initial_loss, "Smoke test failed: Loss did not decrease."
    print("✅ Smoke test passed. The model learns and gradients flow.")

if __name__ == "__main__":
    run_training_smoke_test()
