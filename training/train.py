"""Production Training Loop for Canonical MiniGPT-151M."""
import argparse
import math
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from model.config import canonical_151m_config, tiny_test_config, GPTConfig
from model.transformer import GPT
from training.checkpointing import save_full_checkpoint, load_full_checkpoint

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in set(model.parameters()))

@torch.no_grad()
def evaluate_loss(model: nn.Module, val_dataset: Dataset, device: str, max_batches: int = 10):
    """
    Computes genuine validation loss and perplexity.
    Gradients are strictly disabled via @torch.no_grad().
    """
    model.eval()
    if val_dataset is None or len(val_dataset) == 0:
        model.train()
        return 0.0, 0.0
    
    loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    total_loss = 0.0
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y, use_cache=False)
        total_loss += loss.item()
        count += 1
        if count >= max_batches: break
            
    model.train()  # Safely restore training mode
    avg_loss = total_loss / max(1, count)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, perplexity

def train_production_151m(
    config: GPTConfig = canonical_151m_config,
    train_dataset: Dataset = None,
    val_dataset: Dataset = None,
    max_steps: int = 1000,
    eval_interval: int = 100,
    eval_batches: int = 10,
    batch_size: int = 4,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup_steps: int = 50,
    grad_clip: float = 1.0,
    device: str = None,
    checkpoint_path: str = "checkpoints/minigpt_151m_ckpt.pt",
    resume: bool = False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    print("=== MiniGPT Production Training ===")
    model = GPT(config).to(device)
    actual_params = count_parameters(model)
    print(f"Instantiated Model Parameters: {actual_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, betas=(0.9, 0.95), weight_decay=0.1)
    
    use_cuda = (device == "cuda" and torch.cuda.is_available())
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)
        
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if step > max_steps:
            return min_lr / max_lr
        decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return (min_lr + coeff * (max_lr - min_lr)) / max_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    start_epoch = 0
    if resume and os.path.exists(checkpoint_path):
        start_step, start_epoch = load_full_checkpoint(
            checkpoint_path, model, optimizer, scheduler=scheduler, scaler=scaler, device=device
        )
        print(f"✅ Successfully resumed from step {start_step}")

    model.train()
    print("Training Loop Ready.")
    
    # Safely handle when datasets are omitted (for instantiation tests)
    if train_dataset is None:
        return model, actual_params
        
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    data_iter = iter(train_loader)
    
    # ----------------------------------------------------
    # Production Training Loop
    # ----------------------------------------------------
    print("DEBUG: Dataset loaded, entering training loop now...")
    for step in range(start_step, max_steps):
        # Periodic Validation & Logging
        if step % eval_interval == 0:
            val_loss, val_ppl = evaluate_loss(model, val_dataset, device, max_batches=eval_batches)
            print(f"Step {step} | Val Loss: {val_loss:.4f} | Perplexity: {val_ppl:.4f}")
            save_full_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler, step, start_epoch)
            
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)
            
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        # AMP forward pass explicitly bound
        if use_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                _, loss, _ = model(x, targets=y, use_cache=False)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss, _ = model(x, targets=y, use_cache=False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
        scheduler.step()
        
        if step > 0 and step % 10 == 0:
            print(f"Step {step} | Train Loss: {loss.item():.4f}")
            
    return model, actual_params

def run_deterministic_smoke_test():
    device = "cpu"
    model = GPT(tiny_test_config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    x = torch.tensor([[1, 2, 3, 4, 1, 2, 3, 4]], dtype=torch.long, device=device)
    y = torch.tensor([[2, 3, 4, 1, 2, 3, 4, 1]], dtype=torch.long, device=device)

    initial_loss, final_loss = None, None
    for step in range(30):
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(x, targets=y, use_cache=False)
        loss.backward()
        optimizer.step()
        if step == 0: initial_loss = loss.item()
        final_loss = loss.item()

    assert final_loss < initial_loss, "Smoke test failed: loss did not decrease"
    print(f"✅ Smoke Test Passed: Loss decreased ({initial_loss:.4f} -> {final_loss:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniGPT Studio Training Entry Point")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast deterministic smoke test")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    if args.smoke_test: run_deterministic_smoke_test()
    else: train_production_151m(resume=args.resume)
