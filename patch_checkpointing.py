import os
import subprocess

os.makedirs("training", exist_ok=True)
os.makedirs("tests/integration", exist_ok=True)

# 1. Write training/checkpointing.py
with open("training/checkpointing.py", "w") as f:
    f.write("""import os
import torch
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

def save_full_checkpoint(filepath: str, model, optimizer, scheduler=None, scaler=None, step: int = 0, epoch: int = 0):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    mps_rng = None
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try: mps_rng = torch.mps.get_rng_state()
        except Exception: pass

    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "step": step,
        "epoch": epoch,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "rng_mps": mps_rng
    }
    tmp_filepath = filepath + ".tmp"
    torch.save(state, tmp_filepath)
    os.replace(tmp_filepath, filepath)
    logger.info(f"Checkpoint safely saved: {filepath} at step {step}")

def load_full_checkpoint(filepath: str, model, optimizer, scheduler=None, scaler=None, device="cpu"):
    if not os.path.exists(filepath): return 0, 0
    state = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"): scheduler.load_state_dict(state["scheduler"])
    if scaler and state.get("scaler"): scaler.load_state_dict(state["scaler"])
    
    if "rng_python" in state: random.setstate(state["rng_python"])
    if "rng_numpy" in state: np.random.set_state(state["rng_numpy"])
    if "rng_torch" in state: torch.set_rng_state(state["rng_torch"])
    
    if torch.cuda.is_available() and state.get("rng_cuda"):
        try: torch.cuda.set_rng_state_all(state["rng_cuda"])
        except Exception: pass
            
    if hasattr(torch, "mps") and torch.backends.mps.is_available() and state.get("rng_mps") is not None:
        try: torch.mps.set_rng_state(state["rng_mps"])
        except Exception: pass
            
    logger.info(f"Checkpoint loaded: {filepath}")
    return state.get("step", 0), state.get("epoch", 0)

def save_checkpoint(model, optimizer, step=0, loss=0.0, filepath="checkpoint.pt", **kwargs):
    return save_full_checkpoint(filepath, model, optimizer, step=step)

def load_checkpoint(filepath, model, optimizer, device="cpu"):
    step, _ = load_full_checkpoint(filepath, model, optimizer, device=device)
    return step
""")

# 2. Write training/train.py
with open("training/train.py", "w") as f:
    f.write("""import argparse
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
def evaluate_loss(model: nn.Module, val_dataset: Dataset, device: str, max_batches: int = 10) -> float:
    model.eval()
    if val_dataset is None or len(val_dataset) == 0: return 0.0
    loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    total_loss = 0.0
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y, use_cache=False)
        total_loss += loss.item()
        count += 1
        if count >= max_batches: break
    model.train()
    return total_loss / max(1, count)

def train_production_151m(
    config: GPTConfig = canonical_151m_config,
    train_dataset: Dataset = None,
    val_dataset: Dataset = None,
    max_steps: int = 1000,
    eval_interval: int = 100,
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
""")

# 3. Write tests/integration/test_checkpoint_resume.py
with open("tests/integration/test_checkpoint_resume.py", "w") as f:
    f.write("""import os
import pytest
import torch
from model.config import tiny_test_config
from model.transformer import GPT
from training.checkpointing import save_full_checkpoint, load_full_checkpoint

def test_checkpoint_resume_parity(tmp_path):
    device = "cpu"
    ckpt_path = str(tmp_path / "resume_ckpt.pt")
    
    torch.manual_seed(42)
    model1 = GPT(tiny_test_config).to(device)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=10)
    
    x = torch.randint(0, 300, (2, 16), device=device)
    y = torch.randint(0, 300, (2, 16), device=device)
    
    for step in range(3):
        optimizer1.zero_grad()
        _, loss1, _ = model1(x, targets=y, use_cache=False)
        loss1.backward()
        optimizer1.step()
        scheduler1.step()
        
    save_full_checkpoint(ckpt_path, model1, optimizer1, scheduler1, step=3, epoch=1)
    
    optimizer1.zero_grad()
    _, loss_step4_target, _ = model1(x, targets=y, use_cache=False)
    loss_step4_target.backward()
    optimizer1.step()
    scheduler1.step()
    target_lr = optimizer1.param_groups[0]["lr"]
    
    model2 = GPT(tiny_test_config).to(device)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=10)
    
    loaded_step, loaded_epoch = load_full_checkpoint(
        ckpt_path, model2, optimizer2, scheduler2, device=device
    )
    
    assert loaded_step == 3
    assert loaded_epoch == 1
    assert optimizer2.param_groups[0]["lr"] == scheduler1.get_last_lr()[0]
    
    optimizer2.zero_grad()
    _, loss_step4_resumed, _ = model2(x, targets=y, use_cache=False)
    loss_step4_resumed.backward()
    optimizer2.step()
    scheduler2.step()
    resumed_lr = optimizer2.param_groups[0]["lr"]
    
    assert torch.allclose(loss_step4_target, loss_step4_resumed, atol=1e-6), "Loss diverged! Resumed model did not match continuous model."
    assert target_lr == resumed_lr, "Scheduler failed to restore correct LR state."
""")

print("✅ Files generated successfully. Running tests...")
subprocess.run(["pytest", "tests/integration/test_checkpoint_resume.py", "-v", "-s"])
