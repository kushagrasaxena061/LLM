# training/loop.py
"""Master pretraining loop execution and checkpointing."""

import os
import torch
import time
from typing import Optional
from torch.utils.data import DataLoader
from training.trainer import LLMTrainer
from training.scheduler import CosineWarmupScheduler
from utils.logger import get_logger

logger = get_logger(__name__)

def save_checkpoint(model, optimizer, step: int, loss: float, filepath: str):
    """Saves the model weights and optimizer state to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved", filepath=filepath, step=step, loss=f"{loss:.4f}")

def train_model(
    trainer: LLMTrainer,
    dataloader: DataLoader,
    scheduler: CosineWarmupScheduler,
    max_steps: int,
    eval_interval: int = 100,
    checkpoint_dir: str = "checkpoints"
):
    """
    The main pretraining loop.
    """
    logger.info("Initiating Pretraining Loop", max_steps=max_steps, batches_per_epoch=len(dataloader))
    
    step = 0
    model = trainer.model
    model.train()
    
    # We use an infinite loop over the dataloader until we hit our exact max_steps
    while step < max_steps:
        for x, y in dataloader:
            if step >= max_steps:
                break
                
            t0 = time.time()
            
            # 1. Execute a single forward/backward/update step
            loss = trainer.train_step(x, y)
            
            # 2. Update the learning rate
            scheduler.step()
            
            t1 = time.time()
            dt = t1 - t0
            
            # 3. Logging & Diagnostics
            if step % 10 == 0:
                current_lr = scheduler.get_lr()
                logger.info(
                    f"Step {step:05d}", 
                    loss=f"{loss:.4f}", 
                    lr=f"{current_lr:.2e}", 
                    time_ms=f"{dt*1000:.1f}"
                )
                
            # 4. Checkpointing
            if step > 0 and step % eval_interval == 0:
                save_checkpoint(
                    model, 
                    trainer.optimizer, 
                    step, 
                    loss, 
                    f"{checkpoint_dir}/step_{step}.pt"
                )
                
            step += 1

    # Save final checkpoint
    save_checkpoint(model, trainer.optimizer, step, loss, f"{checkpoint_dir}/final_model.pt")
    logger.info("Pretraining Complete.")
