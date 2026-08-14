# training/trainer.py
"""Optimization engine and training loop for the LLM."""

import torch
import torch.nn as nn
from utils.logger import get_logger

logger = get_logger(__name__)

class LLMTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        device: str,
        max_grad_norm: float = 1.0
    ):
        """
        Initializes the trainer.
        
        Args:
            model: The GPT transformer model.
            optimizer: The PyTorch optimizer (usually AdamW).
            device: 'cuda', 'mps', or 'cpu'.
            max_grad_norm: The maximum allowed gradient size (for clipping).
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.max_grad_norm = max_grad_norm
        
        # GradScaler is required for Mixed Precision (AMP) training on CUDA.
        # It prevents underflow by multiplying gradients by a scale factor.
        # (Note: MPS on Mac doesn't fully support GradScaler yet, so it stays inactive there).
        self.scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

        
        logger.info("Trainer initialized", device=device, max_grad_norm=max_grad_norm)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Executes a single step of training: Forward -> Backward -> Update.
        """
        # 1. Move data to the correct hardware (GPU/MPS/CPU)
        x, y = x.to(self.device), y.to(self.device)
        
        # 2. Clear the old gradients from the previous step
        self.optimizer.zero_grad(set_to_none=True)
        
        # 3. Forward Pass with Automatic Mixed Precision (AMP)
        # This context manager tells PyTorch to use 16-bit math where safe.
        # (autocast for MPS is technically under torch.autocast, but we handle it safely here)
        device_type = "cuda" if self.device == "cuda" else ("mps" if self.device == "mps" else "cpu")
        
        # For simplicity and broad compatibility across Mac/Windows/Linux:
        # We only strictly enforce AMP on CUDA where it is definitively stable.
        if device_type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits, loss, _ = self.model(x, y)
        else:
            logits, loss, _ = self.model(x, y)

        # 4. Backward Pass (Calculate Gradients)
        if device_type == "cuda":
            self.scaler.scale(loss).backward()
            
            # Unscale the gradients before clipping so we clip the true values
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # 5. Optimizer Step (Update the weights)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard backward pass for Mac (MPS) and CPU
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # 5. Optimizer Step
            self.optimizer.step()

        return loss.item()
