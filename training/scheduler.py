# training/scheduler.py
"""Learning Rate Scheduler with Linear Warmup and Cosine Decay."""

import math


class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float):
        """
        Args:
            optimizer: The PyTorch optimizer.
            warmup_steps: Number of steps to linearly increase LR.
            max_steps: Total number of training steps.
            max_lr: The peak learning rate.
            min_lr: The final learning rate at the end of training.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        """Updates the optimizer's learning rate based on the current step."""
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self) -> float:
        """Calculates the exact learning rate for the current step."""
        # 1. Linear Warmup Phase
        if self.current_step < self.warmup_steps:
            return self.max_lr * (self.current_step / self.warmup_steps)
        
        # 2. End of training (hold at minimum LR)
        if self.current_step > self.max_steps:
            return self.min_lr
            
        # 3. Cosine Decay Phase
        decay_ratio = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        
        # Math: 0.5 * (1.0 + cos(pi * ratio)) scales from 1.0 down to 0.0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
