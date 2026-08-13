# evaluation/metrics.py
"""Automated evaluation metrics including Perplexity and generation diagnostics."""

import torch
import math
from utils.logger import get_logger

logger = get_logger(__name__)

def calculate_perplexity(loss: float) -> float:
    """
    Calculates Perplexity from cross-entropy loss.
    PPL = exp(loss)
    """
    try:
        ppl = math.exp(loss)
    except OverflowError:
        ppl = float('inf')
        
    return ppl

class ModelEvaluator:
    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        logger.info("ModelEvaluator initialized")

    @torch.no_grad()
    def evaluate_loss_and_perplexity(self, dataloader) -> dict:
        """
        Evaluates average validation loss and perplexity across a dataset loader.
        """
        self.model.eval()
        total_loss = 0.0
        total_batches = 0
        
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, targets=y)
            if loss is not None:
                total_loss += loss.item()
                total_batches += 1
                
        avg_loss = total_loss / max(total_batches, 1)
        perplexity = calculate_perplexity(avg_loss)
        
        metrics = {
            "validation_loss": avg_loss,
            "perplexity": perplexity
        }
        
        logger.info("Evaluation complete", metrics=metrics)
        return metrics
