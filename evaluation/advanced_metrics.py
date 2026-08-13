# evaluation/advanced_metrics.py
"""Advanced evaluation system for model scaling and performance tracking."""

import torch
import math
from utils.logger import get_logger

logger = get_logger(__name__)

class AdvancedEvaluator:
    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        logger.info("AdvancedEvaluator initialized")

    @torch.no_grad()
    def evaluate(self, train_loader, val_loader) -> dict:
        """
        Evaluates the model across both training and validation sets to compute
        perplexity, next-token accuracy, and the train/val generalization gap.
        """
        self.model.eval()
        
        def eval_split(loader):
            total_loss = 0.0
            total_correct = 0
            total_tokens = 0
            batches = 0
            
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                
                # FIX: Catch the 3rd return value (presents/KV cache) and ignore it with '_'
                logits, loss, _ = self.model(x, targets=y)
                
                if loss is not None:
                    total_loss += loss.item()
                    
                    # Calculate next-token accuracy
                    preds = torch.argmax(logits, dim=-1)
                    # Ignore the -100 padding/masking tokens
                    mask = y != -100
                    total_correct += (preds[mask] == y[mask]).sum().item()
                    total_tokens += mask.sum().item()
                    
                    batches += 1
            
            avg_loss = total_loss / max(batches, 1)
            accuracy = total_correct / max(total_tokens, 1)
            return avg_loss, accuracy

        logger.info("Evaluating training split...")
        train_loss, train_acc = eval_split(train_loader)
        
        logger.info("Evaluating validation split...")
        val_loss, val_acc = eval_split(val_loader)
        
        # Calculate Perplexity (handling infinity for exploding loss)
        try:
            val_perplexity = math.exp(val_loss)
        except OverflowError:
            val_perplexity = float('inf')
            
        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_val_gap": val_loss - train_loss,
            "val_perplexity": val_perplexity,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        }
        
        logger.info("Advanced evaluation complete", metrics=metrics)
        return metrics
