"""Unit tests for periodic validation loop and metrics."""
import os
import math
import pytest
import torch
from torch.utils.data import Dataset
from model.config import tiny_test_config
from model.transformer import GPT
import training.train
from training.train import evaluate_loss, train_production_151m

class DummyDataset(Dataset):
    def __init__(self, size): self.size = size
    def __len__(self): return self.size
    def __getitem__(self, idx): return torch.randint(0, 300, (16,)), torch.randint(0, 300, (16,))

def test_validation_gradient_and_mode_toggling():
    """Tests that model.eval() is used and gradients are disabled during validation."""
    model = GPT(tiny_test_config)
    model.train()
    val_dataset = DummyDataset(10)
    
    assert model.training, "Model should start in train mode."
    
    # Intercept forward pass to verify gradients and eval mode
    grad_status = []
    mode_status = []
    original_forward = model.forward
    
    def mock_forward(*args, **kwargs):
        grad_status.append(torch.is_grad_enabled())
        mode_status.append(model.training)
        return original_forward(*args, **kwargs)
        
    model.forward = mock_forward
    loss, ppl = evaluate_loss(model, val_dataset, "cpu", max_batches=2)
    model.forward = original_forward
    
    assert len(grad_status) == 2, "Should have run exactly 2 batches."
    assert not any(grad_status), "Gradients were explicitly left ENABLED during validation!"
    assert not any(mode_status), "Model was NOT in eval() mode during validation!"
    assert model.training, "Model failed to return to train() mode post-validation!"
    
    assert loss > 0.0, "Validation loss should be finite."
    assert ppl == math.exp(loss), "Perplexity calculation mismatch."

def test_validation_interval_execution(tmp_path):
    """Tests that the periodic validation triggers correctly and yields checkpoints."""
    train_ds = DummyDataset(20)
    val_ds = DummyDataset(10)
    ckpt_path = str(tmp_path / "periodic_ckpt.pt")
    
    # Spy on evaluate_loss
    call_counts = {"eval": 0}
    original_eval = training.train.evaluate_loss
    
    def mock_eval(*args, **kwargs):
        call_counts["eval"] += 1
        return original_eval(*args, **kwargs)
        
    training.train.evaluate_loss = mock_eval
    
    train_production_151m(
        config=tiny_test_config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        max_steps=5,
        eval_interval=2,
        eval_batches=1,
        batch_size=2,
        device="cpu",
        checkpoint_path=ckpt_path
    )
    
    training.train.evaluate_loss = original_eval
    
    # Over 5 steps (0, 1, 2, 3, 4) with interval 2, it should eval at 0, 2, and 4.
    assert call_counts["eval"] == 3, f"Expected 3 validation calls, got {call_counts[eval]}."
    assert os.path.exists(ckpt_path), "Checkpoint was not persisted during the periodic loop!"
