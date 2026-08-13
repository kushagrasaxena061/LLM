# tests/test_training.py
"""Unit tests to verify gradient flow and optimization."""

import torch
from model.config import GPTConfig
from model.transformer import GPT
from training.trainer import LLMTrainer
from configs.base_config import env_config

def test_single_batch_overfit():
    """
    Proves that the model can successfully learn and update its weights 
    by forcing it to memorize a single tiny batch of data.
    """
    # 1. Setup a tiny model
    config = GPTConfig(
        vocab_size=100, 
        context_length=16, 
        d_model=32, 
        n_layers=2, 
        n_heads=2
    )
    model = GPT(config)
    
    # 2. Setup the AdamW Optimizer
    # We use a large learning rate (1e-2) here purely so it learns fast for the test.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    
    # 3. Setup the Trainer
    trainer = LLMTrainer(model=model, optimizer=optimizer, device=env_config.device)
    
    # 4. Create a single static, fake batch of data (Batch=2, SeqLen=16)
    torch.manual_seed(42) # Lock seed so data doesn't change
    x = torch.randint(0, config.vocab_size, (2, 16))
    y = torch.randint(0, config.vocab_size, (2, 16))
    
    # 5. Record the initial loss (before any training)
    initial_loss = trainer.train_step(x, y)
    print(f"\nInitial Loss (Step 0): {initial_loss:.4f}")
    
    # 6. Train on the exact same data for 40 steps
    final_loss = 0.0
    for step in range(1, 41):
        final_loss = trainer.train_step(x, y)
        if step % 10 == 0:
            print(f"Loss at Step {step}: {final_loss:.4f}")
            
    # 7. Assert that learning occurred mathematically
    # If the final loss isn't significantly smaller than the initial loss,
    # it means gradients are not flowing and the weights are not updating!
    assert final_loss < (initial_loss / 2), "Model failed to overfit the single batch!"
    
    print(f"✅ Overfitting Test Passed! The network is successfully learning. Loss dropped from {initial_loss:.4f} to {final_loss:.4f}")
