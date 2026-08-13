# tests/test_scaling_eval.py
"""Unit tests to verify 124M parameter scaling and advanced evaluation metrics."""

import torch
from configs.base_config import env_config
from model.config import GPTConfig
from model.transformer import GPT
from evaluation.advanced_metrics import AdvancedEvaluator

def test_124m_scaling_and_evaluation():
    """Verifies that the model scales to 124M parameters and evaluates correctly."""
    
    # 1. Initialize the 124M GPT-2 Base Configuration
    config = GPTConfig(
        vocab_size=50257,
        context_length=1024,
        d_model=768,
        n_layers=12,
        n_heads=12
    )
    model = GPT(config).to(env_config.device)
    
    # 2. Verify Parameter Target (100M <= parameters <= 300M)
    params = model.get_num_params()
    assert 100_000_000 <= params <= 300_000_000, f"Parameter count {params:,} is out of 100M-300M bounds!"
    
    # 3. Create dummy data loaders to simulate Train and Validation splits
    # We use a tiny batch size and sequence length strictly for CI/CD speed
    x = torch.randint(0, config.vocab_size, (1, 16))
    y = torch.randint(0, config.vocab_size, (1, 16))
    dummy_loader = [(x, y)]
    
    # 4. Execute the Advanced Evaluator
    evaluator = AdvancedEvaluator(model, env_config.device)
    metrics = evaluator.evaluate(train_loader=dummy_loader, val_loader=dummy_loader)
    
    # 5. Assert all required metrics exist
    required_keys = ["train_loss", "val_loss", "train_val_gap", "val_perplexity", "train_accuracy", "val_accuracy"]
    for key in required_keys:
        assert key in metrics, f"Missing required metric: {key}"
        
    print(f"\n✅ 124M Scaling & Advanced Evaluation Passed!")
    print(f"   - Total Parameters: {params:,}")
    print(f"   - Validation Loss: {metrics['val_loss']:.4f}")
    print(f"   - Next-Token Accuracy: {metrics['val_accuracy']*100:.2f}%")
    print(f"   - Train/Val Generalization Gap: {metrics['train_val_gap']:.4f}")
