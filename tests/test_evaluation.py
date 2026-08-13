# tests/test_evaluation.py
"""Unit tests for automated evaluation metrics and scorecards."""

import torch
from configs.base_config import env_config
from model.config import GPTConfig
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from data.dataset import create_dataloader
from evaluation.metrics import ModelEvaluator, calculate_perplexity

def test_perplexity_calculation():
    """Verifies perplexity math from a known loss value."""
    loss = 0.0
    assert calculate_perplexity(loss) == 1.0, "Perplexity of loss 0.0 must be 1.0"
    
    loss = float('inf')
    assert calculate_perplexity(loss) == float('inf'), "Infinite loss should yield infinite perplexity"

def test_evaluator_pipeline():
    """Verifies that ModelEvaluator correctly computes loss and perplexity."""
    config = GPTConfig(
        vocab_size=260,
        context_length=32,
        d_model=32,
        n_layers=2,
        n_heads=2
    )
    model = GPT(config).to(env_config.device)
    
    tokenizer = BPETokenizer(vocab_size=260)
    corpus = "Evaluating language model performance requires robust benchmarks."
    tokenizer.train(corpus)
    
    # Create a small temporary file for dataloader testing
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as f:
        f.write(corpus * 10)
        temp_path = f.name
        
    dataloader = create_dataloader(temp_path, tokenizer, context_length=32, batch_size=2)
    
    evaluator = ModelEvaluator(model, env_config.device)
    metrics = evaluator.evaluate_loss_and_perplexity(dataloader)
    
    assert "validation_loss" in metrics, "Missing validation loss in evaluation metrics!"
    assert "perplexity" in metrics, "Missing perplexity in evaluation metrics!"
    
    print(f"\n✅ Evaluation Framework Test Passed!")
    print(f"   - Validation Loss: {metrics['validation_loss']:.4f}")
    print(f"   - Perplexity:      {metrics['perplexity']:.4f}")
