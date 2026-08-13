# tests/test_prompt_optimizer.py
"""Unit tests for the Prompt Optimization Engine."""

from prompt_engineering.optimizer import PromptOptimizer
from tokenizer.bpe import BPETokenizer

def test_prompt_optimizer():
    """Verifies that the prompt optimizer correctly reduces token counts."""
    # Setup our BPE tokenizer
    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train("please could you kindly help me write a python script to parse json")
    
    optimizer = PromptOptimizer(tokenizer)
    
    bad_prompt = "Please could you kindly help me write a python script to parse json?"
    
    result = optimizer.optimize_prompt(bad_prompt)
    
    assert result["optimized_tokens"] < result["original_tokens"], "Optimizer failed to compress prompt!"
    assert result["tokens_saved"] > 0, "No tokens were saved!"
    
    print(f"\n✅ Prompt Optimization Successful!")
    print(f"   - Original: '{result['original_prompt']}' ({result['original_tokens']} tokens)")
    print(f"   - Optimized: '{result['optimized_prompt']}' ({result['optimized_tokens']} tokens)")
    print(f"   - Savings: {result['tokens_saved']} tokens ({result['savings_percentage']:.1f}%)")
