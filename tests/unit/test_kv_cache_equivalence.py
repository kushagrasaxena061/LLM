# tests/unit/test_kv_cache_equivalence.py
"""Equivalence test proving KV Cache inference produces identical logits to full-sequence forward passes."""

import torch
from configs.base_config import env_config
from model.config import GPTConfig
from model.transformer import GPT

@torch.no_grad()
def test_kv_cache_exact_equivalence():
    """
    CRITICAL INFERENCE TEST:
    Autoregressively generates tokens using:
      Method A: Standard full-sequence forward pass (no cache, O(N^2))
      Method B: Incremental forward pass with KV-Cache (O(N))
    Verifies that the generated token sequences and step logits match within numerical tolerance.
    """
    config = GPTConfig(vocab_size=200, context_length=64, d_model=64, n_layers=4, n_heads=4)
    model = GPT(config).to(env_config.device)
    model.eval()
    
    prompt = torch.tensor([[12, 45, 67, 89]], dtype=torch.long, device=env_config.device)
    max_new_tokens = 8
    
    # -------------------------------------------------------------
    # Method A: Naive Full-Sequence Generation (No KV Cache)
    # -------------------------------------------------------------
    seq_no_cache = prompt.clone()
    logits_history_no_cache = []
    
    for _ in range(max_new_tokens):
        logits, _, _ = model(seq_no_cache, use_cache=False)
        next_token_logits = logits[:, -1, :]
        logits_history_no_cache.append(next_token_logits)
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        seq_no_cache = torch.cat((seq_no_cache, next_token), dim=1)
        
    # -------------------------------------------------------------
    # Method B: KV-Cache Incremental Generation
    # -------------------------------------------------------------
    seq_cached = prompt.clone()
    past_key_values = None
    logits_history_cached = []
    
    # Initial prefill pass
    logits, _, past_key_values = model(seq_cached, past_key_values=None, use_cache=True)
    next_token_logits = logits[:, -1, :]
    logits_history_cached.append(next_token_logits)
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    seq_cached = torch.cat((seq_cached, next_token), dim=1)
    
    # Incremental generation steps
    for _ in range(max_new_tokens - 1):
        logits, _, past_key_values = model(next_token, past_key_values=past_key_values, use_cache=True)
        next_token_logits = logits[:, -1, :]
        logits_history_cached.append(next_token_logits)
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        seq_cached = torch.cat((seq_cached, next_token), dim=1)

    # -------------------------------------------------------------
    # Assertions
    # -------------------------------------------------------------
    # 1. Output token sequence equivalence
    assert torch.equal(seq_no_cache, seq_cached), (
        f"KV-Cache sequence drifted from ground truth!\nNo-Cache: {seq_no_cache.tolist()}\nCached:   {seq_cached.tolist()}"
    )
    
    # 2. Logit-level numerical equivalence
    for step, (l_no_cache, l_cached) in enumerate(zip(logits_history_no_cache, logits_history_cached)):
        max_diff = (l_no_cache - l_cached).abs().max().item()
        assert max_diff < 1e-4, f"Logit divergence at step {step}: max diff = {max_diff}"
        
    print("\n✅ KV-Cache Exact Numerical Equivalence Verified Across All Generation Steps!")
