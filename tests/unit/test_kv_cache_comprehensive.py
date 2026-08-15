import pytest
import torch
from model.config import tiny_test_config
from model.transformer import GPT

@pytest.fixture
def test_setup():
    device = "cpu"
    torch.manual_seed(42)
    model = GPT(tiny_test_config).to(device)
    model.eval()
    return model, device

def test_kv_cache_logit_and_token_parity(test_setup):
    """
    Verifies that cached incremental generation produces EXACT logit 
    and token parity against uncached full-sequence generation across all steps.
    """
    model, device = test_setup
    prompt_ids = torch.randint(0, tiny_test_config.vocab_size, (1, 8), device=device)
    max_new_tokens = 16

    # --- 1. Uncached Full-Context Generation ---
    uncached_tokens = prompt_ids.clone()
    uncached_step_logits = []
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _, _ = model(uncached_tokens, use_cache=False)
            last_logit = logits[:, -1:, :]
            uncached_step_logits.append(last_logit)
            next_token = torch.argmax(last_logit, dim=-1)
            uncached_tokens = torch.cat([uncached_tokens, next_token], dim=1)

    # --- 2. Cached Incremental Generation ---
    cached_tokens = prompt_ids.clone()
    cached_step_logits = []
    past_key_values = None
    curr_input = prompt_ids
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            logits, _, past_key_values = model(curr_input, past_key_values=past_key_values, use_cache=True)
            last_logit = logits[:, -1:, :]
            cached_step_logits.append(last_logit)
            next_token = torch.argmax(last_logit, dim=-1)
            cached_tokens = torch.cat([cached_tokens, next_token], dim=1)
            curr_input = next_token

    # --- 3. Assert Exact Parity ---
    assert torch.equal(uncached_tokens, cached_tokens), "Cached tokens diverged from uncached tokens!"
    
    for i, (u_logit, c_logit) in enumerate(zip(uncached_step_logits, cached_step_logits)):
        assert torch.allclose(u_logit, c_logit, atol=1e-4, rtol=1e-4), f"Logit mismatch at step {i}!"
    print(f"\n✅ Step-by-step logit parity verified across {max_new_tokens} autoregressive steps.")

def test_kv_cache_shapes_and_offsets(test_setup):
    """
    Verifies that cache shapes grow strictly monotonically by 1 per step
    and respect [B, n_heads, seq_len, head_dim].
    """
    model, device = test_setup
    prompt_len = 6
    prompt_ids = torch.randint(0, tiny_test_config.vocab_size, (2, prompt_len), device=device)
    
    past_key_values = None
    curr_input = prompt_ids
    
    with torch.no_grad():
        # Prefill Phase
        _, _, past_key_values = model(curr_input, past_key_values=None, use_cache=True)
        assert len(past_key_values) == tiny_test_config.n_layers
        for k, v in past_key_values:
            assert k.shape == (2, tiny_test_config.n_heads, prompt_len, tiny_test_config.head_dim)
            assert v.shape == (2, tiny_test_config.n_heads, prompt_len, tiny_test_config.head_dim)
            
        # Incremental Steps Phase
        curr_input = torch.randint(0, tiny_test_config.vocab_size, (2, 1), device=device)
        for step in range(1, 6):
            expected_len = prompt_len + step
            _, _, past_key_values = model(curr_input, past_key_values=past_key_values, use_cache=True)
            for k, v in past_key_values:
                assert k.shape == (2, tiny_test_config.n_heads, expected_len, tiny_test_config.head_dim)
                assert v.shape == (2, tiny_test_config.n_heads, expected_len, tiny_test_config.head_dim)
                
    print("\n✅ KV-Cache tensor shapes and positional increment tracking verified.")

def test_cache_reset_isolation(test_setup):
    """
    Ensures that starting a new generation with past_key_values=None does not leak state.
    """
    model, device = test_setup
    prompt_a = torch.randint(0, tiny_test_config.vocab_size, (1, 4), device=device)
    prompt_b = torch.randint(0, tiny_test_config.vocab_size, (1, 4), device=device)
    
    with torch.no_grad():
        # Session A
        logits_a1, _, cache_a = model(prompt_a, past_key_values=None, use_cache=True)
        
        # Session B (Reset)
        logits_b, _, cache_b = model(prompt_b, past_key_values=None, use_cache=True)
        
        # Re-run Session A from clean reset
        logits_a2, _, _ = model(prompt_a, past_key_values=None, use_cache=True)
        
    assert torch.allclose(logits_a1, logits_a2, atol=1e-5), "Cache state leaked across requests!"
    assert not torch.allclose(logits_a1, logits_b, atol=1e-5), "Distinct prompts produced identical output!"
    print("\n✅ Cache session isolation and reset verified.")
