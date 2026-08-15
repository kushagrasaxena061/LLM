import time
import pytest
import torch
from model.config import canonical_151m_config
from model.transformer import GPT

def run_benchmark(model, prompt_ids, max_new_tokens, use_cache, device):
    torch.manual_seed(42)
    past_key_values = None
    curr_input = prompt_ids
    tokens = prompt_ids.clone()
    
    t_start = time.perf_counter()
    t_first_token = None
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            if use_cache:
                logits, _, past_key_values = model(curr_input, past_key_values=past_key_values, use_cache=True)
                next_token = torch.argmax(logits[:, -1:, :], dim=-1)
                curr_input = next_token
            else:
                logits, _, _ = model(tokens, use_cache=False)
                next_token = torch.argmax(logits[:, -1:, :], dim=-1)
                tokens = torch.cat([tokens, next_token], dim=1)
                
            if step == 0:
                t_first_token = time.perf_counter()
                
    t_end = time.perf_counter()
    
    ttft_ms = (t_first_token - t_start) * 1000.0
    total_latency_sec = t_end - t_start
    itl_ms = ((t_end - t_first_token) / (max_new_tokens - 1)) * 1000.0 if max_new_tokens > 1 else 0.0
    tokens_per_sec = max_new_tokens / total_latency_sec
    
    return {
        "ttft_ms": ttft_ms,
        "total_latency_sec": total_latency_sec,
        "itl_ms": itl_ms,
        "tokens_per_sec": tokens_per_sec
    }

def test_kv_cache_performance_benchmark():
    """
    Empirically benchmarks KV-cache vs Uncached generation across standard sequence lengths.
    """
    device = "cpu"
    model = GPT(canonical_151m_config).to(device)
    model.eval()
    
    prompt_len = 32
    max_new_tokens = 30
    prompt_ids = torch.randint(0, canonical_151m_config.vocab_size, (1, prompt_len), device=device)
    
    # Warmup
    _ = run_benchmark(model, prompt_ids, 5, use_cache=True, device=device)
    
    # Run Uncached
    uncached_metrics = run_benchmark(model, prompt_ids, max_new_tokens, use_cache=False, device=device)
    
    # Run Cached
    cached_metrics = run_benchmark(model, prompt_ids, max_new_tokens, use_cache=True, device=device)
    
    print("\n" + "="*70)
    print(" 🚀 EMPIRICAL KV-CACHE PERFORMANCE BENCHMARK (Canonical 151M Model)")
    print("="*70)
    print(f" Prompt Length: {prompt_len} tokens | Generation Length: {max_new_tokens} tokens")
    print("-" * 70)
    print(f" Metric                 | Uncached (O(N²))    | Cached (O(N))       | Delta / Speedup")
    print("-" * 70)
    print(f" TTFT (Time-To-First)   | {uncached_metrics['ttft_ms']:8.2f} ms     | {cached_metrics['ttft_ms']:8.2f} ms     | {cached_metrics['ttft_ms'] - uncached_metrics['ttft_ms']:+.2f} ms")
    print(f" Inter-Token Latency    | {uncached_metrics['itl_ms']:8.2f} ms     | {cached_metrics['itl_ms']:8.2f} ms     | {uncached_metrics['itl_ms'] / cached_metrics['itl_ms']:.2f}x faster")
    print(f" Total Generation Time  | {uncached_metrics['total_latency_sec']:8.3f} s      | {cached_metrics['total_latency_sec']:8.3f} s      | {uncached_metrics['total_latency_sec'] / cached_metrics['total_latency_sec']:.2f}x faster")
    print(f" Generation Throughput  | {uncached_metrics['tokens_per_sec']:8.2f} tok/s   | {cached_metrics['tokens_per_sec']:8.2f} tok/s   | +{cached_metrics['tokens_per_sec'] - uncached_metrics['tokens_per_sec']:.2f} tok/s")
    print("="*70)
    
    # Assert KV-Cache delivers higher throughput on auto-regressive generation
    assert cached_metrics["tokens_per_sec"] > uncached_metrics["tokens_per_sec"], "KV Cache did not improve throughput!"
