# inference/benchmark.py
"""Benchmarking script to evaluate TTFT and Tokens/sec with and without KV Cache."""
# inference/benchmark.py
"""Benchmarking script to evaluate TTFT and Tokens/sec with and without KV Cache."""

import sys
from pathlib import Path

# Automatically add the project root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

import torch
import time
from model.config import GPTConfig
from model.transformer import GPT

# ... (keep the rest of the file exactly as it was)

import torch
import time
from model.config import GPTConfig
from model.transformer import GPT

@torch.no_grad()
def benchmark_generation(model, prompt_ids, max_new_tokens, use_cache: bool, device: str):
    model.eval()
    idx = prompt_ids.clone().to(device)
    past_key_values = None
    
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    # 1. Prefill Phase (Time To First Token)
    logits, _, past_key_values = model(idx, use_cache=use_cache)
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    idx = torch.cat((idx, next_token), dim=1)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    ttft = time.time() - start_time
    
    # 2. Decode Phase (Inter-Token Latency)
    decode_start = time.time()
    
    for _ in range(max_new_tokens - 1):
        if use_cache:
            # ONLY pass the newest token and the cache
            logits, _, past_key_values = model(next_token, past_key_values=past_key_values, use_cache=True)
        else:
            # Naive approach: Pass the ENTIRE sequence every time
            logits, _, _ = model(idx, use_cache=False)
            
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat((idx, next_token), dim=1)
        
    torch.cuda.synchronize() if device == 'cuda' else None
    decode_time = time.time() - decode_start
    
    tokens_per_sec = (max_new_tokens - 1) / decode_time
    return ttft, tokens_per_sec

def run_benchmark(device: str = "mps"):
    print(f"\n--- INFERENCE BENCHMARK ({device.upper()}) ---")
    # Use a larger config to make the performance difference obvious
    config = GPTConfig(vocab_size=1000, context_length=512, d_model=256, n_layers=6, n_heads=8)
    model = GPT(config).to(device)
    
    prompt_ids = torch.randint(0, config.vocab_size, (1, 64)) # 64-token prompt
    tokens_to_generate = 100
    
    # Warmup
    _ = benchmark_generation(model, prompt_ids, 10, use_cache=False, device=device)
    
    print("\nRunning Naive Autoregressive (NO CACHE)...")
    ttft_no_cache, tps_no_cache = benchmark_generation(model, prompt_ids, tokens_to_generate, False, device)
    
    print("Running KV-Cache Optimized (WITH CACHE)...")
    ttft_cache, tps_cache = benchmark_generation(model, prompt_ids, tokens_to_generate, True, device)
    
    print("\n✅ BENCHMARK RESULTS:")
    print(f"Without Cache -> TTFT: {ttft_no_cache*1000:.2f}ms | Throughput: {tps_no_cache:.2f} tokens/sec")
    print(f"With KV Cache -> TTFT: {ttft_cache*1000:.2f}ms | Throughput: {tps_cache:.2f} tokens/sec")
    print(f"🚀 Speedup Multiplier: {tps_cache / tps_no_cache:.2f}x faster")

if __name__ == "__main__":
    from configs.base_config import env_config
    run_benchmark(env_config.device)
