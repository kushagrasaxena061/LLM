import sys
from pathlib import Path

# Force inject the absolute path of the project root
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
from configs.base_config import env_config
from model.config import GPTConfig
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from evaluation.benchmark import InferenceBenchmark

def test_inference_latency_tracking():
    config = GPTConfig(vocab_size=260, context_length=64, d_model=32, n_layers=2, n_heads=2)
    model = GPT(config).to(env_config.device)

    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train("The quick brown fox jumps over the lazy dog. Benchmarking is critical for production deployment.")

    benchmark_engine = InferenceBenchmark(model, tokenizer, env_config.device)

    prompt = "The quick brown fox"
    metrics = benchmark_engine.run_benchmark(prompt, max_new_tokens=15)

    assert metrics["ttft_ms"] > 0
    assert metrics["itl_ms"] >= 0
    assert metrics["tokens_per_second"] > 0

    print("\n✅ Inference Benchmark Test Passed!")
    print(f"   - Time-To-First-Token (TTFT): {metrics['ttft_ms']:.2f} ms")
    print(f"   - Inter-Token Latency (ITL):  {metrics['itl_ms']:.2f} ms")
    print(f"   - Throughput:                 {metrics['tokens_per_second']:.2f} tokens/sec")
    print(f"   - Total Latency:              {metrics['total_latency_sec']:.3f} s")

if __name__ == "__main__":
    test_inference_latency_tracking()
