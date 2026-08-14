# tests/performance/test_quantization_benchmark.py
"""Comprehensive benchmark comparing FP32, FP16, and INT8 model precision."""

import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
from model.config import GPTConfig
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from quantization.quantize import quantize_model_to_int8, get_model_size_mb
from evaluation.benchmark import InferenceBenchmark

def run_quantization_benchmark():
    print("Initializing Model Topologies for Precision Benchmarking...")
    
    # Use a moderately sized model configuration to make latency differences visible
    config = GPTConfig(vocab_size=260, context_length=128, d_model=128, n_layers=4, n_heads=4)

    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train("The quick brown fox jumps over the lazy dog. Optimization is critical for edge deployments.")

    prompt = "The quick brown fox"
    max_tokens = 20
    device = "cpu" # Forced to CPU for native PyTorch dynamic quantization support

    metrics_summary = []

    # 1. FP32 Baseline (Full Precision)
    print("Benchmarking FP32...")
    model_fp32 = GPT(config).to(device)
    size_fp32 = get_model_size_mb(model_fp32)
    bench_fp32 = InferenceBenchmark(model_fp32, tokenizer, device)
    res_fp32 = bench_fp32.run_benchmark(prompt, max_new_tokens=max_tokens)
    metrics_summary.append(("FP32 (Base)", size_fp32, res_fp32))

    # 2. FP16 (Half Precision)
    print("Benchmarking FP16...")
    model_fp16 = GPT(config).to(device).half()
    size_fp16 = get_model_size_mb(model_fp16)
    # PyTorch CPU backend sometimes struggles with native FP16 inference depending on architecture, 
    # but we cast the model to measure the physical size reduction regardless.
    try:
        bench_fp16 = InferenceBenchmark(model_fp16, tokenizer, device)
        res_fp16 = bench_fp16.run_benchmark(prompt, max_new_tokens=max_tokens)
    except Exception as e:
        res_fp16 = {"ttft_ms": 0.0, "itl_ms": 0.0, "tokens_per_second": 0.0}
        print(f"FP16 CPU inference skipped (hardware fallback): {e}")
    metrics_summary.append(("FP16 (Half)", size_fp16, res_fp16))

    # 3. INT8 (Dynamic Quantization)
    print("Benchmarking INT8...")
    model_int8 = quantize_model_to_int8(model_fp32)
    size_int8 = get_model_size_mb(model_int8)
    bench_int8 = InferenceBenchmark(model_int8, tokenizer, device)
    res_int8 = bench_int8.run_benchmark(prompt, max_new_tokens=max_tokens)
    metrics_summary.append(("INT8 (Quantized)", size_int8, res_int8))

    # 4. Generate QUANTIZATION_REPORT.md
    report_lines = [
        "# QUANTIZATION_REPORT.md\n",
        "## Empirical Precision Tradeoff Report\n",
        "This report compares the physical footprint and inference latency of varying numerical precisions.\n",
        "| Precision | Size (MB) | TTFT (ms) | ITL (ms) | Throughput (Tokens/sec) |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]

    print("\n✅ Quantization Benchmark Complete!\n")
    print(f"| Precision | Size (MB) | TTFT (ms) | ITL (ms) | Tokens/sec |")
    print(f"| :--- | :--- | :--- | :--- | :--- |")
    
    for name, size, res in metrics_summary:
        line = f"| **{name}** | {size:.2f} | {res['ttft_ms']:.2f} | {res['itl_ms']:.2f} | {res['tokens_per_second']:.2f} |"
        print(line)
        report_lines.append(line)

    report_lines.extend([
        "\n### Architectural Findings:",
        "- **Memory:** INT8 significantly reduces the model's physical footprint compared to FP32.",
        "- **Throughput:** While INT8 lowers memory bandwidth requirements, execution speed is highly dependent on specific CPU/GPU instruction sets for 8-bit math."
    ])

    report_path = Path(root_dir) / "QUANTIZATION_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

if __name__ == "__main__":
    run_quantization_benchmark()
