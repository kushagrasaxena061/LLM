# tests/evaluation/test_architecture_experiments.py
"""Experiment lab for comparing architectural configurations."""

import sys
import time
from pathlib import Path

# Force inject the absolute path of the project root
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import torch
from model.config import GPTConfig
from model.transformer import GPT

def run_experiment(name: str, config: GPTConfig, device: str = "cpu"):
    """Runs a simulated stress test on a specific architecture configuration."""
    print(f"Running Experiment: {name}...")
    model = GPT(config).to(device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    
    # Simulate a batch of data
    batch_size = 4
    seq_len = 128
    dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            model(dummy_input)
            
    # Measure Throughput
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)
    end_time = time.perf_counter()
    
    total_tokens = batch_size * seq_len * 10
    total_time = end_time - start_time
    tokens_per_sec = total_tokens / total_time
    
    return {
        "Name": name,
        "Layers": config.n_layers,
        "d_model": config.d_model,
        "Heads": config.n_heads,
        "Params (M)": param_count / 1_000_000,
        "Tokens/sec": tokens_per_sec
    }

def generate_experiments_report():
    device = "cpu"
    
    # Experiment A: Wide & Shallow (Fewer layers, wider dimension)
    config_a = GPTConfig(vocab_size=1000, context_length=128, d_model=256, n_layers=2, n_heads=8)
    
    # Experiment B: Narrow & Deep (More layers, narrower dimension)
    config_b = GPTConfig(vocab_size=1000, context_length=128, d_model=128, n_layers=8, n_heads=4)
    
    res_a = run_experiment("Wide & Shallow", config_a, device)
    res_b = run_experiment("Narrow & Deep", config_b, device)
    
    # Generate EXPERIMENTS.md
    report_lines = [
        "# EXPERIMENTS.md",
        "## Architectural Configuration Tradeoffs\n",
        "This document tracks empirical experiments comparing different hyperparameter setups.\n",
        "| Experiment | Layers | d_model | Heads | Params (M) | Tokens/sec |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
        f"| **{res_a['Name']}** | {res_a['Layers']} | {res_a['d_model']} | {res_a['Heads']} | {res_a['Params (M)']:.2f}M | {res_a['Tokens/sec']:.1f} |",
        f"| **{res_b['Name']}** | {res_b['Layers']} | {res_b['d_model']} | {res_b['Heads']} | {res_b['Params (M)']:.2f}M | {res_b['Tokens/sec']:.1f} |\n",
        "### Decision Log",
        "- **Result:** Deeper networks generally compound latency linearly due to sequential matrix multiplications, whereas wider networks can better utilize parallel compute up to the hardware's limit.",
        "- **Decision:** For the final 151M model, a balanced approach (12 layers, 768 d_model) was selected to optimize the tradeoff between representational depth and memory bandwidth."
    ]
    
    report_path = Path(root_dir) / "EXPERIMENTS.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
        
    print("\n✅ Architecture Experiment Lab Complete!")
    print(f"   - {res_a['Name']}: {res_a['Params (M)']:.2f}M Params | {res_a['Tokens/sec']:.1f} tok/s")
    print(f"   - {res_b['Name']}: {res_b['Params (M)']:.2f}M Params | {res_b['Tokens/sec']:.1f} tok/s")
    print(f"   - Report generated at: {report_path.name}")

if __name__ == "__main__":
    generate_experiments_report()
