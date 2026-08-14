# tests/performance/test_prompt_optimizer_benchmark.py
"""Benchmark script to validate prompt optimization token compression."""

import sys
from pathlib import Path

# Force inject the absolute path of the project root
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from tokenizer.bpe import BPETokenizer
from prompt_engineering.optimizer import PromptOptimizer

def generate_benchmark_report():
    # 1. Setup tokenizer with enough vocabulary for the test
    tokenizer = BPETokenizer(vocab_size=300)
    corpus = (
        "The quick brown fox jumps over the lazy dog. "
        "Please could you kindly help me write a python script to parse json. "
        "Can you explain quantum physics to me? I would like to know what is the capital of France. "
        "Write code."
    )
    tokenizer.train(corpus)
    
    optimizer = PromptOptimizer(tokenizer)
    
    # 2. Define the evaluation dataset
    test_prompts = [
        "Please could you kindly help me write a python script to parse json?",
        "Can you explain quantum physics to me?",
        "I would like to know what is the capital of France.",
        "Write code."  # This is already optimized and should stay identical
    ]
    
    # 3. Process and format the report
    report_lines = [
        "# PROMPT_OPTIMIZATION_REPORT.md\n",
        "## Token Compression Benchmark Results\n",
        "This report demonstrates the token reduction and cost-saving impact of the Prompt Optimization engine.\n"
    ]
    
    total_original = 0
    total_optimized = 0
    
    for p in test_prompts:
        res = optimizer.optimize_prompt(p)
        total_original += res["original_tokens"]
        total_optimized += res["optimized_tokens"]
        
        report_lines.append(f"### Test Prompt")
        report_lines.append(f"**Original:** `{res['original_prompt']}` ({res['original_tokens']} tokens)")
        report_lines.append(f"**Optimized:** `{res['optimized_prompt']}` ({res['optimized_tokens']} tokens)")
        report_lines.append(f"**Result:** Saved {res['tokens_saved']} tokens ({res['savings_percentage']:.1f}%)\n")
        
    # Aggregate Metrics
    overall_savings = ((total_original - total_optimized) / total_original) * 100
    report_lines.append("## Aggregate Performance")
    report_lines.append(f"- **Total Original Tokens:** {total_original}")
    report_lines.append(f"- **Total Optimized Tokens:** {total_optimized}")
    report_lines.append(f"- **Overall Compression:** {overall_savings:.2f}% token reduction")
        
    report_content = "\n".join(report_lines)
    
    # 4. Write to disk
    report_path = Path(root_dir) / "PROMPT_OPTIMIZATION_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report_content)
        
    print("\n✅ Prompt Optimization Benchmark Complete!")
    print(f"   - Tested {len(test_prompts)} standard prompts.")
    print(f"   - Overall Token Compression: {overall_savings:.2f}%")
    print(f"   - Official report generated at: {report_path.name}")

if __name__ == "__main__":
    generate_benchmark_report()
