# evaluation/benchmark.py
"""Inference benchmarking engine for TTFT, ITL, and Throughput."""

import time
import torch
from utils.logger import get_logger

logger = get_logger(__name__)

class InferenceBenchmark:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def run_benchmark(self, prompt: str, max_new_tokens: int = 20) -> dict:
        self.model.eval()
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        prompt_length = input_ids.shape[1]

        if str(self.device) == 'cuda':
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        # 1. Prefill Phase (Generate First Token / TTFT)
        logits, _, past_key_values = self.model(input_ids, use_cache=True)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if str(self.device) == 'cuda':
            torch.cuda.synchronize()

        ttft_time = time.perf_counter()
        ttft = (ttft_time - start_time) * 1000

        # 2. Decode Phase (Generate Subsequent Tokens / ITL)
        decode_start_time = time.perf_counter()

        for _ in range(max_new_tokens - 1):
            logits, _, past_key_values = self.model(next_token, past_key_values=past_key_values, use_cache=True)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        if str(self.device) == 'cuda':
            torch.cuda.synchronize()

        decode_end_time = time.perf_counter()

        total_time = decode_end_time - start_time
        decode_time = decode_end_time - decode_start_time

        itl = (decode_time / (max_new_tokens - 1)) * 1000 if max_new_tokens > 1 else 0.0
        tps = max_new_tokens / total_time if total_time > 0 else 0.0

        metrics = {
            "prompt_tokens": prompt_length,
            "generated_tokens": max_new_tokens,
            "ttft_ms": ttft,
            "itl_ms": itl,
            "tokens_per_second": tps,
            "total_latency_sec": total_time
        }

        logger.info("Inference Benchmark Complete", **metrics)
        return metrics
