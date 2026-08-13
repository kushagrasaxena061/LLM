# prompt_engineering/optimizer.py
"""Prompt Optimization Engine to reduce token usage and latency."""

import re
from typing import Dict
from tokenizer.base import BaseTokenizer

class PromptOptimizer:
    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer
        # A basic heuristic list of words that rarely add semantic value to LLM instructions
        self.stop_words = {
            "please", "can", "you", "could", "would", "kindly", "just", 
            "help", "me", "with", "a", "an", "the", "to", "of", "and", "in"
        }

    def analyze_prompt(self, raw_prompt: str) -> Dict:
        """Analyzes a prompt to find optimization opportunities."""
        original_tokens = self.tokenizer.encode(raw_prompt)
        
        return {
            "original_prompt": raw_prompt,
            "original_token_count": len(original_tokens),
            "word_count": len(raw_prompt.split())
        }

    def optimize_prompt(self, raw_prompt: str) -> Dict:
        """
        Strips unnecessary words and formatting to compress the prompt 
        while retaining semantic intent.
        """
        analysis = self.analyze_prompt(raw_prompt)
        
        # 1. Lowercase and strip punctuation for heuristic matching
        clean_words = re.sub(r'[^\w\s]', '', raw_prompt).lower().split()
        
        # 2. Rebuild prompt keeping only semantically heavy words (very naive heuristic for demo)
        optimized_words = [word for word in raw_prompt.split() if word.lower().strip(',.!?') not in self.stop_words]
        optimized_prompt = " ".join(optimized_words)
        
        if not optimized_prompt:  # Fallback if we stripped everything
            optimized_prompt = raw_prompt
            
        optimized_tokens = self.tokenizer.encode(optimized_prompt)
        
        # Calculate savings
        tokens_saved = analysis["original_token_count"] - len(optimized_tokens)
        savings_percentage = (tokens_saved / analysis["original_token_count"]) * 100 if analysis["original_token_count"] > 0 else 0
        
        return {
            "original_prompt": raw_prompt,
            "optimized_prompt": optimized_prompt,
            "original_tokens": analysis["original_token_count"],
            "optimized_tokens": len(optimized_tokens),
            "tokens_saved": tokens_saved,
            "savings_percentage": savings_percentage
        }
