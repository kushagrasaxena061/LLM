# prompt_engineering/optimizer.py
"""Engine for compressing prompts to reduce token usage and latency."""

import re
from typing import Dict
from tokenizer.base import BaseTokenizer
from utils.logger import get_logger

logger = get_logger(__name__)

class PromptOptimizer:
    def __init__(self, tokenizer: BaseTokenizer):
        self.tokenizer = tokenizer
        # Common filler words that consume tokens but provide zero semantic value to an LLM
        self.stop_words = {
            "please", "could", "you", "kindly", "help", "me", 
            "a", "an", "the", "to", "is", "are", "am", "i", 
            "can", "will", "would", "tell", "know", "what"
        }

    def optimize_prompt(self, prompt: str) -> Dict[str, any]:
        """Strips filler words and returns token savings metrics."""
        original_tokens = len(self.tokenizer.encode(prompt))
        
        # Strip punctuation and lowercase for aggressive optimization
        clean_text = re.sub(r'[^\w\s]', '', prompt.lower())
        
        # Filter out the stop words
        words = clean_text.split()
        optimized_words = [w for w in words if w not in self.stop_words]
        optimized_prompt = " ".join(optimized_words)
        
        # Fallback: if the user typed nothing but stop words, keep original
        if not optimized_prompt:
            optimized_prompt = prompt
            
        optimized_tokens = len(self.tokenizer.encode(optimized_prompt))
        saved = original_tokens - optimized_tokens
        
        logger.info("Prompt optimized", original=original_tokens, optimized=optimized_tokens)
        
        return {
            "original_prompt": prompt,
            "original_tokens": original_tokens,
            "optimized_prompt": optimized_prompt,
            "optimized_tokens": optimized_tokens,
            "tokens_saved": saved,
            "savings_percentage": (saved / original_tokens * 100) if original_tokens > 0 else 0.0,
            "optimization_reason": "Stop-word and filler removal"
        }
