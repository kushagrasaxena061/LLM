# inference/generate.py
"""Autoregressive text generation engine with KV Cache support."""

import torch
from tokenizer.base import BaseTokenizer
from model.transformer import GPT

@torch.no_grad()
def generate_text(
    model: GPT, 
    tokenizer: BaseTokenizer, 
    prompt: str, 
    max_new_tokens: int, 
    device: str, 
    temperature: float = 0.7
) -> str:
    """
    Generates text using the trained model, utilizing KV caching for O(1) step latency.
    """
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    past_key_values = None
    
    for _ in range(max_new_tokens):
        # If we have a cache, we only need to pass the very last generated token
        if past_key_values is not None:
            idx_cond = idx[:, [-1]]
        else:
            idx_cond = idx
            
        # FIX: Unpack all 3 values (logits, loss, presents) returned by the upgraded model
        logits, _, past_key_values = model(idx_cond, past_key_values=past_key_values, use_cache=True)
        
        # Get the logits for the last token
        next_token_logits = logits[:, -1, :] / temperature
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        
        # Sample next token from the probability distribution
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to the sequence
        idx = torch.cat((idx, next_token), dim=1)
        
    # Decode the generated tokens back to a string
    generated_ids = idx[0].tolist()
    return tokenizer.decode(generated_ids)
