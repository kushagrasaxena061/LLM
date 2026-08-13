# inference/generate.py
"""Basic Autoregressive Generation Loop."""

import torch
import torch.nn.functional as F
from tokenizer.base import BaseTokenizer
from model.transformer import GPT

@torch.no_grad() # Disable gradient tracking to save massive amounts of memory
def generate_text(
    model: GPT, 
    tokenizer: BaseTokenizer, 
    prompt: str, 
    max_new_tokens: int, 
    device: str,
    temperature: float = 1.0,
) -> str:
    """
    Takes a text prompt and generates subsequent tokens autoregressively.
    """
    model.eval() # Set model to evaluation mode (disables dropout)
    
    # 1. Encode the prompt into integer IDs
    ids = tokenizer.encode(prompt)
    idx = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0) # Shape: (1, SeqLen)
    
    # 2. Autoregressive Loop
    for _ in range(max_new_tokens):
        # Crop context to the maximum allowed by the model
        idx_cond = idx[:, -model.config.context_length:]
        
        # Forward pass (Get logits for the sequence)
        logits, _ = model(idx_cond)
        
        # Pluck out the logits for the very last token
        logits = logits[:, -1, :] # Shape: (1, vocab_size)
        
        # Apply temperature (Higher = more random, Lower = more confident)
        logits = logits / temperature
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution
        next_token = torch.multinomial(probs, num_samples=1) # Shape: (1, 1)
        
        # Append the new token to our running sequence
        idx = torch.cat((idx, next_token), dim=1)
        
    # 3. Decode the final sequence back to text
    final_ids = idx[0].tolist()
    return tokenizer.decode(final_ids)
