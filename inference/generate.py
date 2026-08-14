# inference/generate.py
"""Autoregressive text generation engine with KV cache and stop-token support."""

import torch
from typing import List, Optional
import torch.nn.functional as F

@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    device: str = "cpu",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_tokens: Optional[List[str]] = None,
    use_cache: bool = True
) -> str:
    model.eval()
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [0]
        
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    past_key_values = None
    generated_tokens = []
    
    stop_token_ids = []
    if stop_tokens:
        for st in stop_tokens:
            if st in tokenizer.special_tokens:
                stop_token_ids.append(tokenizer.special_tokens[st])
            else:
                encoded_st = tokenizer.encode(st)
                if encoded_st:
                    stop_token_ids.append(encoded_st[0])

    for step in range(max_new_tokens):
        if use_cache:
            if past_key_values is None:
                logits, _, past_key_values = model(idx, use_cache=True, start_pos=0)
            else:
                start_pos = past_key_values[0][0].shape[2]
                logits, _, past_key_values = model(idx[:, -1:], past_key_values=past_key_values, use_cache=True, start_pos=start_pos)
        else:
            logits, _, _ = model(idx, use_cache=False)
            
        next_token_logits = logits[:, -1, :]
        
        if temperature <= 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token_logits = next_token_logits / temperature
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
        token_id = next_token.item()
        if token_id in stop_token_ids:
            break
            
        generated_tokens.append(token_id)
        idx = torch.cat([idx, next_token], dim=1)
        
    return tokenizer.decode(token_ids + generated_tokens)
