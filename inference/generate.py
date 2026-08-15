"""Autoregressive text generation with KV-cache and ChatML handling."""
import torch
from typing import List, Optional

@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    stop_tokens: Optional[List[int]] = None,
    device: str = "cpu",
    return_full_text: bool = False
) -> str:
    model.eval()
    was_training = model.training
    
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    # Resolve stop tokens dynamically (automatically fetch EOS and ChatML endings)
    if stop_tokens is None:
        stop_tokens = []
        if hasattr(tokenizer, "special_tokens"):
            if "<|endoftext|>" in tokenizer.special_tokens:
                stop_tokens.append(tokenizer.special_tokens["<|endoftext|>"])
            if "<|im_end|>" in tokenizer.special_tokens:
                stop_tokens.append(tokenizer.special_tokens["<|im_end|>"])

    generated_tokens = []
    past_key_values = None
    curr_input = input_ids
    
    for _ in range(max_new_tokens):
        logits, _, past_key_values = model(curr_input, past_key_values=past_key_values, use_cache=True)
        next_token_logits = logits[:, -1, :]
        
        # Deterministic greedy decoding
        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            # Temperature scaling
            next_token_logits = next_token_logits / temperature
            
            # Top-K
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
            # Top-P
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        token_val = next_token.item()
        
        # Strict EOS / Stop Token enforcement
        if token_val in stop_tokens:
            break
            
        generated_tokens.append(token_val)
        curr_input = next_token
        
    if was_training:
        model.train()
        
    # Dedicated assistant-response extraction mechanism:
    # Decode only the generated tokens, entirely avoiding the prompt 
    # (and its ChatML tags) from leaking into the output.
    if return_full_text:
        return tokenizer.decode(token_ids + generated_tokens)
    else:
        clean_response = tokenizer.decode(generated_tokens)
        # Failsafe against model hallucinating structural tags
        for st in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
            clean_response = clean_response.replace(st, "")
        return clean_response.strip()
