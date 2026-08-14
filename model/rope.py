import torch
from typing import Tuple

def precompute_freqs_cis(dim: int, end: int = 4096, theta: float = 10000.0, device: str = 'cpu') -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    seq_len = xq_.shape[1]
    head_dim_half = xq_.shape[-1]
    
    freqs_view = freqs_cis[:seq_len].view(1, seq_len, 1, head_dim_half)
    xq_out = torch.view_as_real(xq_ * freqs_view).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_view).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)
