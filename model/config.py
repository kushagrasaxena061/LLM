from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257
    context_length: int = 2048
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    head_dim: int = 64
    dropout: float = 0.0
    bias: bool = False
    weight_tying: bool = True  # EXPLICITLY DEFINED

    def __post_init__(self):
        # Mathematically enforce head_dim alignment
        self.head_dim = self.d_model // self.n_heads

# THE SINGLE SOURCE OF TRUTH FOR PRODUCTION
canonical_151m_config = GPTConfig(
    vocab_size=50257, context_length=2048, d_model=768, n_layers=12,
    n_heads=12, head_dim=64, dropout=0.0, bias=False, weight_tying=True
)

# STRICTLY FOR UNIT TESTING
tiny_test_config = GPTConfig(
    vocab_size=300, context_length=256, d_model=32, n_layers=2,
    n_heads=2, head_dim=16, dropout=0.0, bias=False, weight_tying=True
)
