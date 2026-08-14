# model/config.py
"""Centralized, authoritative configuration for the MiniGPT 151M architecture."""

from pydantic import BaseModel, Field

class GPTConfig(BaseModel):
    vocab_size: int = Field(default=50257, description="Tokenizer vocabulary size")
    context_length: int = Field(default=2048, description="Maximum sequence length")
    
    # Architecture Dimensions (Canonical 151M Target)
    d_model: int = Field(default=768, description="Hidden embedding dimension")
    n_layers: int = Field(default=12, description="Number of Transformer blocks")
    n_heads: int = Field(default=12, description="Number of attention heads")
    
    # Modern Transformer Enhancements
    dropout: float = Field(default=0.1, description="Dropout rate")
    weight_tying: bool = Field(default=True, description="Tie token embedding and LM head weights")
    
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

# Canonical production configuration instance (~151.86M parameters)
canonical_151m_config = GPTConfig()
