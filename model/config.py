# model/config.py
"""Configuration class defining the hyperparameters of our Transformer."""

from pydantic import BaseModel, Field


class GPTConfig(BaseModel):
    """
    Defines the architecture size for our 100M-300M parameter model.
    These default values roughly mirror a standard 124M parameter GPT-2 base model.
    """
    vocab_size: int = Field(default=50257, description="Size of the tokenizer vocabulary")
    context_length: int = Field(default=1024, description="Maximum sequence length (context window)")
    
    # Architecture Dimensions
    d_model: int = Field(default=768, description="The dimension of the token embeddings (hidden size)")
    n_layers: int = Field(default=12, description="Number of Transformer decoder blocks")
    n_heads: int = Field(default=12, description="Number of attention heads")
    
    # Regularization
    dropout: float = Field(default=0.1, description="Dropout probability for regularization")
    
    # We use this property to dynamically calculate the size of each attention head.
    # If d_model is 768 and we have 12 heads, each head processes 64 dimensions (768 / 12).
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

# Instantiate our default configuration for testing
gpt_config = GPTConfig()
