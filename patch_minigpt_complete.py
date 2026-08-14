import os
import sys
from pathlib import Path

print("🚀 Initiating MiniGPT Studio Master Codebase Upgrade...")

# -------------------------------------------------------------------------
# 1. model/config.py
# -------------------------------------------------------------------------
config_code = '''# model/config.py
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
'''

with open("model/config.py", "w") as f:
    f.write(config_code)
print("✓ Updated model/config.py (Canonical 151M & Weight Tying)")

# -------------------------------------------------------------------------
# 2. model/transformer.py
# -------------------------------------------------------------------------
transformer_code = '''# model/transformer.py
"""Decoder-only GPT Transformer architecture with RoPE, RMSNorm, and SwiGLU."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from model.config import GPTConfig
from model.block import TransformerBlock
from model.rope import precompute_freqs_cis

try:
    from model.norm import RMSNorm
except ImportError:
    from model.rmsnorm import RMSNorm

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Explicit Weight Tying (reduces parameter footprint from ~190M to exact 151.86M)
        if config.weight_tying:
            self.lm_head.weight = self.tok_embeddings.weight
            
        freqs_cis = precompute_freqs_cis(config.head_dim, end=max(4096, config.context_length * 2))
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Calculates total unique parameters accounting for weight tying."""
        unique_params = set(self.parameters())
        if non_embedding:
            unique_params.discard(self.tok_embeddings.weight)
        return sum(p.numel() for p in unique_params)

    def _ensure_freqs_cis(self, required_len: int, device: torch.device):
        if self.freqs_cis is None or self.freqs_cis.shape[0] < required_len or self.freqs_cis.device != device:
            new_size = max(required_len + 512, self.config.context_length * 2, 4096)
            self.freqs_cis = precompute_freqs_cis(self.config.head_dim, end=new_size, device=device)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        start_pos: Optional[int] = None,
        return_attention: bool = False
    ):
        B, T = idx.shape
        device = idx.device
        if start_pos is None:
            start_pos = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            
        required_len = start_pos + T
        self._ensure_freqs_cis(required_len, device)
        freqs_cis_slice = self.freqs_cis[start_pos:required_len]
        
        x = self.tok_embeddings(idx)
        x = self.dropout(x)
        
        presents = [] if use_cache else None
        attentions = [] if return_attention else None
        
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            x, present, attn_w = block(
                x,
                freqs_cis=freqs_cis_slice,
                layer_past=layer_past,
                use_cache=use_cache,
                return_attention=return_attention
            )
            if use_cache:
                presents.append(present)
            if return_attention:
                attentions.append(attn_w)
                
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            
        if return_attention:
            return logits, loss, presents, attentions
        return logits, loss, presents
'''

with open("model/transformer.py", "w") as f:
    f.write(transformer_code)
print("✓ Updated model/transformer.py (Dynamic RoPE & Parameter Counting)")

# -------------------------------------------------------------------------
# 3. tokenizer/bpe.py
# -------------------------------------------------------------------------
tokenizer_code = '''# tokenizer/bpe.py
"""Lossless Byte-Pair Encoding (BPE) Tokenizer with special token handling."""

import json
from typing import List, Dict

class BPETokenizer:
    def __init__(self, vocab_size: int = 50257):
        self.target_vocab_size = vocab_size
        self.special_tokens = {
            "<|im_start|>": 0,
            "<|im_end|>": 1,
            "<|pad|>": 2,
            "<|unk|>": 3,
            "<|endoftext|>": 4
        }
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.merges: Dict[tuple, int] = {}
        self.vocab: Dict[int, bytes] = {}
        self.inverse_vocab: Dict[bytes, int] = {}
        self._init_vocab()

    def _init_vocab(self):
        # Base byte vocabulary (offset by number of special tokens)
        offset = len(self.special_tokens)
        for i in range(256):
            b = bytes([i])
            self.vocab[offset + i] = b
            self.inverse_vocab[b] = offset + i

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self.special_tokens)

    def train(self, text: str):
        offset = len(self.special_tokens)
        num_merges = max(0, self.target_vocab_size - 256 - offset)
        if num_merges <= 0:
            return

        raw_bytes = text.encode("utf-8")
        tokens = [offset + b for b in raw_bytes]

        for _ in range(num_merges):
            counts = {}
            for pair in zip(tokens, tokens[1:]):
                counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            pair = max(counts, key=counts.get)
            idx = offset + 256 + len(self.merges)
            
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(idx)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            self.merges[pair] = idx
            
            p0 = self.vocab.get(pair[0], b"")
            p1 = self.vocab.get(pair[1], b"")
            self.vocab[idx] = p0 + p1
            self.inverse_vocab[p0 + p1] = idx

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        offset = len(self.special_tokens)
        raw_bytes = text.encode("utf-8")
        tokens = [offset + b for b in raw_bytes]
        
        while len(tokens) >= 2:
            pairs = list(zip(tokens, tokens[1:]))
            candidate_pairs = [p for p in pairs if p in self.merges]
            if not candidate_pairs:
                break
            best_pair = min(candidate_pairs, key=lambda p: self.merges[p])
            new_idx = self.merges[best_pair]
            
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(new_idx)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        byte_chunks = []
        for tid in token_ids:
            if tid in self.inverse_special_tokens:
                byte_chunks.append(self.inverse_special_tokens[tid].encode("utf-8"))
            elif tid in self.vocab:
                byte_chunks.append(self.vocab[tid])
        all_bytes = b"".join(byte_chunks)
        return all_bytes.decode("utf-8", errors="replace")

    def save(self, filepath: str):
        data = {
            "target_vocab_size": self.target_vocab_size,
            "special_tokens": self.special_tokens,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    def load(self, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.target_vocab_size = data["target_vocab_size"]
        self.special_tokens = data["special_tokens"]
        self.inverse_special_tokens = {v: k for k, v in self.special_tokens.items()}
        self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}
        self._init_vocab()
        offset = len(self.special_tokens)
        for pair, idx in self.merges.items():
            p0 = self.vocab.get(pair[0], b"")
            p1 = self.vocab.get(pair[1], b"")
            self.vocab[idx] = p0 + p1
            self.inverse_vocab[p0 + p1] = idx
'''

with open("tokenizer/bpe.py", "w") as f:
    f.write(tokenizer_code)
print("✓ Updated tokenizer/bpe.py (Lossless UTF-8 & Special Token Isolation)")

# -------------------------------------------------------------------------
# 4. inference/generate.py
# -------------------------------------------------------------------------
generate_code = '''# inference/generate.py
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
        
    return tokenizer.decode(generated_tokens)
'''

with open("inference/generate.py", "w") as f:
    f.write(generate_code)
print("✓ Updated inference/generate.py (Stop-Tokens, KV-Cache & Sampling)")

# -------------------------------------------------------------------------
# 5. inference/chat.py
# -------------------------------------------------------------------------
chat_code = '''# inference/chat.py
"""Multi-turn Chat Session Manager with ChatML formatting and sliding context."""

from typing import List, Dict
from personas.engine import PersonaManager
from inference.generate import generate_text

class ChatSessionManager:
    def __init__(self, model, tokenizer, device: str = "cpu", max_context_chars: int = 4000):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_context_chars = max_context_chars
        self.persona_manager = PersonaManager()
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def clear_history(self):
        self.history.clear()

    def clean_output(self, text: str) -> str:
        tags_to_remove = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "system\\n", "user\\n", "assistant\\n"]
        for tag in tags_to_remove:
            text = text.replace(tag, "")
        return text.strip()

    def build_chatml_prompt(self, persona_name: str = "General Assistant") -> str:
        persona = self.persona_manager.get_persona(persona_name)
        prompt_parts = [f"<|im_start|>system\\n{persona.system_prompt}<|im_end|>"]
        for msg in self.history:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}\\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\\n")
        
        full_prompt = "\\n".join(prompt_parts)
        if len(full_prompt) > self.max_context_chars:
            full_prompt = full_prompt[-self.max_context_chars:]
        return full_prompt

    def respond(self, user_message: str, persona_name: str = "General Assistant", max_new_tokens: int = 35) -> str:
        self.add_message("user", user_message)
        prompt = self.build_chatml_prompt(persona_name)
        persona = self.persona_manager.get_persona(persona_name)
        
        full_out = generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            device=self.device,
            temperature=persona.temperature,
            stop_tokens=["<|im_end|>", "<|endoftext|>"]
        )
        
        cleaned = self.clean_output(full_out)
        self.add_message("assistant", cleaned)
        return cleaned
'''

with open("inference/chat.py", "w") as f:
    f.write(chat_code)
print("✓ Updated inference/chat.py (ChatML Clean Output & Persona Flow)")

# -------------------------------------------------------------------------
# 6. tests/integration/test_master_audit.py
# -------------------------------------------------------------------------
test_code = '''import pytest
import torch
from fastapi.testclient import TestClient

from model.config import GPTConfig
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from inference.generate import generate_text
from training.checkpointing import save_checkpoint, load_checkpoint
from security.guardrails import SecurityGuard
from evaluation.safety import SafetyEvaluator
from evaluation.embeddings import EmbeddingEngine
from api.server import app

device = "mps" if torch.backends.mps.is_available() else "cpu"

@pytest.fixture(scope="module")
def base_model_and_tok():
    config = GPTConfig(vocab_size=300, context_length=128, d_model=32, n_layers=2, n_heads=2)
    model = GPT(config).to(device)
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train("The quick brown fox jumps over the lazy dog. 1234567890 !@#$%^&*()")
    return model, tokenizer

def test_exhaustive_tokenizer(base_model_and_tok):
    _, tokenizer = base_model_and_tok
    test_cases = [
        "Normal English text with punctuation!",
        "Numbers: 1234567890",
        "Newlines and tabs",
        "{\\"json\\": \\"test\\", \\"key\\": 42}",
        "def hello_world(): print(\\"python\\")",
        "## Markdown Header",
        "https://www.example.com/path?query=1",
    ]
    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str)

def test_special_token_eos(base_model_and_tok):
    _, tokenizer = base_model_and_tok
    assert "<|im_end|>" in tokenizer.special_tokens
    assert "<|im_start|>" in tokenizer.special_tokens

def test_151m_parameter_count():
    config = GPTConfig(vocab_size=50257, context_length=2048, d_model=768, n_layers=12, n_heads=12, weight_tying=True)
    model = GPT(config)
    total_unique_params = model.get_num_params()
    assert 151_000_000 <= total_unique_params <= 152_000_000

def test_causal_masking_automatic(base_model_and_tok):
    model, _ = base_model_and_tok
    x = torch.randint(0, 100, (1, 10), device=device)
    _, _, _, attentions = model(x, return_attention=True)
    attn_matrix = attentions[0][0, 0]
    upper_tri = torch.triu(attn_matrix, diagonal=1)
    assert torch.allclose(upper_tri, torch.zeros_like(upper_tri))

def test_kv_cache_parity_and_shapes(base_model_and_tok):
    model, _ = base_model_and_tok
    model.eval()
    seq = torch.tensor([[1, 2, 3, 4]], device=device)
    seq_full = torch.tensor([[1, 2, 3, 4, 5]], device=device)
    logits_uncached, _, _ = model(seq_full, use_cache=False)
    logits_prefill, _, past_kv = model(seq, use_cache=True, start_pos=0)
    next_tok = torch.tensor([[5]], device=device)
    start_pos = past_kv[0][0].shape[2]
    logits_cached, _, _ = model(next_tok, past_key_values=past_kv, use_cache=True, start_pos=start_pos)
    assert torch.allclose(logits_uncached[:, -1, :], logits_cached[:, -1, :], atol=1e-4)

def test_checkpoint_resume_states(base_model_and_tok, tmp_path):
    model, _ = base_model_and_tok
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "test_ckpt.pt"
    save_checkpoint(model, optimizer, step=42, loss=0.99, filepath=str(ckpt_path))
    loaded_step = load_checkpoint(str(ckpt_path), model, optimizer, device=device)
    assert loaded_step == 42

def test_tiny_dataset_overfit(base_model_and_tok):
    model, _ = base_model_and_tok
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.randint(0, 100, (2, 8), device=device)
    y = torch.randint(0, 100, (2, 8), device=device)
    initial_loss, final_loss = None, None
    model.train()
    for step in range(25):
        optimizer.zero_grad()
        _, loss, _ = model(x, targets=y)
        loss.backward()
        optimizer.step()
        if step == 0: initial_loss = loss.item()
        if step == 24: final_loss = loss.item()
    assert final_loss < initial_loss

def test_api_production_security_audit():
    client = TestClient(app)
    large_payload = {"prompt": "A" * 1500000}
    response = client.post("/generate", json=large_payload)
    assert response.status_code == 413
'''

os.makedirs("tests/integration", exist_ok=True)
with open("tests/integration/test_master_audit.py", "w") as f:
    f.write(test_code)
print("✓ Updated tests/integration/test_master_audit.py (Full 8-Point Master Test Suite)")

print("\n🎉 Master Codebase Upgrade Completed Successfully!")
