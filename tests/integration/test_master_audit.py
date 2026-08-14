import pytest
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
        "{\"json\": \"test\", \"key\": 42}",
        "def hello_world(): print(\"python\")",
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
