# tests/unit/test_chat_manager.py
"""Unit tests for stateful chat session manager."""

from model.config import GPTConfig
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from inference.chat import ChatSessionManager

def test_chat_session_history():
    config = GPTConfig(vocab_size=260, context_length=64, d_model=32, n_layers=2, n_heads=2)
    model = GPT(config)
    tokenizer = BPETokenizer(vocab_size=260)
    tokenizer.train("The quick brown fox jumps over the lazy dog.")
    
    chat = ChatSessionManager(model, tokenizer, device="cpu")
    chat.add_message("user", "Hello")
    chat.add_message("assistant", "Hi there")
    
    prompt = chat.build_chatml_prompt("General Assistant")
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user\nHello<|im_end|>" in prompt
    assert len(chat.history) == 2
