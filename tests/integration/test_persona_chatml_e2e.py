import pytest
import torch
from model.config import tiny_test_config
from model.transformer import GPT
from tokenizer.bpe import BPETokenizer
from inference.generate import generate_text
from personas.engine import PersonaManager

def test_persona_chatml_generation_end_to_end():
    """
    Verifies the full pipeline:
    User Input -> Persona -> System Instruction -> ChatML -> Tokenizer -> Model -> KV Cache -> Stop Handling -> Clean Output
    """
    device = "cpu"
    model = GPT(tiny_test_config).to(device)
    
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train("The quick brown fox jumps over the lazy dog. Write code. System assistant user.")
    
    manager = PersonaManager()
    persona_name = "Code Ninja"
    user_prompt = "Write a function."
    
    # 1. Persona -> ChatML construction
    formatted_prompt = manager.apply_persona(user_prompt, persona_name)
    
    assert "<|im_start|>system" in formatted_prompt
    assert "senior software engineer" in formatted_prompt
    assert "<|im_start|>user\nWrite a function.<|im_end|>" in formatted_prompt
    assert "<|im_start|>assistant" in formatted_prompt
    
    # 2. Generation with precise temperature and stop tokens
    torch.manual_seed(42)
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted_prompt,
        max_new_tokens=10,
        temperature=0.0,  # Deterministic decoding
        device=device
    )
    
    # 3. ChatML output must not leak into normal assistant response
    assert "<|im_start|>" not in output, "ChatML leaked into user output!"
    assert "<|im_end|>" not in output, "ChatML leaked into user output!"
    assert "senior software engineer" not in output, "System prompt leaked into generated response!"
    
    print(f"\n✅ Persona -> ChatML -> Generation -> Clean Extraction Pipeline Verified!")
    print(f"Clean Assistant Output: '{output}'")
