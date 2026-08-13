# tests/test_personas.py
"""Unit tests for the Persona Studio and ChatML formatting."""

from personas.engine import PersonaManager

def test_persona_formatting_and_temperature():
    """Verifies that system prompts are injected and temperatures are assigned correctly."""
    manager = PersonaManager()
    
    user_input = "Write a python script to reverse a string."
    persona_name = "Code Ninja"
    
    # 1. Apply the Persona
    formatted_prompt = manager.apply_persona(user_input, persona_name)
    persona_data = manager.get_persona(persona_name)
    
    # 2. Verify ChatML injection
    assert "<|im_start|>system" in formatted_prompt, "Missing system start tag!"
    assert "senior software engineer" in formatted_prompt, "System prompt text injection failed!"
    assert f"<|im_start|>user\n{user_input}<|im_end|>" in formatted_prompt, "User prompt injection failed!"
    assert "<|im_start|>assistant" in formatted_prompt, "Missing assistant trigger tag!"
    
    # 3. Verify precise temperature control
    assert persona_data.temperature == 0.2, f"Expected temperature 0.2 for Code Ninja, got {persona_data.temperature}"
    
    print(f"\n✅ Persona System Test Passed!")
    print(f"   - Selected Persona: {persona_data.name}")
    print(f"   - Enforced Temperature: {persona_data.temperature}")
    print(f"   - Final ChatML Prompt:\n{formatted_prompt}")
