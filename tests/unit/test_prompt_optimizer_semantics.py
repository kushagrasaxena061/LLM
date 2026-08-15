import pytest

def mock_optimize_prompt(prompt: str) -> str:
    """Simulates the backend optimizer stripping fluff while keeping constraints."""
    stop_words = ["please", "could", "you", "kindly", "help", "me"]
    words = prompt.split()
    return " ".join([w for w in words if w.lower() not in stop_words])

def test_prompt_optimizer_preserves_semantics():
    """Verifies prompt compression does not delete critical negative constraints or parameters."""
    original = "Please could you kindly help me write a python script but DO NOT use the os module, set max_length to 50."
    optimized = mock_optimize_prompt(original)
    
    # Verify token reduction
    assert len(optimized.split()) < len(original.split()), "Prompt was not compressed."
    
    # Verify semantic preservation
    assert "DO NOT" in optimized, "CRITICAL FAILURE: Optimizer stripped a negative constraint!"
    assert "50" in optimized, "CRITICAL FAILURE: Optimizer stripped a numeric parameter!"
    assert "python" in optimized, "CRITICAL FAILURE: Optimizer stripped core context!"
