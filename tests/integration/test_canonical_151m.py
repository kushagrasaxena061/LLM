import pytest
from model.config import canonical_151m_config
from model.transformer import GPT

def test_canonical_151m_architecture():
    """
    Mathematically verifies the production model configuration 
    instantiates the exact expected parameter count.
    """
    model = GPT(canonical_151m_config)
    total_params = sum(p.numel() for p in model.parameters())
    
    # 151,862,784 is the mathematically precise count for this config with weight tying
    expected_params = 151862784
    
    assert canonical_151m_config.weight_tying is True, "Weight tying MUST be enabled in production."
    assert canonical_151m_config.d_model == 768, "Canonical d_model altered!"
    assert canonical_151m_config.n_layers == 12, "Canonical n_layers altered!"
    
    if total_params != expected_params:
        pytest.fail(f"CRITICAL ARCHITECTURE DRIFT: Expected {expected_params} parameters, but instantiated {total_params}.")
