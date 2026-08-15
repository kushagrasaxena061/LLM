from model.config import canonical_151m_config
from model.transformer import GPT


def test_canonical_151m_parameter_count():
    model = GPT(canonical_151m_config)

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    assert total_params == 151_862_784


def test_canonical_configuration():
    config = canonical_151m_config

    assert config.d_model == 768
    assert config.n_layers == 12
    assert config.n_heads == 12
    assert config.head_dim == 64
    assert config.weight_tying is True
