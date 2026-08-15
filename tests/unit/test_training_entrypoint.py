"""Regression tests verifying production training entry point uses canonical 151M configuration."""

import pytest
import torch
from model.config import canonical_151m_config, tiny_test_config, GPTConfig
from model.transformer import GPT
from training.train import count_parameters, train_production_151m


def test_canonical_151m_parameter_count():
    model = GPT(canonical_151m_config)
    total_unique_params = count_parameters(model)
    assert total_unique_params == 151_862_784, (
        f"Canonical model parameter mismatch: expected 151,862,784, got {total_unique_params:,}"
    )


def test_production_training_initializes_canonical_model():
    model, params = train_production_151m(config=canonical_151m_config, max_steps=1, device="cpu")
    assert model.config.d_model == 768
    assert model.config.n_layers == 12
    assert model.config.n_heads == 12
    assert model.config.vocab_size == 50257
    assert model.config.context_length == 2048
    assert model.config.weight_tying is True
    assert params == 151_862_784


def test_tiny_test_config_isolation():
    assert tiny_test_config.d_model == 32
    assert tiny_test_config.n_layers == 2
    assert tiny_test_config.n_heads == 2
    assert tiny_test_config.vocab_size == 300
    tiny_model = GPT(tiny_test_config)
    tiny_params = count_parameters(tiny_model)
    assert tiny_params < 1_000_000
    assert tiny_params != 151_862_784
