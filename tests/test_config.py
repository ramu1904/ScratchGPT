"""Tests for ScratchGPT configuration."""

from dataclasses import FrozenInstanceError

import pytest

from scratchgpt.config import GPTConfig


def test_default_configuration():
    config = GPTConfig()

    assert config.vocab_size == 8192
    assert config.context_length == 256
    assert config.embedding_dim == 320
    assert config.num_layers == 6
    assert config.num_heads == 8
    assert config.feed_forward_dim == 1280
    assert config.dropout == 0.1


def test_head_dimension():
    config = GPTConfig(embedding_dim=256, num_heads=8)

    assert config.head_dim == 32


def test_custom_configuration():
    config = GPTConfig(
        vocab_size=512,
        context_length=32,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=128,
        dropout=0.0,
    )

    assert config.vocab_size == 512
    assert config.context_length == 32
    assert config.embedding_dim == 64
    assert config.num_layers == 2
    assert config.num_heads == 4
    assert config.feed_forward_dim == 128
    assert config.dropout == 0.0
    assert config.head_dim == 16


@pytest.mark.parametrize(
    "field",
    [
        "vocab_size",
        "context_length",
        "embedding_dim",
        "num_layers",
        "num_heads",
        "feed_forward_dim",
    ],
)
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_positive_integer_fields_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        GPTConfig(**{field: value})


@pytest.mark.parametrize(
    "embedding_dim,num_heads",
    [
        (250, 8),
        (256, 7),
        (10, 3),
    ],
)
def test_embedding_dimension_must_divide_evenly(embedding_dim, num_heads):
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5, True, "0.1"])
def test_invalid_dropout_is_rejected(dropout):
    with pytest.raises(ValueError, match="dropout"):
        GPTConfig(dropout=dropout)


def test_configuration_is_immutable():
    config = GPTConfig()

    with pytest.raises(FrozenInstanceError):
        config.embedding_dim = 512
