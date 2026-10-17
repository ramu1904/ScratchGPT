"""Tests for ScratchGPT token and positional embeddings."""

import pytest
import torch

from scratchgpt.config import GPTConfig
from scratchgpt.embeddings import TokenAndPositionEmbedding


@pytest.fixture
def config():
    """Return a small configuration for fast tests."""
    return GPTConfig(
        vocab_size=512,
        context_length=32,
        embedding_dim=64,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=128,
        dropout=0.1,
    )


@pytest.fixture
def embedding_layer(config):
    """Create an embedding layer for testing."""
    return TokenAndPositionEmbedding(config)


def test_output_shape(embedding_layer):
    token_ids = torch.randint(0, 512, (2, 8))

    output = embedding_layer(token_ids)

    assert output.shape == (2, 8, 64)


def test_output_is_floating_point(embedding_layer):
    token_ids = torch.randint(0, 512, (2, 8))

    output = embedding_layer(token_ids)

    assert output.dtype == torch.float32


def test_single_sequence(embedding_layer):
    token_ids = torch.randint(0, 512, (1, 5))

    output = embedding_layer(token_ids)

    assert output.shape == (1, 5, 64)


def test_maximum_context_length(embedding_layer):
    token_ids = torch.randint(0, 512, (2, 32))

    output = embedding_layer(token_ids)

    assert output.shape == (2, 32, 64)


def test_sequence_longer_than_context_is_rejected(embedding_layer):
    token_ids = torch.randint(0, 512, (2, 33))

    with pytest.raises(ValueError, match="context_length"):
        embedding_layer(token_ids)


@pytest.mark.parametrize(
    "token_ids",
    [
        torch.tensor([1, 2, 3]),
        torch.tensor(5),
        torch.ones(2, 3, 4, dtype=torch.long),
    ],
)
def test_invalid_input_dimensions_are_rejected(embedding_layer, token_ids):
    with pytest.raises(ValueError, match="shape"):
        embedding_layer(token_ids)


def test_non_long_token_ids_are_rejected(embedding_layer):
    token_ids = torch.ones(2, 8, dtype=torch.int32)

    with pytest.raises(ValueError, match="torch.long"):
        embedding_layer(token_ids)


def test_empty_sequence_is_rejected(embedding_layer):
    token_ids = torch.empty((2, 0), dtype=torch.long)

    with pytest.raises(ValueError, match="sequence_length"):
        embedding_layer(token_ids)


def test_token_id_outside_vocabulary_is_rejected(embedding_layer):
    token_ids = torch.tensor([[512]], dtype=torch.long)

    with pytest.raises(IndexError):
        embedding_layer(token_ids)


def test_embeddings_receive_gradients(embedding_layer):
    token_ids = torch.randint(0, 512, (2, 8))

    output = embedding_layer(token_ids)
    loss = output.square().mean()
    loss.backward()

    assert embedding_layer.token_embedding.weight.grad is not None
    assert embedding_layer.position_embedding.weight.grad is not None


def test_evaluation_mode_disables_dropout(embedding_layer):
    token_ids = torch.randint(0, 512, (2, 8))
    embedding_layer.eval()

    first_output = embedding_layer(token_ids)
    second_output = embedding_layer(token_ids)

    assert torch.equal(first_output, second_output)


def test_different_positions_have_distinct_parameters(embedding_layer):
    position_weights = embedding_layer.position_embedding.weight

    assert not torch.equal(position_weights[0], position_weights[1])
