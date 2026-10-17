"""Tests for ScratchGPT Transformer blocks."""

import pytest
import torch
from torch import nn

from scratchgpt.config import GPTConfig
from scratchgpt.transformer_block import FeedForward, TransformerBlock


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
def block(config):
    """Return a Transformer block for testing."""
    return TransformerBlock(config)


def test_output_shape(block):
    inputs = torch.randn(2, 8, 64)

    output = block(inputs)

    assert output.shape == inputs.shape


def test_single_sequence(block):
    inputs = torch.randn(1, 5, 64)

    output = block(inputs)

    assert output.shape == (1, 5, 64)


def test_single_token(block):
    inputs = torch.randn(2, 1, 64)

    output = block(inputs)

    assert output.shape == (2, 1, 64)
    assert torch.isfinite(output).all()


def test_output_is_finite(block):
    inputs = torch.randn(2, 8, 64)

    output = block(inputs)

    assert torch.isfinite(output).all()


def test_feed_forward_preserves_shape(config):
    feed_forward = FeedForward(config)
    inputs = torch.randn(2, 8, 64)

    output = feed_forward(inputs)

    assert output.shape == inputs.shape


def test_feed_forward_expands_and_projects_dimensions(config):
    feed_forward = FeedForward(config)

    first_linear = feed_forward.network[0]
    second_linear = feed_forward.network[2]

    assert isinstance(first_linear, nn.Linear)
    assert isinstance(feed_forward.network[1], nn.GELU)
    assert isinstance(second_linear, nn.Linear)

    assert first_linear.in_features == 64
    assert first_linear.out_features == 128
    assert second_linear.in_features == 128
    assert second_linear.out_features == 64


def test_block_contains_two_layer_norms(block):
    assert isinstance(block.layer_norm_1, nn.LayerNorm)
    assert isinstance(block.layer_norm_2, nn.LayerNorm)


def test_block_contains_causal_attention(block):
    from scratchgpt.attention import CausalSelfAttention

    assert isinstance(block.attention, CausalSelfAttention)


def test_gradients_reach_input_and_parameters(block):
    inputs = torch.randn(2, 8, 64, requires_grad=True)

    output = block(inputs)
    loss = output.square().mean()
    loss.backward()

    assert inputs.grad is not None
    assert block.attention.qkv_projection.weight.grad is not None
    assert block.feed_forward.network[0].weight.grad is not None
    assert block.feed_forward.network[2].weight.grad is not None


def test_future_tokens_do_not_affect_earlier_outputs(block):
    block.eval()

    original = torch.randn(1, 6, 64)
    modified = original.clone()
    modified[:, 3:, :] = torch.randn(1, 3, 64) * 10

    original_output = block(original)
    modified_output = block(modified)

    assert torch.allclose(
        original_output[:, :3, :],
        modified_output[:, :3, :],
        atol=1e-6,
    )


def test_evaluation_mode_is_deterministic(block):
    block.eval()
    inputs = torch.randn(2, 8, 64)

    first_output = block(inputs)
    second_output = block(inputs)

    assert torch.equal(first_output, second_output)


@pytest.mark.parametrize(
    "shape",
    [
        (8, 64),
        (2, 8, 4, 16),
    ],
)
def test_invalid_input_dimensions_are_rejected(block, shape):
    inputs = torch.randn(shape)

    with pytest.raises(ValueError, match="shape"):
        block(inputs)


def test_empty_sequence_is_rejected(block):
    inputs = torch.empty(2, 0, 64)

    with pytest.raises(ValueError, match="sequence_length"):
        block(inputs)


def test_wrong_embedding_dimension_is_rejected(block):
    inputs = torch.randn(2, 8, 32)

    with pytest.raises(ValueError, match="embedding dimension"):
        block(inputs)


def test_integer_inputs_are_rejected(block):
    inputs = torch.ones(2, 8, 64, dtype=torch.long)

    with pytest.raises(ValueError, match="floating-point"):
        block(inputs)
