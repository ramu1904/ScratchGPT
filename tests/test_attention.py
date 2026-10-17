"""Tests for ScratchGPT causal self-attention."""

import pytest
import torch

from scratchgpt.attention import CausalSelfAttention


@pytest.fixture
def attention():
    """Return a small attention layer for testing."""
    return CausalSelfAttention(
        embedding_dim=64,
        num_heads=4,
        dropout=0.1,
    )


def test_output_shape(attention):
    inputs = torch.randn(2, 8, 64)

    output = attention(inputs)

    assert output.shape == (2, 8, 64)


def test_single_sequence(attention):
    inputs = torch.randn(1, 5, 64)

    output = attention(inputs)

    assert output.shape == (1, 5, 64)


def test_single_token(attention):
    inputs = torch.randn(2, 1, 64)

    output = attention(inputs)

    assert output.shape == (2, 1, 64)
    assert torch.isfinite(output).all()


def test_output_is_finite(attention):
    inputs = torch.randn(2, 8, 64)

    output = attention(inputs)

    assert torch.isfinite(output).all()


def test_future_tokens_do_not_affect_earlier_outputs(attention):
    attention.eval()

    original = torch.randn(1, 6, 64)
    modified = original.clone()

    # Change tokens at positions 3, 4, and 5.
    modified[:, 3:, :] = torch.randn(1, 3, 64) * 10

    original_output = attention(original)
    modified_output = attention(modified)

    # Outputs at positions 0, 1, and 2 must remain unchanged.
    assert torch.allclose(
        original_output[:, :3, :],
        modified_output[:, :3, :],
        atol=1e-6,
    )


def test_attention_parameters_receive_gradients(attention):
    inputs = torch.randn(2, 8, 64, requires_grad=True)

    output = attention(inputs)
    loss = output.square().mean()
    loss.backward()

    assert inputs.grad is not None
    assert attention.qkv_projection.weight.grad is not None
    assert attention.output_projection.weight.grad is not None


def test_evaluation_mode_is_deterministic(attention):
    attention.eval()
    inputs = torch.randn(2, 8, 64)

    first_output = attention(inputs)
    second_output = attention(inputs)

    assert torch.equal(first_output, second_output)


@pytest.mark.parametrize(
    "embedding_dim,num_heads",
    [
        (0, 4),
        (-64, 4),
        (64, 0),
        (64, -4),
        (64.0, 4),
        (64, 4.0),
        (True, 4),
    ],
)
def test_invalid_dimensions_are_rejected(embedding_dim, num_heads):
    with pytest.raises(ValueError):
        CausalSelfAttention(embedding_dim, num_heads)


@pytest.mark.parametrize(
    "embedding_dim,num_heads",
    [
        (63, 4),
        (64, 7),
    ],
)
def test_non_divisible_dimensions_are_rejected(embedding_dim, num_heads):
    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(embedding_dim, num_heads)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 2.0, True, "0.1"])
def test_invalid_dropout_is_rejected(dropout):
    with pytest.raises(ValueError, match="dropout"):
        CausalSelfAttention(64, 4, dropout)


@pytest.mark.parametrize(
    "shape",
    [
        (8, 64),
        (2, 8, 4, 16),
    ],
)
def test_invalid_input_dimensions_are_rejected(attention, shape):
    inputs = torch.randn(shape)

    with pytest.raises(ValueError, match="shape"):
        attention(inputs)


def test_empty_sequence_is_rejected(attention):
    inputs = torch.empty(2, 0, 64)

    with pytest.raises(ValueError, match="sequence_length"):
        attention(inputs)


def test_wrong_embedding_dimension_is_rejected(attention):
    inputs = torch.randn(2, 8, 32)

    with pytest.raises(ValueError, match="embedding dimension"):
        attention(inputs)


def test_integer_inputs_are_rejected(attention):
    inputs = torch.ones(2, 8, 64, dtype=torch.long)

    with pytest.raises(ValueError, match="floating-point"):
        attention(inputs)
