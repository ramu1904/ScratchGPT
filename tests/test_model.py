"""Tests for the complete ScratchGPT model."""

import math

import pytest
import torch

from scratchgpt.config import GPTConfig
from scratchgpt.loss import next_token_loss
from scratchgpt.model import GPTModel


@pytest.fixture
def config():
    """Return a small model configuration for fast tests."""
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
def model(config):
    """Return a small GPT model for testing."""
    return GPTModel(config)


def test_output_shape(model):
    token_ids = torch.randint(0, 512, (2, 8))

    logits = model(token_ids)

    assert logits.shape == (2, 8, 512)


def test_single_sequence(model):
    token_ids = torch.randint(0, 512, (1, 5))

    logits = model(token_ids)

    assert logits.shape == (1, 5, 512)


def test_single_token(model):
    token_ids = torch.randint(0, 512, (2, 1))

    logits = model(token_ids)

    assert logits.shape == (2, 1, 512)
    assert torch.isfinite(logits).all()


def test_output_logits_are_finite(model):
    token_ids = torch.randint(0, 512, (2, 8))

    logits = model(token_ids)

    assert torch.isfinite(logits).all()


def test_model_uses_configured_number_of_blocks(model, config):
    assert len(model.transformer_blocks) == config.num_layers


def test_output_head_shares_token_embedding_weights(model):
    assert (
        model.output_head.weight
        is model.token_and_position_embedding.token_embedding.weight
    )


def test_parameter_count_matches_unique_parameters(model):
    expected = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    assert model.parameter_count == expected


def test_token_embedding_uses_small_normal_initialization(model):
    weights = model.token_and_position_embedding.token_embedding.weight

    assert abs(weights.mean().item()) < 0.003
    assert weights.std().item() == pytest.approx(0.02, abs=0.003)


def test_position_embedding_uses_small_normal_initialization(model):
    weights = model.token_and_position_embedding.position_embedding.weight

    assert abs(weights.mean().item()) < 0.01
    assert weights.std().item() == pytest.approx(0.02, abs=0.004)


def test_attention_residual_projection_is_scaled(model, config):
    weights = model.transformer_blocks[0].attention.output_projection.weight

    expected_std = 0.02 / math.sqrt(2 * config.num_layers)

    assert weights.std().item() == pytest.approx(expected_std, rel=0.15)


def test_feed_forward_residual_projection_is_scaled(model, config):
    weights = model.transformer_blocks[0].feed_forward.network[2].weight

    expected_std = 0.02 / math.sqrt(2 * config.num_layers)

    assert weights.std().item() == pytest.approx(expected_std, rel=0.15)


def test_initial_cross_entropy_is_reasonable(model):
    model.eval()

    token_ids = torch.randint(0, 512, (2, 8))
    targets = torch.randint(0, 512, (2, 8))

    with torch.no_grad():
        logits = model(token_ids)
        loss = next_token_loss(logits, targets)

    uniform_baseline = math.log(512)

    assert torch.isfinite(loss)
    assert loss.item() < uniform_baseline + 1.0


def test_gradients_reach_model_parameters(model):
    token_ids = torch.randint(0, 512, (2, 8))

    logits = model(token_ids)
    loss = logits.square().mean()
    loss.backward()

    assert model.token_and_position_embedding.token_embedding.weight.grad is not None
    assert model.transformer_blocks[0].attention.qkv_projection.weight.grad is not None
    assert model.transformer_blocks[0].feed_forward.network[0].weight.grad is not None
    assert model.output_head.weight.grad is not None


def test_future_tokens_do_not_affect_earlier_logits(model):
    model.eval()

    original = torch.randint(0, 512, (1, 8))
    modified = original.clone()
    modified[:, 4:] = torch.randint(0, 512, (1, 4))

    original_logits = model(original)
    modified_logits = model(modified)

    assert torch.allclose(
        original_logits[:, :4, :],
        modified_logits[:, :4, :],
        atol=1e-5,
    )


def test_evaluation_mode_is_deterministic(model):
    model.eval()
    token_ids = torch.randint(0, 512, (2, 8))

    first_logits = model(token_ids)
    second_logits = model(token_ids)

    assert torch.equal(first_logits, second_logits)


def test_invalid_input_shape_is_rejected(model):
    token_ids = torch.randint(0, 512, (8,))

    with pytest.raises(ValueError, match="shape"):
        model(token_ids)


def test_non_long_token_ids_are_rejected(model):
    token_ids = torch.ones(2, 8, dtype=torch.int32)

    with pytest.raises(ValueError, match="torch.long"):
        model(token_ids)


def test_empty_sequence_is_rejected(model):
    token_ids = torch.empty(2, 0, dtype=torch.long)

    with pytest.raises(ValueError, match="sequence_length"):
        model(token_ids)


def test_sequence_longer_than_context_is_rejected(model):
    token_ids = torch.randint(0, 512, (1, 33))

    with pytest.raises(ValueError, match="context_length"):
        model(token_ids)


def test_token_id_outside_vocabulary_is_rejected(model):
    token_ids = torch.tensor([[512]], dtype=torch.long)

    with pytest.raises(IndexError):
        model(token_ids)


def test_default_model_is_approximately_ten_million_parameters():
    model = GPTModel(GPTConfig())

    assert model.parameter_count == 10_101_760
    assert 10_000_000 <= model.parameter_count <= 10_200_000
