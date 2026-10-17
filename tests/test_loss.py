"""Tests for ScratchGPT's next-token prediction loss."""

import pytest
import torch

from scratchgpt.loss import next_token_loss


def test_loss_returns_scalar():
    logits = torch.randn(2, 4, 10)
    targets = torch.randint(0, 10, (2, 4))

    loss = next_token_loss(logits, targets)

    assert loss.ndim == 0
    assert loss.item() > 0


def test_loss_is_finite():
    logits = torch.randn(2, 4, 10)
    targets = torch.randint(0, 10, (2, 4))

    loss = next_token_loss(logits, targets)

    assert torch.isfinite(loss)


def test_loss_matches_manual_cross_entropy():
    logits = torch.tensor([[[3.0, 1.0, 0.0], [0.0, 2.0, 1.0]]])
    targets = torch.tensor([[0, 1]], dtype=torch.long)

    actual = next_token_loss(logits, targets)

    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 3),
        targets.reshape(-1),
    )

    assert torch.allclose(actual, expected)


def test_correct_predictions_have_lower_loss():
    targets = torch.tensor([[0, 1]], dtype=torch.long)

    confident_logits = torch.tensor([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]])
    incorrect_logits = torch.tensor([[[0.0, 10.0, 0.0], [10.0, 0.0, 0.0]]])

    confident_loss = next_token_loss(confident_logits, targets)
    incorrect_loss = next_token_loss(incorrect_logits, targets)

    assert confident_loss < incorrect_loss


def test_loss_supports_batch_size_one():
    logits = torch.randn(1, 3, 10)
    targets = torch.randint(0, 10, (1, 3))

    loss = next_token_loss(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_gradients_flow_to_logits():
    logits = torch.randn(2, 4, 10, requires_grad=True)
    targets = torch.randint(0, 10, (2, 4))

    loss = next_token_loss(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_invalid_logits_shape_is_rejected():
    logits = torch.randn(2, 10)
    targets = torch.randint(0, 10, (2,))

    with pytest.raises(ValueError, match="logits"):
        next_token_loss(logits, targets)


def test_invalid_targets_shape_is_rejected():
    logits = torch.randn(2, 4, 10)
    targets = torch.randint(0, 10, (8,))

    with pytest.raises(ValueError, match="targets"):
        next_token_loss(logits, targets)


def test_mismatched_batch_and_sequence_shapes_are_rejected():
    logits = torch.randn(2, 4, 10)
    targets = torch.randint(0, 10, (2, 3))

    with pytest.raises(ValueError, match="matching"):
        next_token_loss(logits, targets)


def test_non_long_targets_are_rejected():
    logits = torch.randn(2, 4, 10)
    targets = torch.zeros(2, 4, dtype=torch.int32)

    with pytest.raises(ValueError, match="torch.long"):
        next_token_loss(logits, targets)


def test_integer_logits_are_rejected():
    logits = torch.ones(2, 4, 10, dtype=torch.long)
    targets = torch.zeros(2, 4, dtype=torch.long)

    with pytest.raises(ValueError, match="floating-point"):
        next_token_loss(logits, targets)


@pytest.mark.parametrize(
    "shape",
    [
        (0, 4, 10),
        (2, 0, 10),
        (2, 4, 0),
    ],
)
def test_empty_logits_dimensions_are_rejected(shape):
    logits = torch.empty(shape)
    targets = torch.empty(shape[:2], dtype=torch.long)

    with pytest.raises(ValueError, match="greater than zero"):
        next_token_loss(logits, targets)
