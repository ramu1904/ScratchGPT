"""Tests for ScratchGPT training and evaluation utilities."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scratchgpt.config import GPTConfig
from scratchgpt.model import GPTModel
from scratchgpt.training import (
    evaluate_loss,
    fit,
    train_one_epoch,
    train_step,
)


@pytest.fixture
def model():
    """Create a small GPT model for fast training tests."""
    config = GPTConfig(
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        feed_forward_dim=32,
        dropout=0.0,
    )
    return GPTModel(config)


@pytest.fixture
def batch():
    """Create a synthetic next-token prediction batch."""
    input_ids = torch.tensor(
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [[2, 3, 4, 5], [6, 7, 8, 9]],
        dtype=torch.long,
    )
    return input_ids, targets


@pytest.fixture
def dataloader(batch):
    """Create a small dataloader containing two examples."""
    input_ids, targets = batch
    dataset = TensorDataset(input_ids, targets)
    return DataLoader(dataset, batch_size=1, shuffle=False)


@pytest.fixture
def optimizer(model):
    """Create an optimizer for the test model."""
    return torch.optim.AdamW(model.parameters(), lr=0.001)


def test_train_step_returns_finite_loss(model, batch, optimizer):
    input_ids, targets = batch

    loss = train_step(model, input_ids, targets, optimizer)

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
    assert loss > 0


def test_train_step_updates_parameters(model, batch, optimizer):
    input_ids, targets = batch
    parameter = model.transformer_blocks[0].attention.qkv_projection.weight
    original_parameter = parameter.detach().clone()

    train_step(model, input_ids, targets, optimizer)

    assert not torch.equal(parameter, original_parameter)


def test_train_step_enables_training_mode(model, batch, optimizer):
    input_ids, targets = batch
    model.eval()

    train_step(model, input_ids, targets, optimizer)

    assert model.training


def test_train_step_creates_gradients(model, batch, optimizer):
    input_ids, targets = batch

    train_step(model, input_ids, targets, optimizer)

    parameter = model.transformer_blocks[0].attention.qkv_projection.weight
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


def test_train_step_supports_multiple_updates(model, batch, optimizer):
    input_ids, targets = batch

    first_loss = train_step(model, input_ids, targets, optimizer)
    second_loss = train_step(model, input_ids, targets, optimizer)

    assert torch.isfinite(torch.tensor(first_loss))
    assert torch.isfinite(torch.tensor(second_loss))


def test_train_step_rejects_invalid_input_shape(model, optimizer):
    input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4]], dtype=torch.long)

    with pytest.raises(ValueError, match="shape"):
        train_step(model, input_ids, targets, optimizer)


def test_train_one_epoch_returns_finite_loss(model, dataloader, optimizer):
    loss = train_one_epoch(model, dataloader, optimizer)

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
    assert loss > 0


def test_train_one_epoch_processes_all_batches(model, dataloader, optimizer):
    loss = train_one_epoch(model, dataloader, optimizer)

    assert torch.isfinite(torch.tensor(loss))
    assert model.training


def test_evaluate_loss_returns_finite_loss(model, dataloader):
    loss = evaluate_loss(model, dataloader)

    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
    assert loss > 0


def test_evaluate_loss_does_not_change_parameters(model, dataloader):
    parameter = model.transformer_blocks[0].attention.qkv_projection.weight
    original_parameter = parameter.detach().clone()

    evaluate_loss(model, dataloader)

    assert torch.equal(parameter, original_parameter)


def test_evaluate_loss_does_not_create_gradients(model, dataloader):
    evaluate_loss(model, dataloader)

    assert all(parameter.grad is None for parameter in model.parameters())


def test_evaluate_loss_sets_evaluation_mode(model, dataloader):
    model.train()

    evaluate_loss(model, dataloader)

    assert not model.training


def test_empty_training_dataloader_is_rejected(model, optimizer):
    dataset = TensorDataset(
        torch.empty((0, 4), dtype=torch.long),
        torch.empty((0, 4), dtype=torch.long),
    )
    dataloader = DataLoader(dataset, batch_size=2)

    with pytest.raises(ValueError, match="must not be empty"):
        train_one_epoch(model, dataloader, optimizer)


def test_empty_evaluation_dataloader_is_rejected(model):
    dataset = TensorDataset(
        torch.empty((0, 4), dtype=torch.long),
        torch.empty((0, 4), dtype=torch.long),
    )
    dataloader = DataLoader(dataset, batch_size=2)

    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_loss(model, dataloader)


def test_fit_returns_history_for_each_epoch(model, dataloader, optimizer):
    history = fit(
        model,
        dataloader,
        dataloader,
        optimizer,
        epochs=3,
    )

    assert len(history) == 3
    assert [entry["epoch"] for entry in history] == [1, 2, 3]


def test_fit_records_finite_losses(model, dataloader, optimizer):
    history = fit(
        model,
        dataloader,
        dataloader,
        optimizer,
        epochs=2,
    )

    for entry in history:
        assert torch.isfinite(torch.tensor(entry["train_loss"]))
        assert torch.isfinite(torch.tensor(entry["val_loss"]))


def test_fit_records_expected_history_keys(model, dataloader, optimizer):
    history = fit(
        model,
        dataloader,
        dataloader,
        optimizer,
        epochs=1,
    )

    assert set(history[0]) == {"epoch", "train_loss", "val_loss"}


def test_fit_updates_parameters(model, dataloader, optimizer):
    parameter = model.transformer_blocks[0].attention.qkv_projection.weight
    original_parameter = parameter.detach().clone()

    fit(model, dataloader, dataloader, optimizer, epochs=1)

    assert not torch.equal(parameter, original_parameter)


@pytest.mark.parametrize("epochs", [0, -1])
def test_fit_rejects_nonpositive_epoch_count(
    model,
    dataloader,
    optimizer,
    epochs,
):
    with pytest.raises(ValueError, match="positive integer"):
        fit(model, dataloader, dataloader, optimizer, epochs=epochs)


def test_fit_rejects_boolean_epoch_count(model, dataloader, optimizer):
    with pytest.raises(ValueError, match="positive integer"):
        fit(model, dataloader, dataloader, optimizer, epochs=True)


def test_fit_rejects_empty_training_dataloader(model, optimizer):
    empty_dataset = TensorDataset(
        torch.empty((0, 4), dtype=torch.long),
        torch.empty((0, 4), dtype=torch.long),
    )
    empty_loader = DataLoader(empty_dataset, batch_size=2)

    with pytest.raises(ValueError, match="must not be empty"):
        fit(model, empty_loader, empty_loader, optimizer, epochs=1)


def test_fit_rejects_empty_validation_dataloader(model, dataloader, optimizer):
    empty_dataset = TensorDataset(
        torch.empty((0, 4), dtype=torch.long),
        torch.empty((0, 4), dtype=torch.long),
    )
    empty_loader = DataLoader(empty_dataset, batch_size=2)

    with pytest.raises(ValueError, match="must not be empty"):
        fit(model, dataloader, empty_loader, optimizer, epochs=1)
