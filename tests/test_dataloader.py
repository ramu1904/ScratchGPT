"""Tests for ScratchGPT DataLoader utilities."""

import pytest
import torch

from scratchgpt.dataset import NextTokenDataset
from scratchgpt.dataloaders import create_dataloaders


@pytest.fixture
def datasets():
    train = NextTokenDataset(list(range(100)), sequence_length=4)
    validation = NextTokenDataset(list(range(100, 140)), sequence_length=4)
    return train, validation


def test_loader_batch_shapes_and_dtypes(datasets):
    train, validation = datasets
    train_loader, validation_loader = create_dataloaders(
        train, validation, batch_size=8
    )
    inputs, targets = next(iter(train_loader))
    assert inputs.shape == (8, 4)
    assert targets.shape == (8, 4)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long

    inputs, targets = next(iter(validation_loader))
    assert inputs.shape == (8, 4)
    assert targets.shape == (8, 4)


def test_validation_loader_preserves_order(datasets):
    train, validation = datasets
    _, validation_loader = create_dataloaders(train, validation, batch_size=8)
    inputs, _ = next(iter(validation_loader))
    assert inputs[0].tolist() == [100, 101, 102, 103]
    assert inputs[1].tolist() == [101, 102, 103, 104]


def test_final_incomplete_batch_is_preserved(datasets):
    train, validation = datasets
    train_loader, _ = create_dataloaders(train, validation, batch_size=8)
    total_examples = sum(batch_inputs.shape[0] for batch_inputs, _ in train_loader)
    assert total_examples == len(train)


def test_seed_reproduces_training_shuffle(datasets):
    train, validation = datasets
    loader_a, _ = create_dataloaders(train, validation, batch_size=8, seed=123)
    loader_b, _ = create_dataloaders(train, validation, batch_size=8, seed=123)
    first_a, _ = next(iter(loader_a))
    first_b, _ = next(iter(loader_b))
    assert torch.equal(first_a, first_b)


def test_training_loader_exposes_its_generator(datasets):
    train, validation = datasets
    train_loader, _ = create_dataloaders(train, validation, seed=123)
    assert isinstance(train_loader.generator, torch.Generator)


def test_training_generator_state_reproduces_next_shuffle(datasets):
    train, validation = datasets
    train_loader, _ = create_dataloaders(train, validation, batch_size=8, seed=123)
    generator = train_loader.generator
    assert generator is not None

    # Advance the generator once, then save the state at an epoch boundary.
    list(iter(train_loader))
    saved_state = generator.get_state()

    expected_batches = [inputs.clone() for inputs, _ in train_loader]
    generator.set_state(saved_state)
    actual_batches = [inputs.clone() for inputs, _ in train_loader]

    assert len(actual_batches) == len(expected_batches)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(actual_batches, expected_batches)
    )


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size_is_rejected(datasets, batch_size):
    train, validation = datasets
    with pytest.raises(ValueError, match="batch_size"):
        create_dataloaders(train, validation, batch_size=batch_size)


@pytest.mark.parametrize("num_workers", [-1, 1.5, True])
def test_invalid_num_workers_is_rejected(datasets, num_workers):
    train, validation = datasets
    with pytest.raises(ValueError, match="num_workers"):
        create_dataloaders(train, validation, num_workers=num_workers)


def test_invalid_seed_type_is_rejected(datasets):
    train, validation = datasets
    with pytest.raises(TypeError, match="seed"):
        create_dataloaders(train, validation, seed=1.5)
