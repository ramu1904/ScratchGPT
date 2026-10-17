"""Tests for ScratchGPT checkpoint saving and loading."""

import pytest
import torch

from scratchgpt.checkpoint import load_checkpoint, save_checkpoint
from scratchgpt.config import GPTConfig
from scratchgpt.model import GPTModel
from scratchgpt.training import train_step


@pytest.fixture
def config():
    """Return a small model configuration."""
    return GPTConfig(
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        num_layers=1,
        num_heads=4,
        feed_forward_dim=32,
        dropout=0.0,
    )


@pytest.fixture
def model(config):
    """Create a small GPT model."""
    return GPTModel(config)


@pytest.fixture
def optimizer(model):
    """Create an AdamW optimizer."""
    return torch.optim.AdamW(model.parameters(), lr=0.001)


@pytest.fixture
def trained_model(model, optimizer):
    """Perform one update so the optimizer has state to restore."""
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    train_step(model, input_ids, targets, optimizer)
    return model


def test_save_checkpoint_creates_file(tmp_path, trained_model, optimizer, config):
    path = tmp_path / "checkpoints" / "latest.pt"
    save_checkpoint(path, trained_model, optimizer, config, 2, [])
    assert path.is_file()


def test_save_checkpoint_creates_parent_directories(
    tmp_path, trained_model, optimizer, config
):
    path = tmp_path / "nested" / "folder" / "checkpoint.pt"
    save_checkpoint(path, trained_model, optimizer, config, 1, [])
    assert path.is_file()


def test_load_checkpoint_restores_model_parameters(
    tmp_path, trained_model, optimizer, config
):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, trained_model, optimizer, config, 2, [])
    expected = {
        name: parameter.detach().clone()
        for name, parameter in trained_model.named_parameters()
    }

    with torch.no_grad():
        for parameter in trained_model.parameters():
            parameter.add_(1.0)

    load_checkpoint(path, trained_model, optimizer, config)

    for name, parameter in trained_model.named_parameters():
        assert torch.equal(parameter, expected[name])


def test_load_checkpoint_restores_training_progress(
    tmp_path, trained_model, optimizer, config
):
    path = tmp_path / "checkpoint.pt"
    history = [
        {"epoch": 1, "train_loss": 3.5, "val_loss": 3.6},
        {"epoch": 2, "train_loss": 3.1, "val_loss": 3.3},
    ]
    save_checkpoint(path, trained_model, optimizer, config, 2, history)
    progress = load_checkpoint(path, trained_model, optimizer, config)
    assert progress["epoch"] == 2
    assert progress["history"] == history


def test_load_checkpoint_restores_optimizer_state(
    tmp_path, trained_model, optimizer, config
):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, trained_model, optimizer, config, 1, [])
    expected = optimizer.state_dict()
    optimizer.state.clear()

    load_checkpoint(path, trained_model, optimizer, config)

    restored = optimizer.state_dict()
    assert restored["state"].keys() == expected["state"].keys()
    for parameter_id, expected_state in expected["state"].items():
        restored_state = restored["state"][parameter_id]
        for key, expected_value in expected_state.items():
            actual_value = restored_state[key]
            if isinstance(expected_value, torch.Tensor):
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def test_load_checkpoint_restores_torch_rng_state(
    tmp_path, trained_model, optimizer, config
):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, trained_model, optimizer, config, 1, [])
    expected_next_values = torch.rand(3)
    load_checkpoint(path, trained_model, optimizer, config)
    actual_next_values = torch.rand(3)
    assert torch.equal(actual_next_values, expected_next_values)


def test_load_checkpoint_restores_training_generator_state(
    tmp_path, trained_model, optimizer, config
):
    path = tmp_path / "checkpoint.pt"
    generator = torch.Generator().manual_seed(123)
    torch.randperm(20, generator=generator)
    save_checkpoint(path, trained_model, optimizer, config, 1, [], generator)

    expected = torch.rand(5, generator=generator)
    load_checkpoint(
        path,
        trained_model,
        optimizer,
        config,
        train_generator=generator,
    )
    actual = torch.rand(5, generator=generator)
    assert torch.equal(actual, expected)


def test_load_checkpoint_requires_generator_state_for_reproducible_resume(
    tmp_path, trained_model, optimizer, config
):
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, trained_model, optimizer, config, 1, [])
    generator = torch.Generator().manual_seed(123)

    with pytest.raises(ValueError, match="generator state"):
        load_checkpoint(
            path,
            trained_model,
            optimizer,
            config,
            train_generator=generator,
        )


def test_load_checkpoint_rejects_mismatched_configuration(
    tmp_path, trained_model, optimizer, config
):
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, trained_model, optimizer, config, 1, [])
    incompatible_config = GPTConfig(
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=32,
        dropout=0.0,
    )
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(path, trained_model, optimizer, incompatible_config)


def test_load_checkpoint_rejects_missing_file(
    tmp_path, trained_model, optimizer, config
):
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(tmp_path / "missing.pt", trained_model, optimizer, config)


@pytest.mark.parametrize("epoch", [-1, True, 1.5])
def test_save_checkpoint_rejects_invalid_epoch(
    tmp_path, trained_model, optimizer, config, epoch
):
    with pytest.raises(ValueError, match="non-negative integer"):
        save_checkpoint(
            tmp_path / "checkpoint.pt",
            trained_model,
            optimizer,
            config,
            epoch,
            [],
        )
