"""Tests for ScratchGPT dataset utilities."""

import pytest
import torch

from scratchgpt.dataset import (
    NextTokenDataset,
    load_token_ids,
    split_token_ids,
)
from scratchgpt.tokenizer import ByteBPETokenizer


def test_next_token_dataset_shifts_targets_by_one():
    dataset = NextTokenDataset([10, 20, 30, 40, 50], sequence_length=3)

    inputs, targets = dataset[0]

    assert inputs.tolist() == [10, 20, 30]
    assert targets.tolist() == [20, 30, 40]


def test_next_token_dataset_length():
    dataset = NextTokenDataset(list(range(10)), sequence_length=4)

    assert len(dataset) == 6


def test_next_token_dataset_returns_long_tensors():
    dataset = NextTokenDataset(list(range(6)), sequence_length=3)

    inputs, targets = dataset[0]

    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long


def test_next_token_dataset_rejects_invalid_sequence_length():
    with pytest.raises(ValueError, match="sequence_length"):
        NextTokenDataset([1, 2, 3], sequence_length=0)


def test_next_token_dataset_requires_enough_tokens():
    with pytest.raises(ValueError, match="more tokens"):
        NextTokenDataset([1, 2, 3], sequence_length=3)


def test_next_token_dataset_rejects_non_integer_tokens():
    with pytest.raises(TypeError, match="only integers"):
        NextTokenDataset([1, 2.5, 3], sequence_length=1)


def test_next_token_dataset_rejects_out_of_range_index():
    dataset = NextTokenDataset([1, 2, 3, 4], sequence_length=2)

    with pytest.raises(IndexError):
        dataset[2]


def test_split_token_ids_creates_separate_datasets():
    token_ids = list(range(100))

    train, validation = split_token_ids(
        token_ids,
        train_fraction=0.8,
        sequence_length=5,
    )

    assert len(train) == 75
    assert len(validation) == 15


def test_split_keeps_validation_tokens_out_of_training():
    token_ids = list(range(100))

    train, validation = split_token_ids(
        token_ids,
        train_fraction=0.8,
        sequence_length=5,
    )

    train_inputs, train_targets = train[len(train) - 1]
    validation_inputs, validation_targets = validation[0]

    assert train_inputs.tolist() == [74, 75, 76, 77, 78]
    assert train_targets.tolist() == [75, 76, 77, 78, 79]
    assert validation_inputs.tolist() == [80, 81, 82, 83, 84]
    assert validation_targets.tolist() == [81, 82, 83, 84, 85]


def test_split_rejects_invalid_fraction():
    with pytest.raises(ValueError, match="between 0 and 1"):
        split_token_ids(list(range(100)), train_fraction=1.0)


def test_split_rejects_too_short_training_segment():
    with pytest.raises(ValueError, match="training split"):
        split_token_ids(
            list(range(10)),
            train_fraction=0.2,
            sequence_length=3,
        )


def test_split_rejects_too_short_validation_segment():
    with pytest.raises(ValueError, match="validation split"):
        split_token_ids(
            list(range(10)),
            train_fraction=0.9,
            sequence_length=3,
        )


def test_load_token_ids_uses_saved_tokenizer(tmp_path):
    text_path = tmp_path / "sample.txt"
    tokenizer_path = tmp_path / "tokenizer.json"

    text_path.write_text("Hello world! ನಮಸ್ಕಾರ 😊", encoding="utf-8")

    tokenizer = ByteBPETokenizer()
    tokenizer.train(["Hello world! नमस्कार 😊"], vocab_size=300)
    tokenizer.save(tokenizer_path)

    token_ids = load_token_ids(text_path, tokenizer_path)

    assert token_ids
    assert tokenizer.decode(token_ids) == "Hello world! ನಮಸ್ಕಾರ 😊"


def test_load_token_ids_rejects_empty_text(tmp_path):
    text_path = tmp_path / "empty.txt"
    tokenizer_path = tmp_path / "tokenizer.json"

    text_path.write_text("", encoding="utf-8")

    ByteBPETokenizer().save(tokenizer_path)

    with pytest.raises(ValueError, match="must not be empty"):
        load_token_ids(text_path, tokenizer_path)
