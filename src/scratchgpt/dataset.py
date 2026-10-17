"""Dataset utilities for ScratchGPT causal language modeling."""

from pathlib import Path

import torch
from torch.utils.data import Dataset

from scratchgpt.tokenizer import ByteBPETokenizer


class NextTokenDataset(Dataset):
    """Create fixed-length input and target sequences for next-token prediction."""

    def __init__(
        self,
        token_ids: list[int],
        sequence_length: int,
        stride: int = 1,
    ) -> None:
        if type(sequence_length) is not int or sequence_length < 1:
            raise ValueError("sequence_length must be a positive integer")

        if type(stride) is not int or stride < 1:
            raise ValueError("stride must be a positive integer")

        if any(type(token_id) is not int for token_id in token_ids):
            raise TypeError("token_ids must contain only integers")

        if len(token_ids) <= sequence_length:
            raise ValueError("token_ids must contain more tokens than sequence_length")

        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.sequence_length = sequence_length
        self.stride = stride

    def __len__(self) -> int:
        """Return the number of available next-token training examples."""
        return (len(self.token_ids) - self.sequence_length - 1) // self.stride + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an input sequence and its one-token-shifted target."""
        if index < 0 or index >= len(self):
            raise IndexError("dataset index out of range")

        start = index * self.stride
        end = start + self.sequence_length

        inputs = self.token_ids[start:end]
        targets = self.token_ids[start + 1 : end + 1]

        return inputs, targets


def load_token_ids(
    text_path: str | Path,
    tokenizer_path: str | Path,
) -> list[int]:
    """Load UTF-8 text and encode it using a saved ScratchGPT tokenizer."""
    text_file = Path(text_path)
    tokenizer_file = Path(tokenizer_path)

    text = text_file.read_text(encoding="utf-8")
    if not text:
        raise ValueError("input text file must not be empty")

    tokenizer = ByteBPETokenizer.load(tokenizer_file)
    token_ids = tokenizer.encode(text)

    if not token_ids:
        raise ValueError("encoded text must contain at least one token")

    return token_ids


def split_token_ids(
    token_ids: list[int],
    train_fraction: float = 0.9,
    sequence_length: int = 128,
    stride: int = 1,
) -> tuple[NextTokenDataset, NextTokenDataset]:
    """Split tokens chronologically, then create separate train/validation datasets."""
    if not isinstance(train_fraction, (int, float)) or isinstance(train_fraction, bool):
        raise TypeError("train_fraction must be a number")

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    if type(sequence_length) is not int or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer")

    if type(stride) is not int or stride < 1:
        raise ValueError("stride must be a positive integer")

    if any(type(token_id) is not int for token_id in token_ids):
        raise TypeError("token_ids must contain only integers")

    split_index = int(len(token_ids) * train_fraction)

    if split_index <= sequence_length:
        raise ValueError("training split is too short for sequence_length")

    if len(token_ids) - split_index <= sequence_length:
        raise ValueError("validation split is too short for sequence_length")

    train_dataset = NextTokenDataset(
        token_ids[:split_index],
        sequence_length,
        stride=stride,
    )
    validation_dataset = NextTokenDataset(
        token_ids[split_index:],
        sequence_length,
        stride=stride,
    )

    return train_dataset, validation_dataset
