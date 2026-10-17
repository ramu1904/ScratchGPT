"""PyTorch DataLoader utilities for ScratchGPT."""

import torch
from torch.utils.data import DataLoader

from scratchgpt.dataset import NextTokenDataset


def create_dataloaders(
    train_dataset: NextTokenDataset,
    validation_dataset: NextTokenDataset,
    batch_size: int = 8,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create reproducible training and validation DataLoaders.

    The seeded generator is attached to the training loader and can be saved
    with a checkpoint for reproducible epoch-boundary resume. Training examples
    are shuffled; validation examples retain their order. num_workers defaults
    to zero for straightforward Windows compatibility.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    if type(seed) is not int:
        raise TypeError("seed must be an integer")

    if type(num_workers) is not int or num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    return train_loader, validation_loader
