"""Save and restore ScratchGPT training checkpoints."""

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer

from scratchgpt.config import GPTConfig


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    config: GPTConfig,
    epoch: int,
    history: list[dict[str, object]],
    train_generator: torch.Generator | None = None,
) -> None:
    """Save model, optimizer, configuration, progress, and RNG states.

    Pass the training DataLoader's generator to make epoch-boundary resumes
    reproduce the subsequent training shuffle when using the same setup.
    """
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "history": history,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "train_generator_state": (
            train_generator.get_state() if train_generator is not None else None
        ),
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    config: GPTConfig,
    map_location: str | torch.device = "cpu",
    train_generator: torch.Generator | None = None,
) -> dict[str, object]:
    """Restore a checkpoint and return its training progress.

    If a training DataLoader generator is supplied, the checkpoint must include
    its state. This prevents silently claiming reproducible resume from an older
    checkpoint that did not save shuffle state.
    """
    checkpoint_path = Path(path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "config",
        "epoch",
        "history",
        "torch_rng_state",
    }
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Checkpoint is missing required fields: {missing}")

    if checkpoint["config"] != asdict(config):
        raise ValueError(
            "Checkpoint configuration does not match the supplied configuration"
        )

    if train_generator is not None and checkpoint.get("train_generator_state") is None:
        raise ValueError(
            "Checkpoint does not contain training DataLoader generator state; "
            "cannot resume reproducibly. Start a new full training run."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())

    saved_cuda_rng_state = checkpoint.get("cuda_rng_state")
    if saved_cuda_rng_state is not None and torch.cuda.is_available():
        if len(saved_cuda_rng_state) != torch.cuda.device_count():
            raise ValueError(
                "Checkpoint CUDA RNG state does not match the available CUDA "
                "device count; exact RNG restoration is not possible."
            )
        torch.cuda.set_rng_state_all([state.cpu() for state in saved_cuda_rng_state])

    if train_generator is not None:
        train_generator.set_state(checkpoint["train_generator_state"].cpu())

    return {
        "epoch": checkpoint["epoch"],
        "history": checkpoint["history"],
    }
