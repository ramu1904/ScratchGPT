"""Training and evaluation utilities for ScratchGPT."""

from torch import nn
from torch.optim import Optimizer

from scratchgpt.loss import next_token_loss


def train_step(
    model: nn.Module,
    input_ids,
    targets,
    optimizer: Optimizer,
) -> float:
    """Perform one optimization step and return the pre-update loss."""
    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids)
    loss = next_token_loss(logits, targets)

    loss.backward()
    optimizer.step()

    return loss.item()


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: Optimizer,
) -> float:
    """Train on all batches and return the token-weighted mean loss."""
    total_loss = 0.0
    total_tokens = 0

    for input_ids, targets in dataloader:
        loss = train_step(model, input_ids, targets, optimizer)

        number_of_tokens = targets.numel()
        total_loss += loss * number_of_tokens
        total_tokens += number_of_tokens

    if total_tokens == 0:
        raise ValueError("Training dataloader must not be empty")

    return total_loss / total_tokens


def evaluate_loss(
    model: nn.Module,
    dataloader,
) -> float:
    """Evaluate token-weighted mean loss without updating parameters."""
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    import torch

    with torch.no_grad():
        for input_ids, targets in dataloader:
            logits = model(input_ids)
            loss = next_token_loss(logits, targets)

            number_of_tokens = targets.numel()
            total_loss += loss.item() * number_of_tokens
            total_tokens += number_of_tokens

    if total_tokens == 0:
        raise ValueError("Evaluation dataloader must not be empty")

    return total_loss / total_tokens


def fit(
    model: nn.Module,
    train_dataloader,
    val_dataloader,
    optimizer: Optimizer,
    epochs: int,
) -> list[dict[str, float | int]]:
    """Train for multiple epochs and record training and validation losses."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")

    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
        )

        val_loss = evaluate_loss(
            model,
            val_dataloader,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

    return history
