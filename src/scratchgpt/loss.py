"""Next-token prediction loss for ScratchGPT."""

import torch
import torch.nn.functional as F


def next_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Calculate mean cross-entropy loss for next-token predictions.

    Args:
        logits: Model outputs with shape (batch_size, sequence_length, vocab_size).
        targets: Correct token IDs with shape (batch_size, sequence_length).

    Returns:
        A scalar tensor containing the mean cross-entropy loss.
    """
    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape (batch_size, sequence_length, vocab_size)"
        )

    if targets.ndim != 2:
        raise ValueError("targets must have shape (batch_size, sequence_length)")

    if logits.shape[:2] != targets.shape:
        raise ValueError(
            "logits and targets must have matching batch and sequence dimensions"
        )

    if targets.dtype != torch.long:
        raise ValueError("targets must have dtype torch.long")

    if not logits.is_floating_point():
        raise ValueError("logits must have a floating-point dtype")

    if any(dimension == 0 for dimension in logits.shape):
        raise ValueError("logits dimensions must be greater than zero")

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
    )
