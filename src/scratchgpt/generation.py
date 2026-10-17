"""Autoregressive text-generation utilities for ScratchGPT."""

import math

import torch
from torch import nn


def generate_tokens(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    do_sample: bool = True,
) -> torch.Tensor:
    """Generate tokens autoregressively from an initial sequence.

    Args:
        model: A model that returns logits with shape
            (batch_size, sequence_length, vocabulary_size).
        input_ids: Initial token IDs with shape (batch_size, sequence_length).
        max_new_tokens: Maximum number of tokens to append.
        temperature: Sampling temperature. Must be finite and positive.
        top_k: If provided, sample only from the top-k token candidates.
        do_sample: If True, sample probabilistically. Otherwise, use greedy
            decoding by selecting the highest-scoring token.

    Returns:
        The original input tokens followed by the generated tokens.
    """
    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch_size, sequence_length)")

    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have a non-empty batch and sequence")

    if input_ids.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("input_ids must contain integer token IDs")

    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 0
    ):
        raise ValueError("max_new_tokens must be a non-negative integer")

    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and greater than zero")

    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1
    ):
        raise ValueError("top_k must be a positive integer or None")

    if not isinstance(do_sample, bool):
        raise TypeError("do_sample must be a boolean")

    config = getattr(model, "config", None)
    context_length = getattr(config, "context_length", None)

    if (
        isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or context_length < 1
    ):
        raise ValueError("model.config.context_length must be a positive integer")

    if max_new_tokens == 0:
        return input_ids.clone()

    generated = input_ids.clone()
    was_training = model.training

    model.eval()

    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # The model sees only the most recent context_length tokens.
                context = generated[:, -context_length:]

                logits = model(context)

                if logits.ndim != 3:
                    raise ValueError(
                        "Model logits must have shape "
                        "(batch_size, sequence_length, vocabulary_size)"
                    )

                if logits.shape[:2] != context.shape:
                    raise ValueError(
                        "Model logits batch and sequence dimensions "
                        "must match the input"
                    )

                # We only need scores for the next token.
                next_token_logits = logits[:, -1, :]

                if next_token_logits.shape[-1] < 1:
                    raise ValueError("Model vocabulary must not be empty")

                if do_sample:
                    if top_k is not None:
                        if top_k > next_token_logits.shape[-1]:
                            raise ValueError("top_k cannot exceed the vocabulary size")

                    # Temperature changes the sharpness of the distribution.
                    scaled_logits = next_token_logits / temperature

                    if top_k is not None:
                        top_values, _ = torch.topk(scaled_logits, k=top_k)

                        # Keep the top-k scores and suppress the rest.
                        threshold = top_values[:, -1].unsqueeze(-1)
                        scaled_logits = scaled_logits.masked_fill(
                            scaled_logits < threshold,
                            float("-inf"),
                        )

                    probabilities = torch.softmax(scaled_logits, dim=-1)

                    # Draw one token for each sequence in the batch.
                    next_token = torch.multinomial(
                        probabilities,
                        num_samples=1,
                    )
                else:
                    # Greedy decoding: choose the highest-scoring token.
                    next_token = next_token_logits.argmax(
                        dim=-1,
                        keepdim=True,
                    )

                generated = torch.cat(
                    [generated, next_token],
                    dim=1,
                )
    finally:
        # Restore the model's original training/evaluation mode.
        model.train(was_training)

    return generated
