"""Transformer block for ScratchGPT."""

import torch
from torch import nn

from scratchgpt.attention import CausalSelfAttention
from scratchgpt.config import GPTConfig


class FeedForward(nn.Module):
    """Apply a nonlinear transformation to each token representation."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(config.embedding_dim, config.feed_forward_dim),
            nn.GELU(),
            nn.Linear(config.feed_forward_dim, config.embedding_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the input while preserving its shape."""
        return self.network(inputs)


class TransformerBlock(nn.Module):
    """Apply pre-normalized causal attention and feed-forward sublayers."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(config.embedding_dim)

        self.attention = CausalSelfAttention(
            embedding_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        self.layer_norm_2 = nn.LayerNorm(config.embedding_dim)
        self.feed_forward = FeedForward(config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return transformed representations with residual connections."""
        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape " "(batch_size, sequence_length, embedding_dim)"
            )

        if inputs.shape[1] == 0:
            raise ValueError("sequence_length must be greater than zero")

        if inputs.shape[2] != self.layer_norm_1.normalized_shape[0]:
            raise ValueError("inputs have an incorrect embedding dimension")

        if not inputs.is_floating_point():
            raise ValueError("inputs must have a floating-point dtype")

        # Sublayer 1: normalize, attend, and add the residual.
        inputs = inputs + self.attention(self.layer_norm_1(inputs))

        # Sublayer 2: normalize, transform, and add the residual.
        inputs = inputs + self.feed_forward(self.layer_norm_2(inputs))

        return inputs
