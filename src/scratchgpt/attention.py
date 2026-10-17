"""Multi-head causal self-attention for ScratchGPT."""

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """Apply multi-head self-attention without looking at future tokens."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if type(embedding_dim) is not int or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")

        if type(num_heads) is not int or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not 0.0 <= dropout < 1.0
        ):
            raise ValueError("dropout must be in the range [0.0, 1.0)")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.qkv_projection = nn.Linear(
            embedding_dim,
            3 * embedding_dim,
        )

        self.attention_dropout = nn.Dropout(p=dropout)

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.output_dropout = nn.Dropout(p=dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return context-aware representations for each input token."""
        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape " "(batch_size, sequence_length, embedding_dim)"
            )

        if inputs.shape[1] == 0:
            raise ValueError("sequence_length must be greater than zero")

        if inputs.shape[2] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"received {inputs.shape[2]}"
            )

        if not inputs.is_floating_point():
            raise ValueError("inputs must have a floating-point dtype")

        batch_size, sequence_length, _ = inputs.shape

        # Project each token into query, key, and value vectors.
        qkv = self.qkv_projection(inputs)

        # Shape: (batch, sequence, 3, heads, head_dim).
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )

        # Move the Q/K/V axis to the front for convenient extraction.
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(dim=0)

        # Calculate scaled similarity scores between queries and keys.
        scores = queries @ keys.transpose(-2, -1)
        scores = scores / (self.head_dim**0.5)

        # Permit each position to attend only to itself and earlier tokens.
        causal_mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=inputs.device,
            )
        )

        scores = scores.masked_fill(~causal_mask, float("-inf"))

        # Convert scores into attention weights.
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        # Combine value vectors using the attention weights.
        context = attention_weights @ values

        # Join the attention heads back into one embedding vector.
        context = context.transpose(1, 2).contiguous()
        context = context.reshape(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        return self.output_dropout(self.output_projection(context))
