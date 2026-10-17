"""Token and positional embeddings for ScratchGPT."""

import torch
from torch import nn

from scratchgpt.config import GPTConfig


class TokenAndPositionEmbedding(nn.Module):
    """Convert token IDs into token and positional representations."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.context_length = config.context_length

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=config.context_length,
            embedding_dim=config.embedding_dim,
        )

        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return combined token and positional embeddings."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, sequence_length)")

        if token_ids.dtype != torch.long:
            raise ValueError("token_ids must have dtype torch.long")

        sequence_length = token_ids.shape[1]

        if sequence_length == 0:
            raise ValueError("sequence_length must be greater than zero")

        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence_length ({sequence_length}) exceeds "
                f"context_length ({self.context_length})"
            )

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(positions)

        combined_vectors = token_vectors + position_vectors

        return self.dropout(combined_vectors)
