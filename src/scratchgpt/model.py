"""Complete decoder-only GPT language model for ScratchGPT."""

import math

import torch
from torch import nn

from scratchgpt.config import GPTConfig
from scratchgpt.embeddings import TokenAndPositionEmbedding
from scratchgpt.transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """Predict the next token using a decoder-only Transformer."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()

        self.config = config

        self.token_and_position_embedding = TokenAndPositionEmbedding(config)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )

        self.final_layer_norm = nn.LayerNorm(config.embedding_dim)

        self.output_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

        # Share the token embedding weights with the output head.
        self.output_head.weight = (
            self.token_and_position_embedding.token_embedding.weight
        )

        # Initialize weights using a GPT-style initialization scheme.
        self.apply(self._initialize_weights)

        # Reduce the initial magnitude of residual-branch output projections.
        residual_scale = 1.0 / math.sqrt(2 * config.num_layers)

        with torch.no_grad():
            for block in self.transformer_blocks:
                block.attention.output_projection.weight.mul_(residual_scale)
                block.feed_forward.network[2].weight.mul_(residual_scale)

    def _initialize_weights(self, module: nn.Module) -> None:
        """Initialize model weights with controlled magnitudes."""
        if isinstance(module, nn.Linear):
            # The output head shares its weight with the token embedding.
            # Skip it to avoid initializing that shared parameter twice.
            if (
                module.weight
                is self.token_and_position_embedding.token_embedding.weight
            ):
                return

            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return next-token logits for every input position."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, sequence_length)")

        if token_ids.dtype != torch.long:
            raise ValueError("token_ids must have dtype torch.long")

        sequence_length = token_ids.shape[1]

        if sequence_length == 0:
            raise ValueError("sequence_length must be greater than zero")

        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence_length ({sequence_length}) exceeds "
                f"context_length ({self.config.context_length})"
            )

        hidden_states = self.token_and_position_embedding(token_ids)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.output_head(hidden_states)

        return logits

    @property
    def parameter_count(self) -> int:
        """Return the number of unique trainable model parameters."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
