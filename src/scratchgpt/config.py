"""Configuration for the ScratchGPT language model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    """Store and validate GPT architecture hyperparameters."""

    vocab_size: int = 8192
    context_length: int = 256
    embedding_dim: int = 320
    num_layers: int = 6
    num_heads: int = 8
    feed_forward_dim: int = 1280
    dropout: float = 0.1

    def __post_init__(self) -> None:
        """Reject invalid architecture settings."""
        positive_integer_fields = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim,
        }

        for name, value in positive_integer_fields.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0.0 <= self.dropout < 1.0
        ):
            raise ValueError("dropout must be in the range [0.0, 1.0)")

    @property
    def head_dim(self) -> int:
        """Return the dimension of each attention head."""
        return self.embedding_dim // self.num_heads
