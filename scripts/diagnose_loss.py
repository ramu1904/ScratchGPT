"""Diagnose ScratchGPT's initial logits and next-token loss."""

import torch
from torch.utils.data import DataLoader

from scratchgpt.config import GPTConfig
from scratchgpt.dataset import NextTokenDataset, load_token_ids
from scratchgpt.loss import next_token_loss
from scratchgpt.model import GPTModel
from scratchgpt.tokenizer import ByteBPETokenizer


def main() -> None:
    tokenizer_path = "data/processed/tokenizer.json"
    train_path = "data/raw/train/tinystories_train.txt"

    tokenizer = ByteBPETokenizer.load(tokenizer_path)
    token_ids = load_token_ids(train_path, tokenizer_path)

    dataset = NextTokenDataset(
        token_ids,
        sequence_length=256,
        stride=256,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    inputs, targets = next(iter(loader))

    torch.manual_seed(42)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=256,
    )
    model = GPTModel(config)
    model.eval()

    with torch.no_grad():
        logits = model(inputs)
        loss = next_token_loss(logits, targets)

    embedding_weights = model.token_and_position_embedding.token_embedding.weight

    print("ScratchGPT initial numerical diagnostics")
    print("=" * 45)
    print(f"Embedding weight mean: {embedding_weights.mean().item():.6f}")
    print(f"Embedding weight std:  {embedding_weights.std().item():.6f}")
    print(f"Logits mean:           {logits.mean().item():.6f}")
    print(f"Logits std:            {logits.std().item():.6f}")
    print(f"Logits minimum:        {logits.min().item():.6f}")
    print(f"Logits maximum:        {logits.max().item():.6f}")
    print(f"Initial cross-entropy: {loss.item():.6f}")
    print(f"Target minimum ID:     {targets.min().item()}")
    print(f"Target maximum ID:     {targets.max().item()}")
    print(f"Vocabulary size:       {config.vocab_size}")
    print(f"Logits shape:          {tuple(logits.shape)}")
    print(f"Targets shape:         {tuple(targets.shape)}")


if __name__ == "__main__":
    main()
