"""Estimate ScratchGPT training workload using the saved tokenizer."""

from pathlib import Path

from scratchgpt.dataset import NextTokenDataset
from scratchgpt.tokenizer import ByteBPETokenizer


def main() -> None:
    tokenizer_path = Path("data/processed/tokenizer.json")
    train_path = Path("data/raw/train/tinystories_train.txt")
    validation_path = Path("data/raw/validation/tinystories_validation.txt")

    sequence_length = 256
    stride = 256
    batch_size = 8

    tokenizer = ByteBPETokenizer.load(tokenizer_path)

    print("ScratchGPT training workload estimate")
    print("=" * 45)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Sequence length: {sequence_length}")
    print(f"Stride: {stride}")
    print(f"Batch size: {batch_size}")
    print()

    for split_name, file_path in (
        ("TRAIN", train_path),
        ("VALIDATION", validation_path),
    ):
        if not file_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        text = file_path.read_text(encoding="utf-8")
        token_ids = tokenizer.encode(text)

        dataset = NextTokenDataset(
            token_ids,
            sequence_length=sequence_length,
            stride=stride,
        )

        batches_per_epoch = (len(dataset) + batch_size - 1) // batch_size

        print(f"{split_name}")
        print(f"  Characters:       {len(text):,}")
        print(f"  Tokens:           {len(token_ids):,}")
        print(f"  Training examples: {len(dataset):,}")
        print(f"  Batches per epoch: {batches_per_epoch:,}")
        print()


if __name__ == "__main__":
    main()
