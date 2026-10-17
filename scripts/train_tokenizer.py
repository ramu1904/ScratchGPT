"""Train and save ScratchGPT's byte-level BPE tokenizer from local text files."""

import argparse
from pathlib import Path
from time import perf_counter

from scratchgpt.tokenizer import ByteBPETokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "train"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "tokenizer.json"
DEFAULT_VOCAB_SIZE = 8192


def load_corpus(
    input_dir: str | Path,
    max_chars: int | None = None,
) -> list[str]:
    """Load UTF-8 .txt files recursively from a directory.

    Files are processed in sorted path order. If max_chars is specified,
    loading stops after that many Unicode characters have been collected.
    """
    input_path = Path(input_dir)

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    if max_chars is not None and (type(max_chars) is not int or max_chars < 1):
        raise ValueError("max_chars must be a positive integer or None")

    text_files = sorted(input_path.rglob("*.txt"))

    if not text_files:
        raise FileNotFoundError(f"No .txt files found in input directory: {input_path}")

    texts: list[str] = []
    chars_loaded = 0

    for text_file in text_files:
        if max_chars is not None and chars_loaded >= max_chars:
            break

        text = text_file.read_text(encoding="utf-8")

        if max_chars is not None:
            remaining = max_chars - chars_loaded
            text = text[:remaining]

        if text:
            texts.append(text)
            chars_loaded += len(text)

    if not any(text.strip() for text in texts):
        raise ValueError(f"No non-whitespace text found in: {input_path}")

    return texts


def train_from_directory(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = 2,
    max_chars: int | None = None,
) -> ByteBPETokenizer:
    """Load a text corpus, train BPE, save the tokenizer, and report results."""
    if type(vocab_size) is not int or vocab_size < 256:
        raise ValueError("vocab_size must be an integer of at least 256")

    if type(min_frequency) is not int or min_frequency < 1:
        raise ValueError("min_frequency must be a positive integer")

    start_time = perf_counter()

    texts = load_corpus(input_dir, max_chars=max_chars)
    chars_loaded = sum(len(text) for text in texts)
    bytes_loaded = sum(len(text.encode("utf-8")) for text in texts)

    print("ScratchGPT tokenizer training")
    print("=" * 32)
    print(f"Input directory: {Path(input_dir).resolve()}")
    print(f"Text files loaded: {len(texts):,}")
    print(f"Characters loaded: {chars_loaded:,}")
    print(f"UTF-8 bytes loaded: {bytes_loaded:,}")
    print(f"Target vocabulary size: {vocab_size:,}")
    print(f"Minimum pair frequency: {min_frequency}")
    print()

    tokenizer = ByteBPETokenizer()
    tokenizer.train(
        texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    destination = Path(output_path)
    tokenizer.save(destination)

    sample = "\n".join(texts)[:500]
    sample_token_count = len(tokenizer.encode(sample))
    sample_char_count = len(sample)

    print("Training complete")
    print("-" * 32)
    print(f"Final vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Learned merges: {len(tokenizer.merges):,}")
    print(f"Sample characters: {sample_char_count:,}")
    print(f"Sample token IDs: {sample_token_count:,}")

    if sample_token_count:
        print(
            "Sample characters per token: "
            f"{sample_char_count / sample_token_count:.2f}"
        )

    if tokenizer.vocab_size < vocab_size:
        print()
        print(
            "WARNING: The tokenizer stopped below the requested vocabulary size. "
            "The corpus may not contain enough eligible pairs at the chosen "
            "minimum frequency."
        )

    print(f"Saved tokenizer: {destination.resolve()}")
    print(f"Elapsed time: {perf_counter() - start_time:.2f} seconds")

    return tokenizer


def main() -> None:
    """Parse command-line arguments and start tokenizer training."""
    parser = argparse.ArgumentParser(
        description="Train ScratchGPT's byte-level BPE tokenizer."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=(
            "Directory containing UTF-8 .txt files "
            "(default: data/raw/train, excluding validation data)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where tokenizer.json will be saved.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help="Maximum vocabulary size, including the 256 base byte tokens.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum count required for a pair to be merged.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Optional maximum number of corpus characters to load.",
    )

    args = parser.parse_args()

    train_from_directory(
        input_dir=args.input_dir,
        output_path=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
