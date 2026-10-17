"""Stream a bounded TinyStories subset into a local UTF-8 text file."""

import argparse
from collections.abc import Iterable
from pathlib import Path

DATASET_ID = "roneneldan/TinyStories"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "tinystories_sample.txt"


def save_streamed_subset(
    records: Iterable[dict[str, object]],
    output_path: str | Path,
    max_chars: int,
) -> tuple[int, int]:
    """Write at most max_chars from streamed dataset records.

    Returns:
        A tuple containing the number of records written and characters saved.
    """
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    stories_written = 0
    chars_written = 0

    with destination.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            if chars_written >= max_chars:
                break

            text = record.get("text")

            if not isinstance(text, str):
                raise ValueError("Each dataset record must contain string text")

            if not text.strip():
                continue

            remaining = max_chars - chars_written
            text_to_write = text[:remaining]

            file.write(text_to_write)
            file.write("\n")

            stories_written += 1
            chars_written += len(text_to_write)

    if chars_written == 0:
        destination.unlink(missing_ok=True)
        raise ValueError("The streamed dataset contained no usable text")

    return stories_written, chars_written


def main() -> None:
    """Stream a limited number of characters from TinyStories."""
    parser = argparse.ArgumentParser(
        description="Stream a small TinyStories subset without downloading all data."
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=25_000,
        help="Maximum number of text characters to save (default: 25000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination UTF-8 text file.",
    )
    args = parser.parse_args()

    if args.max_chars < 1:
        parser.error("--max-chars must be a positive integer")

    # Import here so the helper can be tested without network access.
    from datasets import load_dataset

    print(f"Dataset: {DATASET_ID}")
    print("Loading dataset in streaming mode; the full dataset is not downloaded.")

    dataset = load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
    )

    stories, chars = save_streamed_subset(
        records=dataset,
        output_path=args.output,
        max_chars=args.max_chars,
    )

    print("Streaming complete")
    print(f"Records written: {stories:,}")
    print(f"Characters written: {chars:,}")
    print(f"Saved to: {args.output.resolve()}")
    print("The saved text file is local; subsequent tokenizer runs can reuse it.")


if __name__ == "__main__":
    main()
