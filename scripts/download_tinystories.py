"""Download a bounded TinyStories subset using Hugging Face streaming."""

import argparse
from pathlib import Path

# Initialize PyTorch before datasets creates its streaming iterator.
# This avoids the import-order failure observed in this environment.
import torch  # noqa: F401

from datasets import load_dataset

DATASET_NAME = "roneneldan/TinyStories"
DEFAULT_TRAIN_STORIES = 20_000
DEFAULT_VALIDATION_STORIES = 1_000


def download_split(split: str, limit: int, output_path: Path) -> None:
    """Stream up to limit non-empty stories into a UTF-8 text file."""
    if limit < 1:
        raise ValueError("The story limit must be at least 1.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True,
    )

    story_count = 0
    character_count = 0

    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as file:
            for row in dataset:
                story = row.get("text", "").strip()

                if not story:
                    continue

                file.write(story)
                file.write("\n\n")

                story_count += 1
                character_count += len(story)

                if story_count % 1_000 == 0:
                    print(f"{split}: saved {story_count:,} stories")

                if story_count >= limit:
                    break

        if story_count == 0:
            raise RuntimeError(f"No stories were downloaded for split {split!r}.")

        temporary_path.replace(output_path)

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    file_bytes = output_path.stat().st_size

    print(f"\nCompleted split: {split}")
    print(f"Stories saved: {story_count:,}")
    print(f"Characters saved: {character_count:,}")
    print(f"File size: {file_bytes:,} bytes")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a bounded TinyStories subset using streaming."
    )
    parser.add_argument(
        "--train-stories",
        type=int,
        default=DEFAULT_TRAIN_STORIES,
        help="Number of training stories to retain.",
    )
    parser.add_argument(
        "--validation-stories",
        type=int,
        default=DEFAULT_VALIDATION_STORIES,
        help="Number of validation stories to retain.",
    )
    args = parser.parse_args()

    download_split(
        split="train",
        limit=args.train_stories,
        output_path=Path("data/raw/train/tinystories_train.txt"),
    )

    download_split(
        split="validation",
        limit=args.validation_stories,
        output_path=Path("data/raw/validation/tinystories_validation.txt"),
    )


if __name__ == "__main__":
    main()
