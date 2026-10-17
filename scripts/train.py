"""Train ScratchGPT from scratch on tokenized TinyStories data."""

import argparse
import json
import math
import random
import time
from itertools import islice
from pathlib import Path

import torch

from scratchgpt.checkpoint import load_checkpoint, save_checkpoint
from scratchgpt.config import GPTConfig
from scratchgpt.dataloaders import create_dataloaders
from scratchgpt.dataset import NextTokenDataset, load_token_ids
from scratchgpt.model import GPTModel
from scratchgpt.tokenizer import ByteBPETokenizer
from scratchgpt.training import evaluate_loss, train_one_epoch


def parse_args() -> argparse.Namespace:
    """Parse command-line training options."""
    parser = argparse.ArgumentParser(description="Train ScratchGPT on TinyStories.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit training batches per epoch for a smoke test.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
        help="Limit validation batches for a smoke test.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a compatible completed-epoch checkpoint.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/processed/tokenizer.json"),
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=Path("data/raw/train/tinystories_train.txt"),
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=Path("data/raw/validation/tinystories_validation.txt"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path (defaults to smoke-test or full-run path).",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=None,
        help="History JSON path (defaults to smoke-test or full-run path).",
    )

    args = parser.parse_args()

    for name in ("epochs", "batch_size", "sequence_length", "stride"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        parser.error("--learning-rate must be a finite positive number")

    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")

    for name in ("max_train_batches", "max_validation_batches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested, but CUDA is not available.")

    has_batch_limits = (
        args.max_train_batches is not None or args.max_validation_batches is not None
    )
    if args.resume and has_batch_limits:
        parser.error(
            "--resume cannot be combined with batch limits; resume only "
            "from completed full epochs."
        )

    if args.checkpoint is None:
        args.checkpoint = Path(
            "checkpoints/smoke_test.pt"
            if has_batch_limits
            else "checkpoints/scratchgpt.pt"
        )
    if args.history is None:
        args.history = Path(
            "checkpoints/smoke_test_history.json"
            if has_batch_limits
            else "checkpoints/training_history.json"
        )

    if args.resume and not args.checkpoint.is_file():
        parser.error(f"Cannot resume: checkpoint not found: {args.checkpoint}")

    if args.checkpoint.resolve() == args.history.resolve():
        parser.error("--checkpoint and --history must refer to different files.")

    return args


def select_device(requested: str) -> torch.device:
    """Choose the requested device or select CUDA automatically."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def move_batches(loader, device: torch.device, max_batches: int | None):
    """Move batches to the model's device, optionally limiting batch count."""
    batches = iter(loader)
    if max_batches is not None:
        batches = islice(batches, max_batches)
    for input_ids, targets in batches:
        yield input_ids.to(device), targets.to(device)


def processed_batch_count(loader, limit: int | None) -> int:
    """Return how many batches a complete loader or limited run processes."""
    if limit is None:
        return len(loader)
    return min(len(loader), limit)


def validate_resume_history(
    saved_history: object, saved_epoch: object
) -> list[dict[str, float | int | bool]]:
    """Reject incomplete or malformed progress before continuing training."""
    if not isinstance(saved_history, list):
        raise ValueError("Checkpoint history must be a list")

    if (
        isinstance(saved_epoch, bool)
        or not isinstance(saved_epoch, int)
        or saved_epoch < 1
    ):
        raise ValueError(
            "Resume requires a checkpoint with at least one completed epoch."
        )

    if len(saved_history) != saved_epoch:
        raise ValueError("Checkpoint history and completed epoch count do not match.")

    required_keys = {
        "epoch",
        "train_loss",
        "validation_loss",
        "train_batches_processed",
        "validation_batches_processed",
        "epoch_complete",
        "elapsed_seconds",
    }
    for index, record in enumerate(saved_history, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Checkpoint history record {index} must be an object.")
        if not required_keys.issubset(record):
            raise ValueError(
                f"Checkpoint history record {index} is missing required fields."
            )
        if isinstance(record.get("epoch"), bool) or record.get("epoch") != index:
            raise ValueError(
                "Checkpoint history epochs must be consecutive, starting at 1."
            )
        if record.get("epoch_complete") is not True:
            raise ValueError(
                "Cannot resume from a partial or legacy checkpoint. "
                "Use a checkpoint saved after a completed full epoch."
            )

        for key in ("train_loss", "validation_loss", "elapsed_seconds"):
            value = record.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Checkpoint history field {key!r} must be finite.")

        for key in ("train_batches_processed", "validation_batches_processed"):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"Checkpoint history field {key!r} must be a positive integer."
                )

    return saved_history


def main() -> None:
    """Prepare data, train the model, evaluate it, and save progress."""
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)

    for path in (args.tokenizer, args.train_file, args.validation_file):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    tokenizer = ByteBPETokenizer.load(args.tokenizer)

    print("Loading and encoding training data...")
    train_token_ids = load_token_ids(args.train_file, args.tokenizer)
    print("Loading and encoding validation data...")
    validation_token_ids = load_token_ids(args.validation_file, args.tokenizer)

    train_dataset = NextTokenDataset(
        train_token_ids,
        sequence_length=args.sequence_length,
        stride=args.stride,
    )
    validation_dataset = NextTokenDataset(
        validation_token_ids,
        sequence_length=args.sequence_length,
        stride=args.stride,
    )
    train_loader, validation_loader = create_dataloaders(
        train_dataset,
        validation_dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        context_length=args.sequence_length,
    )
    model = GPTModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, float | int | bool]] = []
    completed_epochs = 0

    if args.resume:
        progress = load_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            config,
            map_location=device,
            train_generator=train_loader.generator,
        )
        history = validate_resume_history(progress["history"], progress["epoch"])
        completed_epochs = progress["epoch"]

        if args.history.is_file():
            try:
                history_on_disk = json.loads(args.history.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"History file is not valid JSON: {args.history}"
                ) from error
            if history_on_disk != history:
                raise ValueError(
                    "History file does not match the checkpoint history. "
                    "Restore the matching history file or choose another --history path."
                )
        else:
            args.history.parent.mkdir(parents=True, exist_ok=True)
            args.history.write_text(
                json.dumps(history, indent=2) + "\n",
                encoding="utf-8",
            )

    print()
    print("ScratchGPT training")
    print("=" * 45)
    print(f"Device: {device}")
    print(f"Parameters: {model.parameter_count:,}")
    print(f"Vocabulary: {config.vocab_size:,}")
    print(f"Training tokens: {len(train_token_ids):,}")
    print(f"Validation tokens: {len(validation_token_ids):,}")
    print(f"Training examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(validation_dataset):,}")
    print(f"Training batches per full epoch: {len(train_loader):,}")
    print(f"Validation batches: {len(validation_loader):,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Stride: {args.stride}")
    print(f"Target completed epochs: {args.epochs}")
    print(f"Already completed epochs: {completed_epochs}")
    print(f"Learning rate: {args.learning_rate:g}")
    print(f"Resume enabled: {args.resume}")
    print()

    if completed_epochs >= args.epochs:
        print(
            "The checkpoint already meets the requested completed-epoch "
            "target. Increase --epochs to continue training."
        )
        return

    for epoch in range(completed_epochs + 1, args.epochs + 1):
        start_time = time.perf_counter()
        train_batches = processed_batch_count(train_loader, args.max_train_batches)
        validation_batches = processed_batch_count(
            validation_loader, args.max_validation_batches
        )
        epoch_complete = train_batches == len(
            train_loader
        ) and validation_batches == len(validation_loader)

        train_loss = train_one_epoch(
            model,
            move_batches(train_loader, device, args.max_train_batches),
            optimizer,
        )
        validation_loss = evaluate_loss(
            model,
            move_batches(validation_loader, device, args.max_validation_batches),
        )
        elapsed_seconds = time.perf_counter() - start_time

        record: dict[str, float | int | bool] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_batches_processed": train_batches,
            "validation_batches_processed": validation_batches,
            "epoch_complete": epoch_complete,
            "elapsed_seconds": elapsed_seconds,
        }
        history.append(record)
        if epoch_complete:
            completed_epochs = epoch

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"status={'complete' if epoch_complete else 'PARTIAL'} | "
            f"train_batches={train_batches}/{len(train_loader)} | "
            f"validation_batches={validation_batches}/{len(validation_loader)} | "
            f"train_loss={train_loss:.4f} | "
            f"validation_loss={validation_loss:.4f} | "
            f"time={elapsed_seconds:.1f}s"
        )

        save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            config,
            completed_epochs,
            history,
            train_generator=train_loader.generator,
        )
        args.history.parent.mkdir(parents=True, exist_ok=True)
        args.history.write_text(
            json.dumps(history, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Checkpoint saved: {args.checkpoint}")
        print(f"History saved: {args.history}")

        if not epoch_complete:
            print(
                "Partial run recorded. This checkpoint cannot be resumed "
                "as though the epoch were complete."
            )
            break

    print()
    print("Training run finished.")


if __name__ == "__main__":
    main()
