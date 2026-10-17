"""Byte-level BPE tokenizer for ScratchGPT."""

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import regex

GPT2_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?\p{L}+"""
    r"""| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{N}]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)

TOKENIZER_FORMAT = "scratchgpt-byte-bpe-v1"


class ByteBPETokenizer:
    """Learn byte-pair merges from text and encode text into token IDs."""

    def __init__(self) -> None:
        self.token_bytes: dict[int, bytes] = {}
        self.byte_to_id: dict[bytes, int] = {}
        self.merges: list[tuple[int, int, int]] = []
        self._reset_vocabulary()

    def _reset_vocabulary(self) -> None:
        """Initialize the vocabulary with all 256 byte values."""
        self.token_bytes = {
            byte_value: bytes([byte_value]) for byte_value in range(256)
        }
        self.byte_to_id = {
            token_bytes: token_id for token_id, token_bytes in self.token_bytes.items()
        }
        self.merges = []

    @property
    def vocab_size(self) -> int:
        """Return the number of tokens in the vocabulary."""
        return len(self.token_bytes)

    def _pretokenize(self, text: str) -> list[str]:
        """Split text into chunks before applying BPE."""
        return regex.findall(GPT2_PATTERN, text)

    @staticmethod
    def _count_pairs(
        sequences: Sequence[list[int]],
    ) -> Counter[tuple[int, int]]:
        """Count adjacent token pairs across all sequences."""
        pair_counts: Counter[tuple[int, int]] = Counter()

        for sequence in sequences:
            pair_counts.update(zip(sequence, sequence[1:]))

        return pair_counts

    @staticmethod
    def _merge_pair(
        sequence: list[int],
        pair: tuple[int, int],
        new_token_id: int,
    ) -> list[int]:
        """Merge non-overlapping pair occurrences from left to right."""
        left_id, right_id = pair
        merged: list[int] = []
        index = 0

        while index < len(sequence):
            if (
                index + 1 < len(sequence)
                and sequence[index] == left_id
                and sequence[index + 1] == right_id
            ):
                merged.append(new_token_id)
                index += 2
            else:
                merged.append(sequence[index])
                index += 1

        return merged

    @staticmethod
    def _build_pair_index(
        sequences: Sequence[list[int]],
    ) -> tuple[Counter[tuple[int, int]], dict[tuple[int, int], set[int]]]:
        """Build pair frequencies and the sequence IDs containing each pair."""
        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_to_sequences: dict[tuple[int, int], set[int]] = {}

        for sequence_id, sequence in enumerate(sequences):
            pairs = list(zip(sequence, sequence[1:]))
            pair_counts.update(pairs)

            for pair in set(pairs):
                pair_to_sequences.setdefault(pair, set()).add(sequence_id)

        return pair_counts, pair_to_sequences

    def train(
        self,
        texts: Sequence[str],
        vocab_size: int = 512,
        min_frequency: int = 2,
    ) -> None:
        """Learn BPE merges using incrementally maintained pair statistics."""
        if isinstance(texts, (str, bytes)):
            raise TypeError("texts must be a sequence of strings, not one string")

        if not isinstance(texts, Sequence):
            raise TypeError("texts must be a sequence of strings")

        if any(not isinstance(text, str) for text in texts):
            raise TypeError("every training item must be a string")

        if type(vocab_size) is not int or vocab_size < 256:
            raise ValueError("vocab_size must be an integer of at least 256")

        if type(min_frequency) is not int or min_frequency < 1:
            raise ValueError("min_frequency must be a positive integer")

        self._reset_vocabulary()

        sequences = [
            list(chunk.encode("utf-8"))
            for text in texts
            for chunk in self._pretokenize(text)
            if chunk
        ]

        pair_counts, pair_to_sequences = self._build_pair_index(sequences)

        while self.vocab_size < vocab_size:
            if not pair_counts:
                break

            best_pair = min(
                pair_counts,
                key=lambda pair: (-pair_counts[pair], pair),
            )

            if pair_counts[best_pair] < min_frequency:
                break

            new_token_id = self.vocab_size
            left_id, right_id = best_pair

            self.token_bytes[new_token_id] = (
                self.token_bytes[left_id] + self.token_bytes[right_id]
            )
            self.merges.append((left_id, right_id, new_token_id))

            affected_sequences = sorted(pair_to_sequences[best_pair])

            for sequence_id in affected_sequences:
                sequence = sequences[sequence_id]
                old_pairs = list(zip(sequence, sequence[1:]))

                # Remove the old sequence's contribution to pair counts.
                for pair in old_pairs:
                    pair_counts[pair] -= 1
                    if pair_counts[pair] == 0:
                        del pair_counts[pair]

                # Remove this sequence from the old pair index.
                for pair in set(old_pairs):
                    sequence_ids = pair_to_sequences[pair]
                    sequence_ids.discard(sequence_id)

                    if not sequence_ids:
                        del pair_to_sequences[pair]

                # Apply the selected merge to this sequence only.
                merged_sequence = self._merge_pair(
                    sequence,
                    best_pair,
                    new_token_id,
                )
                sequences[sequence_id] = merged_sequence

                # Add the updated sequence's contribution.
                new_pairs = list(zip(merged_sequence, merged_sequence[1:]))
                pair_counts.update(new_pairs)

                for pair in set(new_pairs):
                    pair_to_sequences.setdefault(pair, set()).add(sequence_id)

    def encode(self, text: str) -> list[int]:
        """Encode text using learned BPE merge priorities."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        merge_ranks = {
            (left_id, right_id): rank
            for rank, (left_id, right_id, _) in enumerate(self.merges)
        }
        merge_token_ids = {
            (left_id, right_id): new_token_id
            for left_id, right_id, new_token_id in self.merges
        }

        token_ids: list[int] = []

        for chunk in self._pretokenize(text):
            sequence = list(chunk.encode("utf-8"))

            while len(sequence) > 1:
                best_pair = min(
                    (
                        (left, right)
                        for left, right in zip(sequence, sequence[1:])
                        if (left, right) in merge_ranks
                    ),
                    key=lambda pair: merge_ranks[pair],
                    default=None,
                )

                if best_pair is None:
                    break

                sequence = self._merge_pair(
                    sequence,
                    best_pair,
                    merge_token_ids[best_pair],
                )

            token_ids.extend(sequence)

        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs into UTF-8 text."""
        if isinstance(token_ids, (str, bytes)):
            raise TypeError("token_ids must be a sequence of integers")

        if not isinstance(token_ids, Sequence):
            raise TypeError("token_ids must be a sequence of integers")

        byte_chunks: list[bytes] = []

        for token_id in token_ids:
            if type(token_id) is not int:
                raise TypeError("every token ID must be an integer")

            if token_id not in self.token_bytes:
                raise ValueError(f"unknown token ID: {token_id}")

            byte_chunks.append(self.token_bytes[token_id])

        return b"".join(byte_chunks).decode("utf-8")

    def save(self, path: str | Path) -> None:
        """Save the tokenizer format version and ordered merge rules."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format": TOKENIZER_FORMAT,
            "vocab_size": self.vocab_size,
            "merges": [list(merge) for merge in self.merges],
        }

        with destination.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        """Load a tokenizer and reconstruct its vocabulary from merges."""
        source = Path(path)

        with source.open("r", encoding="utf-8") as file:
            payload: Any = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError("tokenizer file must contain a JSON object")

        if payload.get("format") != TOKENIZER_FORMAT:
            raise ValueError("unsupported tokenizer format")

        merges_data = payload.get("merges")
        expected_vocab_size = payload.get("vocab_size")

        if type(expected_vocab_size) is not int:
            raise ValueError("vocab_size must be an integer")

        if not isinstance(merges_data, list):
            raise ValueError("merges must be a list")

        tokenizer = cls()

        for index, merge_data in enumerate(merges_data):
            if (
                not isinstance(merge_data, list)
                or len(merge_data) != 3
                or any(type(token_id) is not int for token_id in merge_data)
            ):
                raise ValueError(f"invalid merge entry at index {index}")

            left_id, right_id, new_token_id = merge_data

            if (
                left_id not in tokenizer.token_bytes
                or right_id not in tokenizer.token_bytes
            ):
                raise ValueError(f"merge {index} references an unknown token")

            if new_token_id != tokenizer.vocab_size:
                raise ValueError(f"merge {index} has an invalid new token ID")

            tokenizer.token_bytes[new_token_id] = (
                tokenizer.token_bytes[left_id] + tokenizer.token_bytes[right_id]
            )
            tokenizer.merges.append((left_id, right_id, new_token_id))

        tokenizer.byte_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in tokenizer.token_bytes.items()
        }

        if tokenizer.vocab_size != expected_vocab_size:
            raise ValueError("vocab_size does not match the merge rules")

        return tokenizer
