from collections import Counter
from pathlib import Path
from time import perf_counter

from scratchgpt.tokenizer import ByteBPETokenizer


def reference_train(texts, vocab_size, min_frequency=2):
    tokenizer = ByteBPETokenizer()
    tokenizer._reset_vocabulary()

    sequences = [
        list(chunk.encode("utf-8"))
        for text in texts
        for chunk in tokenizer._pretokenize(text)
        if chunk
    ]

    while tokenizer.vocab_size < vocab_size:
        counts = Counter(
            pair for sequence in sequences for pair in zip(sequence, sequence[1:])
        )

        if not counts:
            break

        pair = min(counts, key=lambda item: (-counts[item], item))

        if counts[pair] < min_frequency:
            break

        new_id = tokenizer.vocab_size
        left, right = pair

        tokenizer.token_bytes[new_id] = (
            tokenizer.token_bytes[left] + tokenizer.token_bytes[right]
        )
        tokenizer.merges.append((left, right, new_id))

        sequences = [
            tokenizer._merge_pair(sequence, pair, new_id) for sequence in sequences
        ]

    return tokenizer


path = Path("data/raw/tinystories_sample.txt")
text = path.read_text(encoding="utf-8")

print(f"Corpus size: {len(text):,} characters")
print(f"Target vocabulary: 512")

start = perf_counter()
reference = reference_train([text], vocab_size=512)
reference_seconds = perf_counter() - start

start = perf_counter()
optimized = ByteBPETokenizer()
optimized.train([text], vocab_size=512)
optimized_seconds = perf_counter() - start

assert reference.merges == optimized.merges
assert reference.token_bytes == optimized.token_bytes

print(f"Reference: {reference_seconds:.3f} seconds")
print(f"Optimized: {optimized_seconds:.3f} seconds")
print(f"Speedup: {reference_seconds / optimized_seconds:.2f}x")
print(f"Final vocabulary: {optimized.vocab_size}")
print("Merge sequences match exactly.")
