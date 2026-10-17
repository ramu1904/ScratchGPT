from collections import Counter
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
        pair_counts = Counter()

        for sequence in sequences:
            for pair in zip(sequence, sequence[1:]):
                pair_counts[pair] += 1

        if not pair_counts:
            break

        best_pair = min(
            pair_counts,
            key=lambda pair: (-pair_counts[pair], pair),
        )

        if pair_counts[best_pair] < min_frequency:
            break

        new_token_id = tokenizer.vocab_size
        left_id, right_id = best_pair

        tokenizer.token_bytes[new_token_id] = (
            tokenizer.token_bytes[left_id] + tokenizer.token_bytes[right_id]
        )
        tokenizer.merges.append((left_id, right_id, new_token_id))

        tokenizer_sequences = [
            tokenizer._merge_pair(sequence, best_pair, new_token_id)
            for sequence in sequences
        ]
        sequences = tokenizer_sequences

    return tokenizer


corpora = [
    ["banana banana bandana", "banana bandana"] * 4,
    ["The quick brown fox jumps over the lazy dog."] * 8,
    ["abababababab", "bababababa"] * 8,
]

for index, texts in enumerate(corpora, start=1):
    reference = reference_train(texts, vocab_size=320)
    optimized = ByteBPETokenizer()

    start = perf_counter()
    optimized.train(texts, vocab_size=320)
    optimized_seconds = perf_counter() - start

    assert (
        optimized.merges == reference.merges
    ), f"Corpus {index}: merge sequences differ"
    assert optimized.token_bytes == reference.token_bytes

    print(
        f"Corpus {index}: PASS | "
        f"vocabulary={optimized.vocab_size} | "
        f"optimized training={optimized_seconds:.4f}s"
    )

print("All reference comparisons passed.")
