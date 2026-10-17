import pytest


import json
from scratchgpt.tokenizer import TOKENIZER_FORMAT

from scratchgpt.tokenizer import ByteBPETokenizer


@pytest.fixture
def tokenizer() -> ByteBPETokenizer:
    return ByteBPETokenizer()


def test_initial_vocabulary_has_256_tokens(
    tokenizer: ByteBPETokenizer,
) -> None:
    assert tokenizer.vocab_size == 256


def test_all_byte_values_have_unique_token_ids(
    tokenizer: ByteBPETokenizer,
) -> None:
    assert set(tokenizer.token_bytes) == set(range(256))
    assert len(tokenizer.byte_to_id) == 256

    for byte_value in range(256):
        token_id = tokenizer.byte_to_id[bytes([byte_value])]
        assert token_id == byte_value
        assert tokenizer.token_bytes[token_id] == bytes([byte_value])


def test_encode_ascii_text(tokenizer: ByteBPETokenizer) -> None:
    assert tokenizer.encode("ABC") == [65, 66, 67]


def test_decode_ascii_token_ids(tokenizer: ByteBPETokenizer) -> None:
    assert tokenizer.decode([65, 66, 67]) == "ABC"


def test_round_trip_english_text(tokenizer: ByteBPETokenizer) -> None:
    text = "ScratchGPT learns from scratch!"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_round_trip_kannada_and_emoji(
    tokenizer: ByteBPETokenizer,
) -> None:
    text = "ನಮಸ್ಕಾರ! ಕನ್ನಡ ಮತ್ತು AI 🤖🚀"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_empty_string_round_trip(
    tokenizer: ByteBPETokenizer,
) -> None:
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""


def test_non_string_input_is_rejected(
    tokenizer: ByteBPETokenizer,
) -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        tokenizer.encode(123)  # type: ignore[arg-type]


def test_unknown_token_id_is_rejected(
    tokenizer: ByteBPETokenizer,
) -> None:
    with pytest.raises(ValueError, match="unknown token ID"):
        tokenizer.decode([256])


def test_non_integer_token_id_is_rejected(
    tokenizer: ByteBPETokenizer,
) -> None:
    with pytest.raises(TypeError, match="every token ID must be an integer"):
        tokenizer.decode([65, "66"])  # type: ignore[list-item]


def test_training_learns_at_least_one_merge() -> None:
    tokenizer = ByteBPETokenizer()
    tokenizer.train(["abababab", "abababab"], vocab_size=260)

    assert tokenizer.vocab_size > 256
    assert len(tokenizer.merges) == tokenizer.vocab_size - 256


def test_training_respects_vocabulary_limit() -> None:
    tokenizer = ByteBPETokenizer()
    tokenizer.train(["banana banana banana"], vocab_size=258)

    assert tokenizer.vocab_size <= 258


def test_training_stops_when_no_pair_meets_minimum_frequency() -> None:
    tokenizer = ByteBPETokenizer()
    tokenizer.train(["abc"], vocab_size=300, min_frequency=2)

    assert tokenizer.vocab_size == 256


def test_training_is_deterministic() -> None:
    texts = ["banana banana", "bandana banana"]

    first = ByteBPETokenizer()
    second = ByteBPETokenizer()

    first.train(texts, vocab_size=280, min_frequency=2)
    second.train(texts, vocab_size=280, min_frequency=2)

    assert first.merges == second.merges
    assert first.token_bytes == second.token_bytes


def test_trained_tokenizer_round_trips_text() -> None:
    tokenizer = ByteBPETokenizer()
    tokenizer.train(
        ["low lower lowest", "low low low", "ಕನ್ನಡ AI 🤖"],
        vocab_size=300,
    )

    text = "low lowest ಕನ್ನಡ AI 🤖"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_training_rejects_single_string() -> None:
    tokenizer = ByteBPETokenizer()

    with pytest.raises(TypeError, match="sequence of strings"):
        tokenizer.train("banana", vocab_size=260)  # type: ignore[arg-type]


def test_training_rejects_non_string_items() -> None:
    tokenizer = ByteBPETokenizer()

    with pytest.raises(TypeError, match="every training item"):
        tokenizer.train(["valid", 123], vocab_size=260)  # type: ignore[list-item]


@pytest.mark.parametrize("vocab_size", [255, 0, -1, 256.5, True])
def test_training_rejects_invalid_vocabulary_size(
    vocab_size: int,
) -> None:
    tokenizer = ByteBPETokenizer()

    with pytest.raises(ValueError, match="vocab_size"):
        tokenizer.train(["banana"], vocab_size=vocab_size)


@pytest.mark.parametrize("min_frequency", [0, -1, 1.5, True])
def test_training_rejects_invalid_min_frequency(
    min_frequency: int,
) -> None:
    tokenizer = ByteBPETokenizer()

    with pytest.raises(ValueError, match="min_frequency"):
        tokenizer.train(["banana"], min_frequency=min_frequency)


def test_save_and_load_preserves_tokenizer(tmp_path) -> None:
    texts = ["banana banana", "low lower lowest", "ಕನ್ನಡ AI 🤖"]

    original = ByteBPETokenizer()
    original.train(texts, vocab_size=280, min_frequency=2)

    sample = "banana lowest ಕನ್ನಡ AI 🤖"
    expected_ids = original.encode(sample)

    save_path = tmp_path / "tokenizer.json"
    original.save(save_path)

    loaded = ByteBPETokenizer.load(save_path)

    assert loaded.vocab_size == original.vocab_size
    assert loaded.merges == original.merges
    assert loaded.token_bytes == original.token_bytes
    assert loaded.encode(sample) == expected_ids
    assert loaded.decode(expected_ids) == sample


def test_save_creates_parent_directories(tmp_path) -> None:
    tokenizer = ByteBPETokenizer()
    path = tmp_path / "nested" / "tokenizer.json"

    tokenizer.save(path)

    assert path.exists()
    assert ByteBPETokenizer.load(path).vocab_size == 256


def test_saved_file_has_expected_format(tmp_path) -> None:
    tokenizer = ByteBPETokenizer()
    tokenizer.train(["banana banana"], vocab_size=260)

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    assert payload["format"] == TOKENIZER_FORMAT
    assert payload["vocab_size"] == tokenizer.vocab_size
    assert len(payload["merges"]) == tokenizer.vocab_size - 256


def test_load_rejects_unsupported_format(tmp_path) -> None:
    path = tmp_path / "tokenizer.json"
    path.write_text(
        json.dumps(
            {
                "format": "unknown-version",
                "vocab_size": 256,
                "merges": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported tokenizer format"):
        ByteBPETokenizer.load(path)


def test_load_rejects_invalid_merge_id(tmp_path) -> None:
    path = tmp_path / "tokenizer.json"
    path.write_text(
        json.dumps(
            {
                "format": TOKENIZER_FORMAT,
                "vocab_size": 257,
                "merges": [[0, 1, 300]],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid new token ID"):
        ByteBPETokenizer.load(path)


def test_load_rejects_vocab_size_mismatch(tmp_path) -> None:
    path = tmp_path / "tokenizer.json"
    path.write_text(
        json.dumps(
            {
                "format": TOKENIZER_FORMAT,
                "vocab_size": 300,
                "merges": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="vocab_size does not match"):
        ByteBPETokenizer.load(path)
