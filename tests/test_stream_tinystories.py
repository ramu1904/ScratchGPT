import pytest

from scripts.stream_tinystories import save_streamed_subset


def test_saves_only_requested_number_of_characters(tmp_path) -> None:
    output = tmp_path / "raw" / "sample.txt"
    records = [
        {"text": "First story"},
        {"text": "Second story"},
    ]

    stories, chars = save_streamed_subset(records, output, max_chars=15)

    assert stories == 2
    assert chars == 15
    assert len(output.read_text(encoding="utf-8").replace("\n", "")) == 15


def test_preserves_unicode_text(tmp_path) -> None:
    output = tmp_path / "sample.txt"

    stories, chars = save_streamed_subset(
        [{"text": "ಕನ್ನಡ 🤖"}],
        output,
        max_chars=100,
    )

    assert stories == 1
    assert chars == len("ಕನ್ನಡ 🤖")
    assert "ಕನ್ನಡ 🤖" in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("max_chars", [0, -1, 1.5, True])
def test_rejects_invalid_character_limit(tmp_path, max_chars) -> None:
    with pytest.raises(ValueError, match="max_chars must be a positive integer"):
        save_streamed_subset([], tmp_path / "sample.txt", max_chars)


def test_rejects_records_without_string_text(tmp_path) -> None:
    with pytest.raises(ValueError, match="must contain string text"):
        save_streamed_subset(
            [{"other": "not a text record"}],
            tmp_path / "sample.txt",
            max_chars=100,
        )


def test_rejects_empty_stream(tmp_path) -> None:
    with pytest.raises(ValueError, match="no usable text"):
        save_streamed_subset(
            [{"text": "  "}],
            tmp_path / "sample.txt",
            max_chars=100,
        )
