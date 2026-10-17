import pytest

from scratchgpt.tokenizer import ByteBPETokenizer
from scripts.train_tokenizer import load_corpus, train_from_directory


def test_load_corpus_reads_text_files_in_sorted_order(tmp_path) -> None:
    (tmp_path / "b.txt").write_text("Second file.", encoding="utf-8")
    (tmp_path / "a.txt").write_text("First file.", encoding="utf-8")
    (tmp_path / "notes.md").write_text(
        "This file should be ignored.",
        encoding="utf-8",
    )

    texts = load_corpus(tmp_path)

    assert texts == ["First file.", "Second file."]


def test_load_corpus_reads_nested_text_files(tmp_path) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    (nested_dir / "story.txt").write_text(
        "ಕನ್ನಡ and 🤖",
        encoding="utf-8",
    )

    texts = load_corpus(tmp_path)

    assert texts == ["ಕನ್ನಡ and 🤖"]


def test_load_corpus_respects_max_chars(tmp_path) -> None:
    (tmp_path / "a.txt").write_text("abcdef", encoding="utf-8")
    (tmp_path / "b.txt").write_text("ghijkl", encoding="utf-8")

    texts = load_corpus(tmp_path, max_chars=8)

    assert sum(len(text) for text in texts) == 8
    assert texts == ["abcdef", "gh"]


def test_load_corpus_rejects_missing_directory(tmp_path) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Input directory does not exist"):
        load_corpus(missing_dir)


def test_load_corpus_rejects_directory_without_text_files(tmp_path) -> None:
    (tmp_path / "notes.md").write_text("Some notes.", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No .txt files found"):
        load_corpus(tmp_path)


@pytest.mark.parametrize("max_chars", [0, -1, 1.5, True])
def test_load_corpus_rejects_invalid_max_chars(tmp_path, max_chars) -> None:
    (tmp_path / "sample.txt").write_text("Sample text.", encoding="utf-8")

    with pytest.raises(ValueError, match="max_chars must be a positive integer"):
        load_corpus(tmp_path, max_chars=max_chars)


def test_training_pipeline_saves_reloadable_tokenizer(tmp_path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    (input_dir / "stories.txt").write_text(
        "banana banana\napple banana\nbanana apple\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "processed" / "tokenizer.json"

    tokenizer = train_from_directory(
        input_dir=input_dir,
        output_path=output_path,
        vocab_size=280,
        min_frequency=2,
    )

    assert output_path.exists()

    loaded = ByteBPETokenizer.load(output_path)
    sample = "banana apple"

    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.merges == tokenizer.merges
    assert loaded.encode(sample) == tokenizer.encode(sample)
    assert loaded.decode(loaded.encode(sample)) == sample
