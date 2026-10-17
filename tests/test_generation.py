"""Tests for ScratchGPT text generation."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scratchgpt.config import GPTConfig
from scratchgpt.generation import generate_tokens
from scratchgpt.model import GPTModel


class DummyModel(nn.Module):
    """Small predictable model for testing generation behavior."""

    def __init__(self, context_length=4, vocab_size=8):
        super().__init__()
        self.config = SimpleNamespace(context_length=context_length)
        self.vocab_size = vocab_size
        self.seen_contexts = []

    def forward(self, input_ids):
        self.seen_contexts.append(input_ids.clone())

        logits = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            self.vocab_size,
            device=input_ids.device,
        )

        # Token 1 is always the highest-scoring next token.
        logits[:, :, 1] = 10.0

        return logits


@pytest.fixture
def small_model():
    config = GPTConfig(
        vocab_size=32,
        context_length=8,
        embedding_dim=16,
        num_layers=2,
        num_heads=4,
        feed_forward_dim=32,
        dropout=0.0,
    )

    return GPTModel(config)


@pytest.fixture
def input_ids():
    return torch.tensor([[2, 5, 7]], dtype=torch.long)


def test_generation_appends_requested_number_of_tokens(
    input_ids,
):
    model = DummyModel()

    output = generate_tokens(
        model,
        input_ids,
        max_new_tokens=3,
        do_sample=False,
    )

    assert output.shape == (1, 6)
    assert torch.equal(output[:, :3], input_ids)
    assert torch.equal(output[:, 3:], torch.tensor([[1, 1, 1]]))


def test_generation_preserves_input_tensor(input_ids):
    model = DummyModel()
    original = input_ids.clone()

    generate_tokens(
        model,
        input_ids,
        max_new_tokens=2,
        do_sample=False,
    )

    assert torch.equal(input_ids, original)


def test_zero_new_tokens_returns_unchanged_copy(input_ids):
    model = DummyModel()

    output = generate_tokens(
        model,
        input_ids,
        max_new_tokens=0,
    )

    assert torch.equal(output, input_ids)
    assert output is not input_ids
    assert model.seen_contexts == []


def test_greedy_decoding_selects_highest_logit(input_ids):
    model = DummyModel()

    output = generate_tokens(
        model,
        input_ids,
        max_new_tokens=4,
        do_sample=False,
    )

    assert output[0, -4:].tolist() == [1, 1, 1, 1]


def test_sampling_with_temperature_and_top_k(input_ids):
    model = DummyModel()

    output = generate_tokens(
        model,
        input_ids,
        max_new_tokens=5,
        temperature=0.8,
        top_k=2,
        do_sample=True,
    )

    assert output.shape == (1, 8)
    assert torch.equal(output[:, :3], input_ids)

    # Token 1 is the only token with a competitive score.
    assert output[0, -5:].tolist() == [1, 1, 1, 1, 1]


def test_generation_respects_context_window():
    model = DummyModel(context_length=4)
    input_ids = torch.tensor([[2, 3, 4, 5]])

    output = generate_tokens(
        model,
        input_ids,
        max_new_tokens=3,
        do_sample=False,
    )

    assert output.shape == (1, 7)
    assert len(model.seen_contexts) == 3

    assert model.seen_contexts[0].tolist() == [[2, 3, 4, 5]]
    assert model.seen_contexts[1].tolist() == [[3, 4, 5, 1]]
    assert model.seen_contexts[2].tolist() == [[4, 5, 1, 1]]

    assert all(context.shape[1] <= 4 for context in model.seen_contexts)


def test_generation_supports_a_batch():
    model = DummyModel()
    input_ids = torch.tensor([[2, 3], [4, 5]])

    output = generate_tokens(
        model,
        input_ids,
        max_new_tokens=2,
        do_sample=False,
    )

    assert output.shape == (2, 4)
    assert output[:, -2:].tolist() == [[1, 1], [1, 1]]


def test_generation_disables_gradients_and_restores_training_mode(
    small_model,
    input_ids,
):
    small_model.train()

    output = generate_tokens(
        small_model,
        input_ids,
        max_new_tokens=2,
        do_sample=False,
    )

    assert output.shape == (1, 5)
    assert small_model.training is True


def test_generation_restores_evaluation_mode(input_ids):
    model = DummyModel()
    model.eval()

    generate_tokens(
        model,
        input_ids,
        max_new_tokens=1,
        do_sample=False,
    )

    assert model.training is False


def test_generation_works_with_real_gpt_model(
    small_model,
    input_ids,
):
    output = generate_tokens(
        small_model,
        input_ids,
        max_new_tokens=2,
        do_sample=False,
    )

    assert output.shape == (1, 5)
    assert output.dtype == input_ids.dtype
    assert torch.equal(output[:, :3], input_ids)


@pytest.mark.parametrize("max_new_tokens", [-1, 1.5, True])
def test_invalid_max_new_tokens_raises(max_new_tokens, input_ids):
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_tokens(
            DummyModel(),
            input_ids,
            max_new_tokens=max_new_tokens,
        )


@pytest.mark.parametrize("temperature", [0, -1, float("inf"), float("nan")])
def test_invalid_temperature_raises(temperature, input_ids):
    with pytest.raises(ValueError, match="temperature"):
        generate_tokens(
            DummyModel(),
            input_ids,
            max_new_tokens=1,
            temperature=temperature,
        )


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_top_k_raises(top_k, input_ids):
    with pytest.raises(ValueError, match="top_k"):
        generate_tokens(
            DummyModel(),
            input_ids,
            max_new_tokens=1,
            top_k=top_k,
        )


def test_top_k_cannot_exceed_vocabulary(input_ids):
    with pytest.raises(ValueError, match="vocabulary size"):
        generate_tokens(
            DummyModel(vocab_size=8),
            input_ids,
            max_new_tokens=1,
            top_k=9,
        )


@pytest.mark.parametrize(
    "invalid_input",
    [
        torch.tensor([1, 2, 3]),
        torch.empty((0, 3), dtype=torch.long),
        torch.empty((1, 0), dtype=torch.long),
        torch.tensor([[1.0, 2.0]]),
    ],
)
def test_invalid_input_ids_raise(invalid_input):
    with pytest.raises((ValueError, TypeError)):
        generate_tokens(
            DummyModel(),
            invalid_input,
            max_new_tokens=1,
        )


def test_invalid_sampling_flag_raises(input_ids):
    with pytest.raises(TypeError, match="do_sample"):
        generate_tokens(
            DummyModel(),
            input_ids,
            max_new_tokens=1,
            do_sample=1,
        )
