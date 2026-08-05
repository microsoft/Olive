# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

import pytest
import torch
from datasets import Dataset

from olive.data.component.text_generation import TextGenStrategy, text_gen_pre_process


@pytest.fixture(name="propagate_olive_logs", autouse=True)
def propagate_olive_logs_fixture():
    """Olive disables propagation on its root logger, so caplog cannot see the records without this.

    This mirrors the inline propagation toggling already used elsewhere in the suite, e.g. in
    test/engine/test_engine.py, test/common/test_copy_dir.py and test/hardware/test_accelerator.py.
    """
    logger = logging.getLogger("olive")
    original = logger.propagate
    logger.propagate = True
    yield
    logger.propagate = original


class FakeEncoding(dict):
    """Dict that also supports attribute access, like transformers' BatchEncoding."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class FakeTokenizer:
    """Minimal word level tokenizer so the tests don't need to download a real tokenizer."""

    def __init__(self, pad_token_id: int = 0):
        self.padding_side = "right"
        self.pad_token_id = pad_token_id

    def encode(self, text, add_special_tokens=False):
        # only the number of tokens matters for these tests, the token ids themselves are arbitrary
        return [len(word) for word in text.split()]

    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        max_length=None,
        padding=False,
        return_tensors=None,
        **kwargs,
    ):
        texts = [text] if isinstance(text, str) else text
        input_ids = [self.encode(single_text) for single_text in texts]
        if truncation and max_length is not None:
            input_ids = [ids[:max_length] for ids in input_ids]
        attention_mask = [[1] * len(ids) for ids in input_ids]
        if padding == "max_length" and max_length is not None:
            attention_mask = [mask + [0] * (max_length - len(mask)) for mask in attention_mask]
            input_ids = [ids + [self.pad_token_id] * (max_length - len(ids)) for ids in input_ids]
        if return_tensors == "pt":
            return FakeEncoding(input_ids=torch.tensor(input_ids), attention_mask=torch.tensor(attention_mask))
        if isinstance(text, str):
            return FakeEncoding(input_ids=input_ids[0], attention_mask=attention_mask[0])
        return FakeEncoding(input_ids=input_ids, attention_mask=attention_mask)


def make_dataset(num_rows: int, words_per_row: int) -> Dataset:
    return Dataset.from_dict({"text": [" ".join(["word"] * words_per_row) for _ in range(num_rows)]})


def get_kwargs(max_samples, max_seq_len: int = 8, **kwargs) -> dict:
    return {
        "strategy": TextGenStrategy.JOIN,
        "add_special_tokens": False,
        "max_samples": max_samples,
        "max_seq_len": max_seq_len,
        "joiner": "",
        **kwargs,
    }


@pytest.mark.parametrize(
    ("strategy", "expected_step", "expected_required", "expected_samples"),
    [
        # JOIN: step is max_seq_len, so 8 + (10 - 1) * 8 = 80 tokens are needed, 16 tokens give 2 samples
        (TextGenStrategy.JOIN, 8, 80, 2),
        # JOIN_SLIDING_WINDOW: step is stride, so 8 + (10 - 1) * 4 = 44 tokens are needed, 16 tokens give 3 samples
        (TextGenStrategy.JOIN_SLIDING_WINDOW, 4, 44, 3),
    ],
)
def test_text_gen_pre_process_warns_when_join_corpus_exhausted_before_max_samples(
    strategy, expected_step, expected_required, expected_samples, caplog
):
    # setup
    # 4 rows x 4 tokens = 16 tokens
    dataset = make_dataset(num_rows=4, words_per_row=4)
    kwargs = get_kwargs(max_samples=10, strategy=strategy, stride=4)

    # execute
    with caplog.at_level(logging.WARNING, logger="olive.data.component.text_generation"):
        result = text_gen_pre_process(dataset, FakeTokenizer(), kwargs)

    # assert
    assert len(result) == expected_samples
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert f"Only {expected_samples} samples were generated" in message
    assert "max_samples=10" in message
    assert f"step={expected_step}" in message
    assert f"= {expected_required} tokens" in message


def test_text_gen_pre_process_warns_with_stride_remedy_when_strategy_is_sliding_window(caplog):
    # setup
    dataset = make_dataset(num_rows=4, words_per_row=4)
    kwargs = get_kwargs(max_samples=10, strategy=TextGenStrategy.JOIN_SLIDING_WINDOW, stride=4)

    # execute
    with caplog.at_level(logging.WARNING, logger="olive.data.component.text_generation"):
        text_gen_pre_process(dataset, FakeTokenizer(), kwargs)

    # assert
    assert "lower stride" in caplog.records[0].getMessage()


def test_text_gen_pre_process_delivers_max_samples_when_corpus_is_an_exact_fit(caplog):
    # setup
    # 4 rows x 8 tokens = 32 tokens, exactly max_samples x max_seq_len = 4 x 8
    dataset = make_dataset(num_rows=4, words_per_row=8)
    kwargs = get_kwargs(max_samples=4)

    # execute
    with caplog.at_level(logging.WARNING, logger="olive.data.component.text_generation"):
        result = text_gen_pre_process(dataset, FakeTokenizer(), kwargs)

    # assert
    assert len(result) == 4
    assert all(len(sample[0]["input_ids"]) == 8 for sample in result)
    assert not caplog.records


def test_text_gen_pre_process_warns_when_random_strategy_cannot_fill_max_samples(caplog):
    # setup
    # each row is shorter than max_seq_len and there is only one row, so no sample can be built
    dataset = make_dataset(num_rows=1, words_per_row=2)
    kwargs = get_kwargs(max_samples=3, strategy=TextGenStrategy.JOIN_RANDOM, random_seed=0, random_retries=5)

    # execute
    with caplog.at_level(logging.WARNING, logger="olive.data.component.text_generation"):
        result = text_gen_pre_process(dataset, FakeTokenizer(), kwargs)

    # assert
    assert len(result) == 0
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "max_samples=3" in message
    assert "random_retries=5" in message
    # max_samples must not be presented as part of the token requirement for the random strategy
    assert "max_samples does not increase that requirement" in message
    assert "lower max_samples" not in message


@pytest.mark.parametrize(
    "strategy",
    [TextGenStrategy.JOIN, TextGenStrategy.JOIN_SLIDING_WINDOW],
)
def test_text_gen_pre_process_does_not_warn_when_max_samples_delivered(strategy, caplog):
    # setup
    # 20 rows x 8 tokens = 160 tokens, more than enough for 4 samples with either step
    dataset = make_dataset(num_rows=20, words_per_row=8)
    kwargs = get_kwargs(max_samples=4, strategy=strategy, stride=4)

    # execute
    with caplog.at_level(logging.WARNING, logger="olive.data.component.text_generation"):
        result = text_gen_pre_process(dataset, FakeTokenizer(), kwargs)

    # assert
    assert len(result) == 4
    assert not caplog.records


def test_text_gen_pre_process_does_not_warn_when_max_samples_is_none(caplog):
    # setup
    dataset = make_dataset(num_rows=4, words_per_row=4)
    kwargs = get_kwargs(max_samples=None)

    # execute
    with caplog.at_level(logging.WARNING, logger="olive.data.component.text_generation"):
        text_gen_pre_process(dataset, FakeTokenizer(), kwargs)

    # assert
    assert not caplog.records


def test_text_gen_pre_process_warns_when_line_by_line_has_too_few_rows(caplog):
    # setup
    # only 3 usable rows for 5 requested samples
    dataset = make_dataset(num_rows=3, words_per_row=8)
    kwargs = get_kwargs(max_samples=5, strategy=TextGenStrategy.LINE_BY_LINE)

    # execute
    with caplog.at_level(logging.WARNING, logger="olive.data.component.text_generation"):
        result = text_gen_pre_process(dataset, FakeTokenizer(), kwargs)

    # assert
    assert len(result) == 3
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "Only 3 samples were generated" in message
    assert "3 non-empty rows" in message
    assert "lower max_samples" in message


def test_text_gen_pre_process_warns_when_line_by_line_drops_short_rows(caplog):
    # setup
    # every row is shorter than max_seq_len and drop_short_sequences filters them all out
    dataset = make_dataset(num_rows=5, words_per_row=2)
    kwargs = get_kwargs(
        max_samples=5,
        strategy=TextGenStrategy.LINE_BY_LINE,
        pad_to_max_len=False,
        drop_short_sequences=True,
    )

    # execute
    with caplog.at_level(logging.WARNING, logger="olive.data.component.text_generation"):
        result = text_gen_pre_process(dataset, FakeTokenizer(), kwargs)

    # assert
    assert len(result) == 0
    assert "drop_short_sequences=True" in caplog.records[0].getMessage()


def test_text_gen_pre_process_does_not_warn_when_line_by_line_delivers_max_samples(caplog):
    # setup
    dataset = make_dataset(num_rows=5, words_per_row=8)
    kwargs = get_kwargs(max_samples=5, strategy=TextGenStrategy.LINE_BY_LINE)

    # execute
    with caplog.at_level(logging.WARNING, logger="olive.data.component.text_generation"):
        result = text_gen_pre_process(dataset, FakeTokenizer(), kwargs)

    # assert
    assert len(result) == 5
    assert not caplog.records


def test_text_gen_pre_process_warns_when_line_by_line_random_finds_no_sample(caplog):
    # setup
    # random_retries=0 means no row is ever accepted, so every sample is skipped
    dataset = make_dataset(num_rows=5, words_per_row=2)
    kwargs = get_kwargs(
        max_samples=4,
        strategy=TextGenStrategy.LINE_BY_LINE_RANDOM,
        random_seed=0,
        random_retries=0,
        pad_to_max_len=False,
        drop_short_sequences=True,
    )

    # execute
    with caplog.at_level(logging.WARNING, logger="olive.data.component.text_generation"):
        result = text_gen_pre_process(dataset, FakeTokenizer(), kwargs)

    # assert
    assert len(result) == 0
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "max_samples=4" in message
    assert "random_retries=0" in message


def test_text_gen_pre_process_drops_short_rows_when_line_by_line_random_exhausts_retries(caplog):
    # setup
    # every row is shorter than max_seq_len, so all random_retries attempts are rejected. The loop variable
    # still holds the last rejected row, which used to be emitted because a BatchEncoding is always truthy.
    dataset = make_dataset(num_rows=5, words_per_row=2)
    kwargs = get_kwargs(
        max_samples=4,
        strategy=TextGenStrategy.LINE_BY_LINE_RANDOM,
        random_seed=0,
        random_retries=3,
        pad_to_max_len=False,
        drop_short_sequences=True,
    )

    # execute
    with caplog.at_level(logging.WARNING, logger="olive.data.component.text_generation"):
        result = text_gen_pre_process(dataset, FakeTokenizer(), kwargs)

    # assert
    assert len(result) == 0
    assert len(caplog.records) == 1
    assert "drop_short_sequences=True" in caplog.records[0].getMessage()


def test_text_gen_pre_process_does_not_warn_when_line_by_line_random_delivers_max_samples(caplog):
    # setup
    dataset = make_dataset(num_rows=5, words_per_row=8)
    kwargs = get_kwargs(max_samples=4, strategy=TextGenStrategy.LINE_BY_LINE_RANDOM, random_seed=0)

    # execute
    with caplog.at_level(logging.WARNING, logger="olive.data.component.text_generation"):
        result = text_gen_pre_process(dataset, FakeTokenizer(), kwargs)

    # assert
    assert len(result) == 4
    assert not caplog.records
