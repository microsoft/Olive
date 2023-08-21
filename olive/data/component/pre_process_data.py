# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from copy import deepcopy
from enum import Enum

import torch
from pydantic import validator

from olive.common.config_utils import ConfigBase
from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry

IGNORE_INDEX = -100


@Registry.register_default_pre_process()
def pre_process(_dataset):
    """Pre-process data.

    Args:
        data (object): Data to be pre-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Pre-processed data.
    """
    return _dataset


def _huggingface_pre_process_helper(dataset, model_name, input_cols, label_cols, map_func, **kwargs):
    """Pre-process data.

    Args:
        data (object): Data to be pre-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Pre-processed data.
    """
    # output type is list
    tokenized_datasets = dataset.map(
        map_func,
        batched=kwargs.get("batched", True),
        remove_columns=dataset.column_names,
    )
    tokenized_datasets.set_format("torch", output_all_columns=True)
    return tokenized_datasets


@Registry.register_pre_process()
def huggingface_pre_process(_dataset, model_name, input_cols, label_cols, max_samples=None, **kwargs):
    """Pre-process data.

    Args:
        data (object): Data to be pre-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Pre-processed data.
    """
    from transformers import AutoTokenizer

    def _tokenizer_and_align_labels(examples):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_inputs = tokenizer(
            *[examples[input_col] for input_col in input_cols],
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            is_split_into_words=kwargs.get("is_split_into_words", False),
            add_special_tokens=kwargs.get("add_special_tokens", True),
        )
        # TODO: support multiple label columns if needed
        tokenized_inputs["label"] = examples[label_cols[0]]
        # huggingface dataset api limit to return dict and arrow table
        return tokenized_inputs

    tokenized_datasets = _huggingface_pre_process_helper(
        _dataset, model_name, input_cols, label_cols, _tokenizer_and_align_labels, **kwargs
    )
    # label_cols is ["label"] since we added label_cols[0] as "label" to tokenized_inputs
    return BaseDataset(tokenized_datasets, label_cols=["label"], max_samples=max_samples)


@Registry.register_pre_process()
def ner_huggingface_preprocess(_dataset, model_name, input_cols, label_cols, max_samples=None, **kwargs):
    """
    Pre-process data for ner task.
    """
    from transformers import AutoTokenizer

    def _align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = 0 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(0)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)
        return new_labels

    def _tokenizer_and_align_labels(examples):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_inputs = tokenizer(
            *[examples[input_col] for input_col in input_cols],
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            is_split_into_words=kwargs.get("is_split_into_words", True),
            add_special_tokens=kwargs.get("add_special_tokens", False),
        )
        all_labels = examples[label_cols[0]]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(_align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["label"] = new_labels
        return tokenized_inputs

    tokenized_datasets = _huggingface_pre_process_helper(
        _dataset, model_name, input_cols, label_cols, _tokenizer_and_align_labels, **kwargs
    )
    return BaseDataset(tokenized_datasets, label_cols=["label"], max_samples=max_samples)


class TextGenDatasetType(str, Enum):
    """Text generation dataset type."""

    CORPUS = "corpus"  # single text source, e.g. a book
    PAIR = "pair"  # two text sources, e.g. an input and an output


class TextGenCorpusStrategy(str, Enum):
    """Strategy for tokenizing a corpus."""

    LINE_BY_LINE = "line-by-line"  # each line is a sequence, in order of appearance
    LINE_BY_LINE_RANDOM = "line-by-line-random"  # each line is a sequence, in random order
    JOIN = "join"  # join all lines into a single sequence, split into non-overlapping sequences
    JOIN_RANDOM = "join-random"  # join all lines into a single sequence, split into random sequences
    JOIN_SLIDING_WINDOW = (  # join all lines into a single sequence, split into overlapping sequences
        "join-sliding-window"
    )


class TextGenPairFormat(str, Enum):
    """Format of the pair dataset."""

    # ALPACA format, https://huggingface.co/datasets/tatsu-lab/alpaca
    # instruction, input (optional), output
    ALPACA = "alpaca"
    # OIG chip2 format, https://huggingface.co/datasets/laion/OIG unified_chip2.jsonl
    # "<human>: ...\n<bot>:..."
    CHIP2 = "chip2"
    # self_instruct format, https://huggingface.co/datasets/yizhongw/self_instruct/viewer/self_instruct
    # prompt, completion
    SELF_INSTRUCT = "self_instruct"
    # default format: input, output
    DEFAULT = "default"
    # custom, user-defined format. Must provide input_col and output_col in kwargs
    CUSTOM = "custom"


class TextGenParams(ConfigBase):
    max_samples: int = None
    source_max_len: int
    # TODO: currently only support padding to max length since we preprocess all data at once
    # might have to expose collator for dataloader to support dynamic padding of batches
    # if false, cannot gaurantee all sequences are same length. data loader will have to handle this
    pad_to_max_len: bool = True  # pad sequences to max_len, ignored for JOIN corpus strategy
    drop_short_sequences: bool = False  # drop sequences shorter than max_len. Mutually exclusive with pad_to_max_len
    add_special_tokens: bool = True  # add special tokens, ignored for JOIN corpus strategy

    @validator("drop_short_sequences", always=True)
    def _check_padding(cls, v, values):
        if "pad_to_max_len" not in values:
            ValueError("Invalid pad_to_max_len")
        if v and values["pad_to_max_len"]:
            raise ValueError("pad_to_max_len and drop_short_sequences cannot both be True")
        return v


class TextGenCorpusParams(TextGenParams):
    text_cols: list  # list of text columns
    corpus_strategy: TextGenCorpusStrategy = TextGenCorpusStrategy.LINE_BY_LINE
    stride: int = None  # required when corpus_strategy is JOIN_SLIDING_WINDOW
    joiner: str = " "  # delimiter to use when joining the rows of the input columns.
    random_seed: int = None  # random seed for LINE_BY_LINE_RANDOM and JOIN_RANDOM
    random_retries: int = (
        10  # number of resamples to try before giving up when a sample is too short for RANDOM strategies
    )

    @validator("stride", always=True)
    def _check_stride(cls, v, values):
        if "corpus_strategy" not in values:
            raise ValueError("Invalid corpus_strategy")
        if values["corpus_strategy"] == TextGenCorpusStrategy.JOIN_SLIDING_WINDOW and v is None:
            raise ValueError("stride must be specified when corpus_strategy is JOIN_SLIDING_WINDOW")
        return v

    @validator("corpus_strategy", always=True)
    def _check_max_samples(cls, v, values):
        if "max_samples" not in values:
            raise ValueError("Invalid max_samples")
        if "random" in v and values["max_samples"] is None:
            raise ValueError("max_samples must be specified when corpus_strategy is random")
        return v

    @validator("random_seed", always=True)
    def _check_random(cls, v, values):
        if "corpus_strategy" not in values:
            raise ValueError("Invalid corpus_strategy")
        if "random" in values["corpus_strategy"] and v is None:
            raise ValueError("random_seed must be specified when corpus_strategy is random")
        return v


@Registry.register_pre_process()
def text_generation_huggingface_pre_process(
    _dataset, model_name: str, dataset_type: TextGenDatasetType, source_max_len: int, max_samples=None, **kwargs
):
    from transformers import AutoTokenizer

    all_kwargs = deepcopy(kwargs)
    all_kwargs.update({"max_samples": max_samples, "source_max_len": source_max_len})

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if dataset_type == TextGenDatasetType.CORPUS:
        return text_gen_corpus_pre_process(_dataset, tokenizer, all_kwargs)
    else:
        raise NotImplementedError(f"dataset_type {dataset_type} not implemented yet")


def text_gen_corpus_pre_process(_dataset, tokenizer, all_kwargs):
    args = TextGenCorpusParams(**all_kwargs)
    from random import Random

    from datasets import Dataset as HFDataset

    # gather text from all input columns
    text_list = []
    for input_col in args.text_cols:
        text_list += _dataset[input_col]

    tokenized_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "target_ids": [],
    }

    if "join" in args.corpus_strategy:
        # delimiter between the text sequences
        text = args.joiner.join(text_list)

        # in order to make processing faster we will only tokenize as much as needed
        # assumes that num words > num tokens
        split_text = text.split(" ")
        num_text = len(split_text)

        seqlen = args.source_max_len

        if args.corpus_strategy != TextGenCorpusStrategy.JOIN_RANDOM:
            # no randomization, just use contiguous blocks of tokens
            if args.corpus_strategy == TextGenCorpusStrategy.JOIN_SLIDING_WINDOW:
                step, context = args.stride, args.stride
            else:
                step, context = seqlen, None

            max_text = args.max_samples * seqlen if args.max_samples is not None else num_text
            max_text = min(max_text, num_text)
            encodings = tokenizer(" ".join(split_text[:max_text]), add_special_tokens=False, return_tensors="pt")

            num_tokens = encodings.input_ids.shape[1]
            # loop over the number of tokens
            # all inputs must be seqlen long
            for begin_loc in range(0, num_tokens - seqlen, step):
                # end_loc is the beginning of the next sequence
                end_loc = begin_loc + seqlen
                # get the input sequence
                input_ids = encodings.input_ids[0, begin_loc:end_loc].clone()
                _append_text_gen_input_ids(tokenized_inputs, input_ids, tokenizer, context=context)
        else:
            # randomization, sample random blocks of tokens
            rng = Random(args.random_seed)
            cache = {}

            for _ in range(args.max_samples):
                resamples = 0
                encodings = None
                while resamples < args.random_retries:
                    # randint is inclusive, so we need to subtract 1
                    begin_loc = rng.randint(0, num_text - seqlen - 1)
                    # heuristic to make sure we don't get a sequence that is too short
                    if begin_loc not in cache:
                        encodings = tokenizer(
                            " ".join(split_text[begin_loc : begin_loc + seqlen]),  # noqa E203
                            add_special_tokens=False,
                            return_tensors="pt",
                        )
                        cache[begin_loc] = encodings
                    else:
                        encodings = cache[begin_loc]
                    if encodings.input_ids.shape[1] >= seqlen:
                        # found a good sample
                        break
                    resamples += 1
                if not encodings:
                    continue
                input_ids = encodings.input_ids[0, :seqlen]
                _append_text_gen_input_ids(tokenized_inputs, input_ids, tokenizer)

    else:
        # each line is a sequence
        num_samples = 0
        if args.corpus_strategy == TextGenCorpusStrategy.LINE_BY_LINE:
            for text in text_list:
                encodings = tokenizer(
                    text,
                    max_length=args.source_max_len,
                    truncation=True,
                    padding="max_length" if args.pad_to_max_len else False,
                    add_special_tokens=args.add_special_tokens,
                    return_tensors="pt",
                )
                if args.drop_short_sequences and encodings.input_ids.shape[1] < args.source_max_len:
                    continue
                _append_text_gen_input_ids(tokenized_inputs, encodings.input_ids[0], tokenizer)
                num_samples += 1
                if args.max_samples is not None and num_samples >= args.max_samples:
                    break
        else:
            # randomization, sample random lines
            rng = Random(args.random_seed)
            cache = {}
            for _ in range(args.max_samples):
                resamples = 0
                encodings = None
                while resamples < args.random_retries:
                    i = rng.randint(0, len(text_list) - 1)
                    if i not in cache:
                        encodings = tokenizer(
                            text_list[i],
                            max_length=args.source_max_len,
                            truncation=True,
                            padding="max_length" if args.pad_to_max_len else False,
                            add_special_tokens=args.add_special_tokens,
                            return_tensors="pt",
                        )
                        cache[i] = encodings
                    else:
                        encodings = cache[i]
                    if not args.drop_short_sequences or encodings.input_ids.shape[1] >= args.source_max_len:
                        break
                    resamples += 1
                if not encodings:
                    continue
                _append_text_gen_input_ids(tokenized_inputs, encodings.input_ids[0], tokenizer)

    # convert to HFDataset
    hf_dataset = HFDataset.from_dict(tokenized_inputs)
    hf_dataset.set_format("torch", output_all_columns=True)

    # return BaseDataset
    return BaseDataset(hf_dataset, ["target_ids"], max_samples=args.max_samples)


def _append_text_gen_input_ids(tokenized_inputs, input_ids, tokenizer, context: int = None, ignore_index=IGNORE_INDEX):
    """Convert input_ids to inputs dict and append to tokenized_inputs."""
    inputs = {"input_ids": input_ids}

    # create attention_mask
    attention_mask = (
        torch.ones_like(input_ids) if tokenizer.pad_token_id is None else input_ids.ne(tokenizer.pad_token_id)
    )
    inputs["attention_mask"] = attention_mask

    # create target_ids
    # target is not shifted by 1 since causal lm models shifts internally when computing loss
    target_ids = input_ids.clone()
    # set context to ignore_index
    if context is not None:
        target_ids[:-context] = ignore_index
    # set padding to ignore_index
    target_ids[attention_mask != 1] = ignore_index
    inputs["target_ids"] = target_ids

    # add to list
    for k, v in inputs.items():
        tokenized_inputs[k].append(v)
