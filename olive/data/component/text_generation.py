# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from enum import Enum
from random import Random

import torch
from pydantic import validator

from olive.common.config_utils import ConfigBase, validate_config
from olive.data.component.dataset import BaseDataset
from olive.data.constants import IGNORE_INDEX

logger = logging.getLogger(__name__)


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
    """
    Common parameters for text generation tasks.

    Base dataclass for text generation tasks.
    """

    max_samples: int = None  # max number of samples to use, None for all
    source_max_len: int  # max length of source sequence
    # TODO: currently only support padding to max length since we preprocess all data at once
    # might have to expose collator for dataloader to support dynamic padding of batches
    # if false, cannot gaurantee all sequences are same length. data loader will have to handle this during collation
    pad_to_max_len: bool = True  # pad sequences to max_len, ignored for JOIN corpus strategy
    padding_side: str = "right"  # side to pad sequences
    # use eos token as pad token, only valid if the tokenizer does not have a pad token
    # won't mask the actual eos token, so the model can still generate eos tokens
    use_eos_token_as_pad_token: bool = False
    drop_short_sequences: bool = False  # drop sequences shorter than max_len. Mutually exclusive with pad_to_max_len
    add_special_tokens: bool = True  # add bos and eos tokens

    @validator("drop_short_sequences", always=True)
    def _check_padding(cls, v, values):
        if "pad_to_max_len" not in values:
            ValueError("Invalid pad_to_max_len")
        if v and values["pad_to_max_len"]:
            raise ValueError("pad_to_max_len and drop_short_sequences cannot both be True")
        return v


class TextGenCorpusParams(TextGenParams):
    """Parameters for text generation task with 'corpus' dataset type."""

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


class TextGenPairParams(TextGenParams):
    """Parameters for text generation task with 'pair' dataset type."""

    pair_format: TextGenPairFormat = TextGenPairFormat.DEFAULT
    input_col: str = None  # required when pair_format is CUSTOM
    output_col: str = None  # required when pair_format is CUSTOM
    target_max_len: int  # max length of target sequence
    ignore_source_in_labels: bool = True  # set source tokens to ignore_index in labels

    @validator("input_col", "output_col", always=True)
    def _check_custom(cls, v, field, values):
        if "pair_format" not in values:
            raise ValueError("Invalid pair_format")
        if values["pair_format"] == TextGenPairFormat.CUSTOM and v is None:
            raise ValueError(f"{field.name} must be specified when pair_format is CUSTOM")
        return v


def text_gen_corpus_pre_process(_dataset, model_name, all_kwargs):
    """
    Pre-process data for text generation task with 'corpus' dataset type.

    The input dataset is expected to have one or more text columns.
    Depending on the corpus_strategy, the sequences are either joined together or processed individually.
    """

    from datasets import Dataset as HFDataset

    args = validate_config(all_kwargs, TextGenCorpusParams, warn_unused_keys=True)

    # no need for pad_token in join strategies, won't be used
    tokenizer = get_tokenizer(model_name, args, fail_on_no_pad_token="join" not in args.corpus_strategy)

    # gather text from all input columns
    text_list = []
    for input_col in args.text_cols:
        text_list += _dataset[input_col]

    tokenized_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
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
                # we use the stride as both the step between sequences and the context size
                step, context = args.stride, args.stride
            else:
                # JOIN corpus_strategy
                # text is split into non-overlapping sequences and there is no context
                step, context = seqlen, None

            # only take as much text as needed
            # assumes that num words > num tokens, so we can use num tokens as an upper bound
            max_text = args.max_samples * seqlen if args.max_samples is not None else num_text
            max_text = min(max_text, num_text)
            # tokenize the text
            encodings = tokenizer(" ".join(split_text[:max_text]), add_special_tokens=False, return_tensors="pt")

            num_tokens = encodings.input_ids.shape[1]
            # loop over the number of tokens
            # all inputs must be seqlen long
            for begin_loc in range(0, num_tokens - seqlen, step):
                # end_loc is the beginning of the next sequence
                end_loc = begin_loc + seqlen
                # get the input sequence
                input_ids = encodings.input_ids[0, begin_loc:end_loc].clone()
                attention_mask = torch.ones_like(input_ids)
                append_text_gen_input_ids(tokenized_inputs, input_ids, attention_mask, context=context)
        else:
            # randomization, sample random blocks of tokens
            rng = Random(args.random_seed)
            cache = {}
            for _ in range(args.max_samples):
                resamples = 0
                encodings = None
                while resamples < args.random_retries:
                    # sample a random block of tokens by sampling a random starting location
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
                    # could not find a good sample after resampling
                    continue
                input_ids = encodings.input_ids[0, :seqlen]
                attention_mask = torch.ones_like(input_ids)
                append_text_gen_input_ids(tokenized_inputs, input_ids, attention_mask)

    else:
        if args.add_special_tokens:
            # add bos and eos tokens before tokenizing
            # some tokenizers like LlamaTokenizer do not add eos token
            text_list = [f"{tokenizer.bos_token} {text} {tokenizer.eos_token}" for text in text_list]

        # each line is a sequence
        if args.corpus_strategy == TextGenCorpusStrategy.LINE_BY_LINE:
            # batched tokenization might be faster so lets tokenize all the text at once
            if not args.max_samples:
                batched_encodings = batch_tokenize_text(text_list, tokenizer, args)
                for input_ids, attention_mask in batched_encodings:
                    append_text_gen_input_ids(tokenized_inputs, input_ids, attention_mask)
            else:
                total_samples = len(text_list)
                num_samples = 0
                begin_loc = 0
                while True:
                    if num_samples >= args.max_samples or begin_loc >= total_samples:
                        # we have reached max_samples or the end of the text_list
                        break
                    # get as many samples as possible without going over max_samples
                    samples_to_get = min(args.max_samples - num_samples, total_samples - begin_loc)
                    # batch tokenize
                    batched_encodings = batch_tokenize_text(
                        text_list[begin_loc : begin_loc + samples_to_get], tokenizer, args  # noqa E203
                    )
                    for input_ids, attention_mask in batched_encodings:
                        append_text_gen_input_ids(tokenized_inputs, input_ids, attention_mask)
                    # update counters
                    num_samples += len(batched_encodings)
                    begin_loc += samples_to_get
        else:
            # randomization, sample random lines
            rng = Random(args.random_seed)
            cache = {}
            for _ in range(args.max_samples):
                resamples = 0
                encodings = None
                while resamples < args.random_retries:
                    # sample a random line
                    # randint is inclusive, so we need to subtract 1
                    i = rng.randint(0, len(text_list) - 1)
                    if i not in cache:
                        encodings = tokenizer(
                            text_list[i],
                            max_length=args.source_max_len,
                            truncation=True,
                            padding="max_length" if args.pad_to_max_len else False,
                            add_special_tokens=False,
                            return_tensors="pt",  # this is fine since there is only one sample
                        )
                        cache[i] = encodings
                    else:
                        encodings = cache[i]
                    if not args.drop_short_sequences or encodings.input_ids.shape[1] >= args.source_max_len:
                        # found a good sample
                        break
                    encodings = None
                    resamples += 1
                if not encodings:
                    # could not find a good sample after resampling
                    continue
                append_text_gen_input_ids(tokenized_inputs, encodings.input_ids[0], encodings.attention_mask[0])

    # convert to HFDataset
    hf_dataset = HFDataset.from_dict(tokenized_inputs)
    hf_dataset.set_format("torch", output_all_columns=True)

    # return BaseDataset
    return BaseDataset(hf_dataset, ["labels"], max_samples=args.max_samples)


def batch_tokenize_text(text_list, tokenizer, args):
    """Batch tokenize text."""
    batched_encodings = tokenizer(
        text_list,
        max_length=args.source_max_len,
        truncation=True,
        padding="max_length" if args.pad_to_max_len else False,
        add_special_tokens=False,
    )
    batched_encodings = [
        (torch.tensor(input_ids), torch.tensor(attention_mask))
        for input_ids, attention_mask in zip(batched_encodings.input_ids, batched_encodings.attention_mask)
    ]
    if args.drop_short_sequences:
        batched_encodings = [
            (input_ids, attention_mask)
            for input_ids, attention_mask in batched_encodings
            if len(input_ids) >= args.source_max_len
        ]
    return batched_encodings


# based on https://github.com/artidoro/qlora/blob/main/qlora.py
def text_gen_pair_pre_process(_dataset, model_name, all_kwargs):
    """
    Pre-process data for text generation task with 'pair' dataset type.

    Dataset is expected to have two text columns: input and output.
    An example is a dataset with pairs of prompts and completions.

    The input (truncate to source_max_len) and output (truncate to target_max_len) are concatenated together.
    """
    from datasets import Dataset as HFDataset

    args = validate_config(all_kwargs, TextGenPairParams, warn_unused_keys=True)

    tokenizer = get_tokenizer(model_name, args)

    # format dataset based on pair_format
    # the formatted dataset has two columns: input and output
    dataset = format_pair_dataset(_dataset, args.pair_format, args.input_col, args.output_col)
    if args.max_samples is not None:
        # truncate dataset to max_samples
        # makes tokenization faster
        dataset = dataset.select(range(args.max_samples))

    # extract elements
    sources = dataset["input"]
    targets = dataset["output"]
    if args.add_special_tokens:
        # add bos and eos tokens
        # input and output are concatenated, so add the bos and eos tokens to the input and output respectively
        sources = [f"{tokenizer.bos_token} {source}" for source in sources]
        targets = [f"{target} {tokenizer.eos_token}" for target in targets]

    # tokenize
    tokenized_sources = tokenizer(sources, max_length=args.source_max_len, truncation=True, add_special_tokens=False)
    tokenized_targets = tokenizer(targets, max_length=args.target_max_len, truncation=True, add_special_tokens=False)

    # build tokenized_inputs
    tokenized_inputs = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    # max_len is the max length of the concatenated input and output
    # if pad_to_max_len is True, max_len is the max length of the concatenated input and output
    max_len = args.source_max_len + args.target_max_len
    for tokenized_source, tokenized_target in zip(tokenized_sources["input_ids"], tokenized_targets["input_ids"]):
        # concatenate input and output
        input_ids = torch.tensor(tokenized_source + tokenized_target)
        attention_mask = torch.ones_like(input_ids)
        if args.drop_short_sequences and input_ids.shape[0] < max_len:
            # skip short sequences if drop_short_sequences is True
            continue
        if args.pad_to_max_len:
            # add padding to max_len
            padding_size = max_len - input_ids.shape[0]
            pad = (0, padding_size) if args.padding_side == "right" else (padding_size, 0)
            input_ids = torch.nn.functional.pad(input_ids, pad, value=tokenizer.pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, pad, value=0)
        # if ignore_source_in_labels is True, the source tokens are treated as context and set to ignore_index in labels
        context = len(tokenized_source) if args.ignore_source_in_labels else None
        append_text_gen_input_ids(tokenized_inputs, input_ids, attention_mask, context=context)

    # convert to HFDataset
    hf_dataset = HFDataset.from_dict(tokenized_inputs)
    hf_dataset.set_format("torch", output_all_columns=True)

    return BaseDataset(hf_dataset, ["labels"], max_samples=args.max_samples)


def get_tokenizer(model_name, args, fail_on_no_pad_token=True):
    """Get tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=args.padding_side)

    if args.use_eos_token_as_pad_token:
        if tokenizer.pad_token:
            logger.warning(
                "use_eos_token_as_pad_token is True, but tokenizer has a pad_token. "
                "The pad_token will be used instead of the eos_token."
            )
        elif not tokenizer.eos_token:
            raise ValueError("use_eos_token_as_pad_token is True, but tokenizer does not have an eos_token")
        else:
            logger.debug("Using eos_token as pad_token")
            tokenizer.pad_token = tokenizer.eos_token

    if fail_on_no_pad_token and args.pad_to_max_len and not tokenizer.pad_token:
        raise ValueError(
            "pad_to_max_len is True, but tokenizer does not have a pad_token. Set use_eos_token_as_pad_token to True or"
            " add a pad_token to the tokenizer"
        )

    return tokenizer


def append_text_gen_input_ids(
    tokenized_inputs, input_ids, attention_mask=None, context: int = None, ignore_index=IGNORE_INDEX
):
    """Convert input_ids to inputs dict and append to tokenized_inputs."""
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    # create labels
    # target is not shifted by 1 since causal lm models shifts internally when computing loss
    labels = input_ids.clone()
    # set context to ignore_index
    if context is not None:
        labels[:context] = ignore_index
    # set padding to ignore_index
    labels[attention_mask != 1] = ignore_index
    inputs["labels"] = labels

    # add to list
    for k, v in inputs.items():
        tokenized_inputs[k].append(v)


# based on https://github.com/artidoro/qlora/blob/main/qlora.py
def format_pair_dataset(dataset, pair_format, input_col=None, output_col=None):
    """Format dataset based on pair_format."""

    # format for input in ALPACA pair format
    # instruction, input (optional), output
    ALPACA_PROMPT_DICT = {
        "prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
        ),
        "prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response: "
        ),
    }

    def extract_alpaca_dataset(example):
        # extract new input from instruction and input
        if example.get("input", "") != "":
            prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
        else:
            prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
        return {"input": prompt_format.format(**example)}

    if pair_format == TextGenPairFormat.ALPACA:
        dataset = dataset.map(extract_alpaca_dataset, remove_columns=["instruction"])
    elif pair_format == TextGenPairFormat.CHIP2:
        # separate the human and bot text into input and output
        dataset = dataset.map(
            lambda x: {
                "input": x["text"].split("\n<bot>: ")[0].replace("<human>: ", ""),
                "output": x["text"].split("\n<bot>: ")[1],
            }
        )
    elif pair_format == TextGenPairFormat.SELF_INSTRUCT:
        # rename prompt and completion to input and output
        dataset = dataset.map(
            lambda x: {
                "input": x["prompt"],
                "output": x["completion"],
            }
        )
    elif pair_format == TextGenPairFormat.CUSTOM:
        # rename input_col and output_col to input and output
        dataset = dataset.map(
            lambda x: {
                "input": x[input_col],
                "output": x[output_col],
            }
        )
    elif pair_format == TextGenPairFormat.DEFAULT:
        # do nothing
        pass
    else:
        raise ValueError(f"Invalid pair_format: {pair_format}")

    # remove unused columns, keep only input and output
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ["input", "output"]])
    return dataset
