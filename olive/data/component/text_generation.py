# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from enum import Enum
from pathlib import Path
from random import Random
from typing import Callable, Dict, List, Optional, Union

import torch
import transformers

from olive.common.config_utils import ConfigBase, validate_config, validate_object
from olive.common.pydantic_v1 import validator
from olive.common.user_module_loader import UserModuleLoader
from olive.data.component.dataset import BaseDataset
from olive.data.constants import IGNORE_INDEX


class TextGenStrategy(str, Enum):
    """Strategy for tokenizing a dataset."""

    LINE_BY_LINE = "line-by-line"  # each line is a sequence, in order of appearance
    LINE_BY_LINE_RANDOM = "line-by-line-random"  # each line is a sequence, in random order
    JOIN = "join"  # join all lines into a single sequence, split into non-overlapping sequences
    JOIN_RANDOM = "join-random"  # join all lines into a single sequence, split into random sequences
    JOIN_SLIDING_WINDOW = (  # join all lines into a single sequence, split into overlapping sequences
        "join-sliding-window"
    )


class TextGenParams(ConfigBase):
    """Common parameters for text generation tasks.

    Base dataclass for text generation tasks.
    """

    user_script: Union[str, Path] = None  # user script use to define formatting functions
    script_dir: Union[str, Path] = None  # directory with user script dependencies
    max_samples: int = None  # max number of samples to use, None for all
    max_seq_len: int = 1024  # max length of sequence
    # TODO(jambayk): currently only support padding to max length since we preprocess all data at once
    # might have to expose collator for dataloader to support dynamic padding of batches
    # if false, cannot guarantee all sequences are same length. data loader will have to handle this during collation
    pad_to_max_len: bool = True  # pad sequences to max_len, ignored for JOIN corpus strategy
    drop_short_sequences: bool = False  # drop sequences shorter than max_len. Mutually exclusive with pad_to_max_len
    add_special_tokens: bool = False  # add bos and eos tokens to each sequence
    use_attention_mask: bool = True  # add attention mask to each example
    # one of text_formatting_func, text_template, or text_cols must be provided
    # priority: text_formatting_func > text_template > text_cols
    # function that formats the text, must take in an example dict and return a string
    text_formatting_func: Union[str, Callable] = None  # name of formatting function for text
    # a python f-string template for the text with {column_name} as placeholders
    text_template: str = None
    # list of text columns, columns are concatenated together using a space
    text_cols: Union[str, List[str]] = "text"
    # in JOIN strategies, the rows of text_cols are concatenated together
    corpus_strategy: TextGenStrategy = TextGenStrategy.JOIN
    stride: int = None  # required when corpus_strategy is JOIN_SLIDING_WINDOW
    # text to join the rows of input columns when corpus_strategy is JOIN
    # add_special_tokens: "{bos_token} {text_col1} {eos_token} {joiner} {bos_token} {text_col2} {eos_token}..."
    # no add_special_tokens: "{text_col1} {joiner} {text_col2}..."
    # if None, joined with a space
    joiner: str = "\n\n"
    processing_batch_size: int = 1024  # number of examples to process at a time
    random_seed: int = None  # random seed for LINE_BY_LINE_RANDOM and JOIN_RANDOM
    random_retries: int = (
        10  # number of resamples to try before giving up when a sample is too short for RANDOM strategies
    )

    @validator("drop_short_sequences", always=True)
    def _check_padding(cls, v, values):
        if "pad_to_max_len" not in values:
            raise ValueError("Invalid pad_to_max_len")
        if v and values["pad_to_max_len"]:
            raise ValueError("pad_to_max_len and drop_short_sequences cannot both be True")
        return v

    @validator("text_formatting_func")
    def _check_text_formatting_func(cls, v, values, field):
        return validate_object(v, values, field)

    @validator("text_cols", always=True)
    def _check_text_cols(cls, v, values):
        alternatives = ["text_formatting_func", "text_template"]
        for alternate in alternatives:
            # for good validation error, check that all alternates are in values
            if alternate not in values:
                raise ValueError(f"Invalid {alternate}")

        if not (v or any(values[option] for option in alternatives)):
            # check that at least one alternate is specified
            raise ValueError(f"One of text_cols, {', '.join(alternatives)} must be specified")

        if v is not None and isinstance(v, str):
            v = [v]
        return v

    @validator("stride", always=True)
    def _check_stride(cls, v, values):
        if "corpus_strategy" not in values:
            raise ValueError("Invalid corpus_strategy")
        if values["corpus_strategy"] == TextGenStrategy.JOIN_SLIDING_WINDOW and v is None:
            raise ValueError("stride must be specified when corpus_strategy is JOIN_SLIDING_WINDOW")
        return v

    @validator("corpus_strategy", always=True)
    def _check_max_samples(cls, v, values):
        if "max_samples" not in values:
            raise ValueError("Invalid max_samples")
        if "random" in v and values["max_samples"] is None:
            raise ValueError("max_samples must be specified when corpus_strategy is random")
        return v

    @validator("corpus_strategy", always=True)
    def _check_use_attention_mask(cls, v, values):
        if "use_attention_mask" not in values:
            raise ValueError("Invalid use_attention_mask")
        if "pad_to_max_len" not in values:
            raise ValueError("Invalid pad_to_max_len")
        use_attention_mask = values["use_attention_mask"]
        pad_to_max_len = values["pad_to_max_len"]
        if "join" in v:
            # both True and False are valid since attention_mask is all 1s
            return v
        if not use_attention_mask and pad_to_max_len:
            raise ValueError(
                "pad_to_max_len is True but use_attention_mask is False. Attention mask is required for padding!"
            )
        return v

    @validator("random_seed", always=True)
    def _check_random(cls, v, values):
        if "corpus_strategy" not in values:
            raise ValueError("Invalid corpus_strategy")
        if "random" in values["corpus_strategy"] and v is None:
            raise ValueError("random_seed must be specified when corpus_strategy is random")
        return v

    def get_user_module_loader(self):
        """Get user module loader."""
        return UserModuleLoader(self.user_script, self.script_dir)


def text_gen_pre_process(dataset, tokenizer, all_kwargs):
    """Pre-process data for text generation task.

    The input dataset is expected to have one or more text columns.
    Depending on the corpus_strategy, the sequences are either joined together or processed individually.
    """
    from datasets import Dataset as HFDataset

    args = validate_config(all_kwargs, TextGenParams, warn_unused_keys=True)

    if isinstance(args.text_formatting_func, str):
        # load text_formatting_func
        args.text_formatting_func = args.get_user_module_loader().load_object(args.text_formatting_func)
    # get text from dataset
    dataset = dataset.map(
        lambda x: {
            "text": get_text(
                x, args.text_formatting_func, args.text_template, args.text_cols, args.add_special_tokens, tokenizer
            )
        }
    )
    text_list = dataset["text"]
    total_examples = len(text_list)  # total number of examples

    tokenized_inputs = {
        "input_ids": [],
        "labels": [],
    }
    if args.use_attention_mask:
        tokenized_inputs["attention_mask"] = []
    if "join" in args.corpus_strategy:
        joiner_tokens = tokenizer.encode(args.joiner, add_special_tokens=False) if args.joiner else []

        if args.corpus_strategy != TextGenStrategy.JOIN_RANDOM:
            # no randomization, just use contiguous blocks of tokens
            if args.corpus_strategy == TextGenStrategy.JOIN_SLIDING_WINDOW:
                # we use the stride as both the step between sequences and the context size
                step, context = args.stride, args.max_seq_len - args.stride
            else:
                # JOIN corpus_strategy
                # text is split into non-overlapping sequences and there is no context
                step, context = args.max_seq_len, None

            example_idx = 0  # index of the first example in the current batch
            num_samples = 0  # samples processed so far
            overflow = []  # tokens overflowed from the previous batch of examples
            # we will process in batches to make tokenization faster
            # better than joining all text together and tokenizing all at once
            while True:
                if args.max_samples is not None and num_samples >= args.max_samples:
                    # we have reached max_samples
                    break
                if example_idx >= total_examples:
                    # we have reached the end of the text_list
                    break

                examples_to_get = min(args.processing_batch_size, total_examples - example_idx)
                # batch tokenize
                batched_input_ids = tokenizer(
                    text_list[example_idx : example_idx + examples_to_get],  # noqa: E203, RUF100
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]

                # join all the input_ids together with joiner_tokens
                joined_input_ids = overflow
                for input_ids in batched_input_ids:
                    joined_input_ids += input_ids + joiner_tokens

                end_loc = 0  # position of unused token in joined_input_ids
                # '- args.max_seq_len ' is used to make sure we don't get a sequence that is too short
                for begin_loc in range(0, len(joined_input_ids) - args.max_seq_len, step):
                    # end_loc is the beginning of the next sequence
                    end_loc = begin_loc + args.max_seq_len
                    # get the input sequence
                    input_ids = torch.tensor(joined_input_ids[begin_loc:end_loc])
                    append_text_gen_input_ids(
                        tokenized_inputs,
                        input_ids,
                        tokenizer,
                        context=context,
                        use_attention_mask=args.use_attention_mask,
                    )
                    num_samples += 1
                    if args.max_samples is not None and num_samples >= args.max_samples:
                        # we have reached max_samples
                        break
                # update counters
                example_idx += examples_to_get
                overflow = joined_input_ids[end_loc:]
        else:
            # randomization, sample random blocks of tokens
            rng = Random(args.random_seed)
            # cache to store tokenized examples
            cache = {}
            for _ in range(args.max_samples):
                resamples = 0
                # will try to sample sequences random_retries times before giving up
                while resamples < args.random_retries:
                    # sample a beginning example
                    # randint is inclusive, so we need to subtract 1
                    begin_example_idx = rng.randint(0, total_examples - 1)
                    joined_input_ids = []
                    # loop through the examples until we have enough tokens
                    for i in range(begin_example_idx, total_examples):
                        # get the input_ids
                        if i not in cache:
                            cache[i] = tokenizer(
                                text_list[i],
                                add_special_tokens=False,
                                truncation=False,
                            )["input_ids"]
                        joined_input_ids += cache[i] + joiner_tokens
                        # stop if we have enough tokens
                        if len(joined_input_ids) >= args.max_seq_len:
                            break
                    # add to samples if we have enough tokens
                    if len(joined_input_ids) >= args.max_seq_len:
                        # found a good example
                        input_ids = torch.tensor(joined_input_ids[: args.max_seq_len])
                        append_text_gen_input_ids(
                            tokenized_inputs, input_ids, tokenizer, use_attention_mask=args.use_attention_mask
                        )
                        break
                    resamples += 1
    else:
        # each line is a sequence
        if args.corpus_strategy == TextGenStrategy.LINE_BY_LINE:
            # batched tokenization might be faster so lets tokenize all the text at once
            if not args.max_samples:
                batched_input_ids = batch_tokenize_text(text_list, tokenizer, args)
                for native_input_ids in batched_input_ids:
                    input_ids = torch.tensor(native_input_ids)
                    append_text_gen_input_ids(
                        tokenized_inputs, input_ids, tokenizer, use_attention_mask=args.use_attention_mask
                    )
            else:
                example_idx = 0  # index of the first example in the current batch
                num_samples = 0
                while True:
                    if num_samples >= args.max_samples or example_idx >= total_examples:
                        # we have reached max_samples or the end of the text_list
                        break
                    # get as many examples as possible without going over max_samples
                    examples_to_get = min(args.max_samples - num_samples, total_examples - example_idx)
                    # batch tokenize
                    batched_input_ids = batch_tokenize_text(
                        text_list[example_idx : example_idx + examples_to_get], tokenizer, args  # noqa: E203, RUF100
                    )
                    for native_input_ids in batched_input_ids:
                        input_ids = torch.tensor(native_input_ids)
                        append_text_gen_input_ids(
                            tokenized_inputs, input_ids, tokenizer, use_attention_mask=args.use_attention_mask
                        )
                    # update counters
                    num_samples += len(batched_input_ids)
                    example_idx += examples_to_get
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
                            max_length=args.max_seq_len,
                            truncation=True,
                            padding="max_length" if args.pad_to_max_len else False,
                            add_special_tokens=False,
                            return_tensors="pt",
                        )
                        cache[i] = encodings
                    else:
                        encodings = cache[i]
                    if not args.drop_short_sequences or encodings.input_ids.shape[1] >= args.max_seq_len:
                        # found a good sample
                        break
                    resamples += 1
                if not encodings:
                    # could not find a good sample after resampling
                    continue
                append_text_gen_input_ids(
                    tokenized_inputs, encodings.input_ids[0], tokenizer, use_attention_mask=args.use_attention_mask
                )

    # convert to HFDataset
    hf_dataset = HFDataset.from_dict(tokenized_inputs)
    hf_dataset.set_format("torch", output_all_columns=True)

    # return BaseDataset
    return BaseDataset(hf_dataset, ["labels"], max_samples=args.max_samples)


def get_text(
    example: Dict[str, str],
    formatting_func: Optional[Callable] = None,
    template: str = None,
    cols: List[str] = None,
    add_special_tokens: bool = False,
    tokenizer: transformers.PreTrainedTokenizer = None,
):
    """Get text from example using formatting_func, template, or cols."""
    text = None
    if formatting_func:
        text = formatting_func(example)
    elif template:
        text = template.format(**example)
    elif cols:
        text = " ".join([example[col] for col in cols])
    else:
        raise ValueError("None of formatting_func, template, or cols is specified")
    if add_special_tokens:
        # add bos and eos tokens
        text = f"{tokenizer.bos_token} {text} {tokenizer.eos_token}"
    return text


def batch_tokenize_text(text_list, tokenizer, args):
    """Batch tokenize text."""
    batched_encodings = tokenizer(
        text_list,
        max_length=args.max_seq_len,
        truncation=True,
        padding="max_length" if args.pad_to_max_len else False,
        add_special_tokens=False,
    )
    batched_input_ids = batched_encodings.input_ids
    if args.drop_short_sequences:
        batched_input_ids = [input_ids for input_ids in batched_input_ids if len(input_ids) >= args.max_seq_len]
    return batched_input_ids


def append_text_gen_input_ids(
    tokenized_inputs, input_ids, tokenizer, context: int = None, ignore_index=IGNORE_INDEX, use_attention_mask=True
):
    """Convert input_ids to inputs dict and append to tokenized_inputs."""
    inputs = {"input_ids": input_ids}

    # create attention_mask
    if use_attention_mask:
        attention_mask = (
            torch.ones_like(input_ids)
            if tokenizer.pad_token_id is None
            else input_ids.ne(tokenizer.pad_token_id).to(input_ids.dtype)  # is boolean otherwise
        )
        inputs["attention_mask"] = attention_mask

    # create labels
    # target is not shifted by 1 since causal lm models shifts internally when computing loss
    labels = input_ids.clone()
    # set context to ignore_index
    if context is not None:
        labels[:context] = ignore_index
    # set padding to ignore_index
    if use_attention_mask:
        labels[attention_mask != 1] = ignore_index
    inputs["labels"] = labels

    # add to list
    for k, v in inputs.items():
        tokenized_inputs[k].append(v)
