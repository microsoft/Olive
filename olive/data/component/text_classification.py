# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import List

from olive.common.config_utils import validate_config
from olive.data.component.utils import TokenizerConfig


class TextClassificationParams(TokenizerConfig):
    """Parameters for text classification task"""

    input_cols: List[str]  # input columns
    label_cols: List[str]  # label columns, we take in a list but only use the first one for now
    max_samples: int = None  # max number of samples to use, None for all
    max_length: int = None  # max length of input sequence, used for padding and truncation
    is_split_into_words: bool = False  # whether the input is already split into words
    align_labels: bool = False  # whether to align labels with tokens
    model_config_path: str = None  # path to model config file, used to align labels. If None, use model_name instead


def text_classification_pre_process(dataset, model_name, **kwargs):
    """Pre-process data for text classification task."""
    from transformers import AutoConfig

    args = validate_config(kwargs, TextClassificationParams, warn_unused_keys=True)

    # do this before tokenization to avoid unnecessary tokenization
    if args.max_samples is not None:
        dataset = dataset.select(range(args.max_samples))

    tokenizer, tokenization_kwargs = args.get_tokenizer(model_name, args.max_length)
    tokenization_kwargs["is_split_into_words"] = args.is_split_into_words

    # TODO: add the complete data operation mapping like:
    # align_labels -> align_labels_with_mapping
    # Also to support customized operation arguments from users
    if args.align_labels:
        model_hf_config = AutoConfig.from_pretrained(args.model_config_path or model_name)
        if model_hf_config and model_hf_config.label2id:
            dataset = dataset.align_labels_with_mapping(model_hf_config.label2id, args.label_cols[0])

    def tokenize_and_add_labels(examples):
        tokenized_examples = tokenizer(examples, **tokenization_kwargs)
        # TODO: support multiple label columns if needed
        tokenized_examples["label"] = examples[args.label_cols[0]]
        return tokenized_examples
