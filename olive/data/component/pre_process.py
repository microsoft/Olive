# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry


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


@Registry.register_pre_process()
def huggingface_pre_process(_dataset, model_name, input_cols, label_cols, **kwargs):
    """Pre-process data.

    Args:
        data (object): Data to be pre-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Pre-processed data.
    """
    from transformers import AutoTokenizer

    dataset = _dataset

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

    # output type is list
    tokenized_datasets = dataset.map(
        _tokenizer_and_align_labels,
        batched=kwargs.get("batched", True),
        remove_columns=dataset.column_names,
    )
    tokenized_datasets.set_format("torch", output_all_columns=True)
    return BaseDataset(tokenized_datasets, label_cols=label_cols)
