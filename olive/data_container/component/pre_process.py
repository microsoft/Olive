# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from olive.data_container.constants import DataComponentType, DefaultDataComponent
from olive.data_container.registry import Registry


@Registry.register(DataComponentType.POST_PROCESS, DefaultDataComponent.POST_PROCESS.value)
def pre_process(data):
    """Pre-process data.

    Args:
        data (object): Data to be pre-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Pre-processed data.
    """
    return data


@Registry.register(DataComponentType.POST_PROCESS)
def huggingface_pre_process(dataset, model_name, input_cols, label_cols, batched=True, **kwargs):
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
            *examples[input_cols],
            padding=True,
            truncation=True,
            **kwargs,
        )
        # TODO: support multiple label columns if needed
        tokenized_inputs["label"] = examples[label_cols[0]]
        return tokenized_inputs

    tokenized_datasets = dataset.map(
        _tokenizer_and_align_labels,
        batched=batched,
        remove_columns=dataset.column_names,
    )
    return tokenized_datasets
