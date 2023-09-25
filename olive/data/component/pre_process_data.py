# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from copy import deepcopy

from olive.data.component.dataset import BaseDataset
from olive.data.component.text_generation import (
    TextGenDatasetType,
    text_gen_corpus_pre_process,
    text_gen_pair_pre_process,
)
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


def _huggingface_pre_process_helper(dataset, model_name, input_cols, label_cols, map_func, **kwargs):
    """Pre-process data.

    Args:
        dataset (object): Data to be pre-processed.
        model_name (str): Name of the huggingface model.
        input_cols (list): List of input columns.
        label_cols (list): List of label columns.
        map_func (function): Function to be applied to the dataset.
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
        _dataset (object): Data to be pre-processed.
        model_name (str): Name of the huggingface model.
        input_cols (list): List of input columns.
        label_cols (list): List of label columns.
        max_samples (int, optional): Max number of samples to use. Defaults to None.
        **kwargs: Additional arguments.

    Returns:
        object: Pre-processed data.
    """
    from transformers import AutoConfig, AutoTokenizer

    def _tokenizer_and_align_labels(examples):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenized_inputs = tokenizer(
            *[examples[input_col] for input_col in input_cols],
            padding=kwargs.get("padding", True),
            truncation=kwargs.get("truncation", True),
            is_split_into_words=kwargs.get("is_split_into_words", False),
            add_special_tokens=kwargs.get("add_special_tokens", True),
        )
        # TODO(trajep): support multiple label columns if needed
        tokenized_inputs["label"] = examples[label_cols[0]]
        # huggingface dataset api limit to return dict and arrow table
        return tokenized_inputs

    model_config_path = kwargs.pop("model_config_path", None)
    # TODO(trajep): add the complete data operation mapping like:
    # align_labels -> align_labels_with_mapping
    # Also to support customized operation arguments from users
    if kwargs.pop("align_labels", False):
        model_hf_config = AutoConfig.from_pretrained(model_config_path or model_name)
        if model_hf_config and model_hf_config.label2id:
            _dataset = _dataset.align_labels_with_mapping(model_hf_config.label2id, label_cols[0])

    tokenized_datasets = _huggingface_pre_process_helper(
        _dataset, model_name, input_cols, label_cols, _tokenizer_and_align_labels, **kwargs
    )
    # label_cols is ["label"] since we added label_cols[0] as "label" to tokenized_inputs
    return BaseDataset(tokenized_datasets, label_cols=["label"], max_samples=max_samples)


@Registry.register_pre_process()
def ner_huggingface_preprocess(_dataset, model_name, input_cols, label_cols, max_samples=None, **kwargs):
    """Pre-process data for ner task."""
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


@Registry.register_pre_process()
def text_generation_huggingface_pre_process(
    _dataset, model_name: str, dataset_type: TextGenDatasetType, source_max_len: int, max_samples=None, **kwargs
):
    """Pre-process data for text generation task.

    Args:
        _dataset (object): Data to be pre-processed.
        model_name (str): Name of the huggingface model.
        dataset_type (TextGenDatasetType): Type of the dataset - 'corpus' or 'pair'.
        source_max_len (int): Max length of source sequence. For corpus, this is the max length of each sequence.
            For pair, this is the max length of the input sequence.
        max_samples (int, optional): Max number of samples to use. Defaults to None.
        **kwargs: Additional arguments.
            The common arguments are the fields in olive.data.component.text_generation.TextGenParams.
            'corpus' arguments are the fields in olive.data.component.text_generation.TextGenCorpusParams.
            'pair' arguments are the fields in olive.data.component.text_generation.TextGenPairParams.
            Note: the TextGenCorpusParams and TextGenPairParams subclasses already include the common arguments.
    """
    from transformers import AutoTokenizer

    all_kwargs = deepcopy(kwargs)
    all_kwargs.update({"max_samples": max_samples, "source_max_len": source_max_len})

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if dataset_type == TextGenDatasetType.CORPUS:
        return text_gen_corpus_pre_process(_dataset, tokenizer, all_kwargs)
    else:
        return text_gen_pair_pre_process(_dataset, tokenizer, all_kwargs)
