# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------


from copy import deepcopy

from olive.data.component.hf.text_classification import ner_pre_process, text_classification_pre_process
from olive.data.component.hf.text_generation import (
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


@Registry.register_pre_process()
def huggingface_pre_process(_dataset, model_name, input_cols, label_cols, max_samples=None, **kwargs):
    """Pre-process data.

    Args:
        _dataset (object): Data to be pre-processed.
        model_name (str): Name of the huggingface model.
        input_cols (List[str]): Input columns.
        label_cols (List[str]): Label columns.
        **kwargs: Additional arguments.
            The arguments are the fields in olive.data.component.hf.text_classification.TextClassificationParams.

    Returns:
        object: Pre-processed data.
    """

    all_kwargs = deepcopy(kwargs)
    all_kwargs.update({"max_samples": max_samples, "input_cols": input_cols, "label_cols": label_cols})

    return text_classification_pre_process(_dataset, model_name, **all_kwargs)


@Registry.register_pre_process()
def ner_huggingface_preprocess(_dataset, model_name, input_cols, label_cols, max_samples=None, **kwargs):
    """
    Pre-process data for ner task.

    Args:
        _dataset (object): Data to be pre-processed.
        model_name (str): Name of the huggingface model.
        input_cols (List[str]): Input columns.
        label_cols (List[str]): Label columns.
        max_samples (int, optional): Max number of samples to use. Defaults to None.
        **kwargs: Additional arguments.
            The arguments are the fields in olive.data.component.hf.text_classification.CommonClassificationParams.

    Returns:
        object: Pre-processed data.
    """
    all_kwargs = deepcopy(kwargs)
    all_kwargs.update({"max_samples": max_samples, "input_cols": input_cols, "label_cols": label_cols})

    return ner_pre_process(_dataset, model_name, **all_kwargs)


@Registry.register_pre_process()
def text_generation_huggingface_pre_process(
    _dataset, model_name: str, dataset_type: TextGenDatasetType, source_max_len: int, max_samples=None, **kwargs
):
    """
    Pre-process data for text generation task.

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

    all_kwargs = deepcopy(kwargs)
    all_kwargs.update({"max_samples": max_samples, "source_max_len": source_max_len})

    if dataset_type == TextGenDatasetType.CORPUS:
        return text_gen_corpus_pre_process(_dataset, model_name, all_kwargs)
    else:
        return text_gen_pair_pre_process(_dataset, model_name, all_kwargs)
