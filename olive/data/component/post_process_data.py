# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
import transformers

from olive.data.registry import Registry


@Registry.register_default_post_process()
def post_process(_output_data, **kwargs):
    return _output_data


@Registry.register_post_process()
def text_classification_post_process(_output_data, **kwargs):
    """Post-process data.

    Args:
        data (object): Model output to be post-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Post-processed data.
    """
    if isinstance(_output_data, transformers.modeling_outputs.SequenceClassifierOutput):
        _, preds = torch.max(_output_data.logits, dim=1)
    else:
        _, preds = torch.max(_output_data, dim=1)

    return preds


@Registry.register_post_process()
def ner_post_process(_output_data, **kwargs):
    """Post-process data for NER task.

    Args:
        data (object): Model output to be post-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Post-processed data.
    """
    if isinstance(_output_data, transformers.modeling_outputs.TokenClassifierOutput):
        logits = _output_data.logits
    else:
        logits = _output_data
    return torch.argmax(logits, dim=-1)


@Registry.register_post_process()
def text_generation_post_process(_output_data, **kwargs):
    """Post-process data for text generation task.

    Args:
        data (object): Model output to be post-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Post-processed data.
    """
    if isinstance(_output_data, transformers.modeling_outputs.ModelOutput):
        preds = _output_data.logits
    elif isinstance(_output_data, dict):
        if "logits" in _output_data:
            preds = _output_data["logits"]
        elif "last_hidden_state" in _output_data:
            preds = _output_data["last_hidden_state"]
        else:
            raise ValueError("`logits` or `last_hidden_state` not found in model output.")
    else:
        preds = _output_data
    return preds
