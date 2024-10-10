# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import transformers

from olive.data.registry import Registry


@Registry.register_post_process()
@Registry.register_default_post_process()
@Registry.register_post_process("skip_post_process")
def post_process(output_data, **kwargs):
    """Post-process data.

    Args:
        output_data (object): Model output to be post-processed.
        **kwargs: Additional named arguments.

    Returns:
        object: Post-processed data.

    """
    return output_data


@Registry.register_post_process()
def text_classification_post_process(output_data, **kwargs):
    """Post-process data.

    Args:
        output_data (object): Model output to be post-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Post-processed data.

    """
    import torch

    if isinstance(output_data, transformers.modeling_outputs.SequenceClassifierOutput):
        _, preds = torch.max(output_data.logits, dim=1)
    else:
        _, preds = torch.max(output_data, dim=1)

    return preds


@Registry.register_post_process()
def ner_post_process(output_data, **kwargs):
    """Post-process data for NER task.

    Args:
        output_data (object): Model output to be post-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Post-processed data.

    """
    import torch

    if isinstance(output_data, transformers.modeling_outputs.TokenClassifierOutput):
        logits = output_data.logits
    else:
        logits = output_data
    return torch.argmax(logits, dim=-1)


@Registry.register_post_process()
def text_generation_post_process(output_data, **kwargs):
    """Post-process data for text generation task.

    Args:
        output_data (object): Model output to be post-processed.
        **kwargs: Additional arguments.

    Returns:
        object: Post-processed data.

    """
    if isinstance(output_data, transformers.modeling_outputs.ModelOutput):
        preds = output_data.logits
    elif isinstance(output_data, dict):
        if "logits" in output_data:
            preds = output_data["logits"]
        elif "last_hidden_state" in output_data:
            preds = output_data["last_hidden_state"]
        else:
            raise ValueError("`logits` or `last_hidden_state` not found in model output.")
    else:
        preds = output_data
    return preds
