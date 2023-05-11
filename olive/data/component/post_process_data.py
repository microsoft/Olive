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
