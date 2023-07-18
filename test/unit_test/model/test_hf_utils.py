# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch

from olive.model.hf_utils import load_huggingface_model_from_model_class, load_huggingface_model_from_task


def test_load_huggingface_model_from_task():
    # The model name and task type is gotten from
    # https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/pipelines#transformers.pipeline
    task = "text-classification"
    model_name = "Intel/bert-base-uncased-mrpc"

    model = load_huggingface_model_from_task(task, model_name)
    assert isinstance(model, torch.nn.Module)


def test_load_huggingface_model_from_model_class():
    model_class = "AutoModelForSequenceClassification"
    model_name = "Intel/bert-base-uncased-mrpc"
    model = load_huggingface_model_from_model_class(model_class, model_name)
    assert isinstance(model, torch.nn.Module)
