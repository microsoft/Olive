<<<<<<< HEAD:test/unit_test/hf_utils/test_hf_utils.py
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch

from olive.hf_utils import load_huggingface_model_from_model_class, load_huggingface_model_from_task


def test_load_huggingface_model_from_task():
    # The model name and task type is gotten from
    # https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/pipelines#transformers.pipeline
    task = "automatic-speech-recognition"
    model_name = "facebook/wav2vec2-base-960h"

    model = load_huggingface_model_from_task(task, model_name)
    assert isinstance(model, torch.nn.Module)


def test_load_huggingface_model_from_model_class():
    model_class = "Wav2Vec2ForCTC"
    model_name = "facebook/wav2vec2-base-960h"

    model = load_huggingface_model_from_model_class(model_class, model_name)
    assert isinstance(model, torch.nn.Module)
=======
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch

from olive.model.hf_utils import load_huggingface_model_from_model_class, load_huggingface_model_from_task


def test_load_huggingface_model_from_task():
    # The model name and task type is gotten from
    # https://huggingface.co/docs/transformers/v4.28.1/en/main_classes/pipelines#transformers.pipeline
    task = "automatic-speech-recognition"
    model_name = "facebook/wav2vec2-base-960h"

    model = load_huggingface_model_from_task(task, model_name)
    assert isinstance(model, torch.nn.Module)


def test_load_huggingface_model_from_model_class():
    model_class = "Wav2Vec2ForCTC"
    model_name = "facebook/wav2vec2-base-960h"
    model = load_huggingface_model_from_model_class(model_class, model_name)
    assert isinstance(model, torch.nn.Module)
>>>>>>> 5ec0a52c973f1addd2a0491e2fdf38d5e2b56224:test/unit_test/model/test_hf_utils.py
