# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from olive.evaluator.metric import Metric
from olive.evaluator.olive_evaluator import OliveEvaluator, PyTorchEvaluator
from olive.model import PyTorchModelHandler


class _Transcriber(torch.nn.Module):
    """Exercise the native handler without loading a speech model."""

    def forward(self, audio=None, input_values=None, **kwargs):
        audio = audio if audio is not None else input_values
        if isinstance(audio, list):
            batch_size = len(audio)
        else:
            batch_size = audio.shape[0] if audio.ndim > 1 else 1
        return ["hello"] * batch_size


def _collate(batch, input_kind):
    arrays, labels = zip(*batch)
    if input_kind == "ragged_arrays":
        inputs = list(arrays)
    elif input_kind == "ragged_dicts":
        inputs = [{"audio": array, "file_name": str(i)} for i, array in enumerate(arrays)]
    else:
        audio = np.stack(arrays)
        if input_kind == "numpy":
            inputs = audio
        elif input_kind in ("tensor", "tensor_grad"):
            inputs = torch.from_numpy(audio).requires_grad_(input_kind == "tensor_grad")
        else:
            inputs = {"file_name": [str(i) for i in range(len(arrays))], "audio": torch.from_numpy(audio)}
    return inputs, list(labels)


@pytest.mark.parametrize("input_kind", ["numpy", "tensor", "tensor_grad", "dict", "ragged_arrays", "ragged_dicts"])
@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_pytorch_speech_duration_counts_every_audio_sample(input_kind, sample_rate):
    from functools import partial

    lengths = [4000, 8000, 12000] if input_kind.startswith("ragged") else [8000] * 3
    dataset = [(np.zeros(length, dtype=np.float32), "hello") for length in lengths]
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=partial(_collate, input_kind=input_kind))
    metric = Metric.model_validate(
        {
            "name": "speech",
            "type": "accuracy",
            "sub_types": [{"name": "rtfx"}],
            "data_config": {
                "name": "speech_data",
                "pre_process_data_config": {"type": "skip_pre_process", "params": {"sample_rate": sample_rate}},
            },
        }
    )
    model = PyTorchModelHandler(model_loader=lambda _: _Transcriber())

    with patch("olive.evaluator.olive_evaluator.time.perf_counter", side_effect=[0.0, 1.0, 2.0, 3.0]):
        outputs, targets = PyTorchEvaluator()._inference_text(model, metric, dataloader)

    assert outputs.preds == targets == ["hello"] * 3
    expected_duration = sum(lengths) / sample_rate
    assert outputs.logits["total_audio_duration"] == pytest.approx(expected_duration)
    assert outputs.logits["total_inference_time"] == 2.0
    result = OliveEvaluator.compute_accuracy(metric, outputs, targets)
    assert result["rtfx"].value == round(expected_duration / 2.0, 2)


@pytest.mark.parametrize("input_kind", ["numpy", "tensor", "dict"])
def test_pytorch_speech_duration_preserves_unbatched_mono(input_kind):
    audio = np.zeros(8000, dtype=np.float32)
    if input_kind == "tensor":
        inputs = torch.from_numpy(audio)
    elif input_kind == "dict":
        inputs = {"audio": audio, "file_name": "mono.wav"}
    else:
        inputs = audio
    metric = Metric.model_validate({"name": "speech", "type": "accuracy", "sub_types": [{"name": "rtfx"}]})
    model = PyTorchModelHandler(model_loader=lambda _: _Transcriber())
    with patch("olive.evaluator.olive_evaluator.time.perf_counter", side_effect=[0.0, 1.0]):
        outputs, targets = PyTorchEvaluator()._inference_text(model, metric, [(inputs, "hello")])
    assert outputs.preds == targets == ["hello"]
    assert outputs.logits["total_audio_duration"] == 0.5


@pytest.mark.parametrize("audio_key", ["audio", "input_values"])
@pytest.mark.parametrize("numpy_inputs", [False, True])
@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_pytorch_speech_duration_excludes_masked_padding(audio_key, numpy_inputs, sample_rate):
    def collate(batch):
        arrays, labels = zip(*batch)
        max_length = max(len(array) for array in arrays)
        audio = np.zeros((len(arrays), max_length), dtype=np.float32)
        mask = np.zeros_like(audio, dtype=np.int64)
        for index, array in enumerate(arrays):
            audio[index, : len(array)] = array
            mask[index, : len(array)] = 1
        if not numpy_inputs:
            audio, mask = torch.from_numpy(audio), torch.from_numpy(mask)
        # Metadata and masks may precede the waveform in a custom model's inputs.
        return {"attention_mask": mask, audio_key: audio}, list(labels)

    lengths = [4000, 8000, 12000]
    dataset = [(np.zeros(length, dtype=np.float32), "hello") for length in lengths]
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate)
    metric = Metric.model_validate(
        {
            "name": "speech",
            "type": "accuracy",
            "sub_types": [{"name": "rtfx"}],
            "data_config": {
                "name": "speech_data",
                "pre_process_data_config": {"type": "skip_pre_process", "params": {"sample_rate": sample_rate}},
            },
        }
    )
    model = PyTorchModelHandler(model_loader=lambda _: _Transcriber())
    with patch("olive.evaluator.olive_evaluator.time.perf_counter", side_effect=[0.0, 1.0, 2.0, 3.0]):
        outputs, targets = PyTorchEvaluator()._inference_text(model, metric, dataloader)
    assert outputs.preds == targets == ["hello"] * 3
    expected_duration = sum(lengths) / sample_rate
    assert outputs.logits["total_audio_duration"] == pytest.approx(expected_duration)
    assert OliveEvaluator.compute_accuracy(metric, outputs, targets)["rtfx"].value == round(expected_duration / 2, 2)
