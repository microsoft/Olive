# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from olive.common.quant.utils import WeightQuantizer
from olive.passes.pytorch.multimodal_quantization import (
    ActivationRangeObserver,
    MultimodalCalibrationMasks,
    ProtectedInputChannelLinear,
    hadamard_mean_outlier_ratio,
    modality_balanced_reconstruction_loss,
    split_multimodal_calibration_batch,
)
from olive.passes.pytorch.train_utils import get_calibration_dataset


def test_get_calibration_dataset_preserves_targets_only_when_requested():
    inputs = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2),
    }
    labels = torch.tensor([[-100, 2]])
    data_config = SimpleNamespace(
        to_data_container=lambda: SimpleNamespace(create_dataloader=lambda: [(inputs, labels)])
    )

    with patch("olive.passes.pytorch.train_utils.validate_config", return_value=data_config):
        without_labels = get_calibration_dataset(object(), data_config)
        with_labels = get_calibration_dataset(object(), data_config, include_labels=True)

    assert "labels" not in without_labels[0]
    assert torch.equal(with_labels[0]["labels"], labels)


def test_split_multimodal_calibration_batch_derives_answer_mask_and_strips_metadata():
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4),
        "labels": torch.tensor([[-100, -100, 3, 4]]),
        "vision_mask": torch.tensor([[False, True, False, False]]),
        "pixel_values": torch.randn(1, 3, 2, 2),
    }

    model_inputs, masks = split_multimodal_calibration_batch(batch)

    assert "vision_mask" not in model_inputs
    assert "pixel_values" in model_inputs
    assert torch.equal(masks.vision, batch["vision_mask"])
    assert torch.equal(masks.answer, batch["labels"].ne(-100))


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"vision_mask": torch.ones(1, 3, dtype=torch.bool)}, "match input_ids"),
        ({"answer_mask": torch.tensor([[False, True, False, False]])}, "must not overlap"),
        (
            {
                "vision_mask": torch.tensor([[False, False, False, True]]),
                "attention_mask": torch.tensor([[1, 1, 1, 0]]),
                "labels": torch.full((1, 4), -100),
            },
            "must not select padded",
        ),
    ],
)
def test_split_multimodal_calibration_batch_rejects_invalid_masks(update, match):
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4),
        "labels": torch.tensor([[-100, -100, 3, 4]]),
        "vision_mask": torch.tensor([[False, True, False, False]]),
    }
    batch.update(update)

    with pytest.raises(ValueError, match=match):
        split_multimodal_calibration_batch(batch)


def test_modality_balanced_reconstruction_loss_weights_vision_error():
    reference = torch.zeros(1, 2, 2)
    candidate = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    masks = MultimodalCalibrationMasks(
        vision=torch.tensor([[True, False]]),
        answer=torch.tensor([[False, True]]),
    )

    loss = modality_balanced_reconstruction_loss(reference, candidate, masks, vision_weight=0.25)

    assert loss.item() == pytest.approx((2 * 2.0 + 0.25 * 2 * 1.0) / 4)


def test_activation_range_observer_reports_reproducible_qparams():
    observer = ActivationRangeObserver(bits=8, symmetric=False)
    observer.update(torch.tensor([-2.0, 0.0, 6.0]))
    observer.update(torch.tensor([1.0, 3.0]))

    qparams = observer.qparams()

    assert qparams["minimum"] == -2.0
    assert qparams["maximum"] == 6.0
    assert qparams["scale"] == pytest.approx(8 / 255)
    assert qparams["numel"] == 5
    assert qparams["num_batches"] == 2


def test_protected_input_channel_linear_is_algebraically_exact():
    torch.manual_seed(0)
    linear = torch.nn.Linear(8, 5)
    split = ProtectedInputChannelLinear(linear)
    inputs = torch.randn(2, 3, 8)

    assert torch.allclose(split(inputs), linear(inputs), atol=1e-6, rtol=1e-6)


def test_protecting_hadamard_mean_channel_reduces_constructed_quantization_error():
    weight = torch.tensor(
        [
            [20.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            [-18.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0],
            [16.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            [-14.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0, -6.0],
        ]
    )
    quantizer = WeightQuantizer(bits=4, symmetric=True, group_size=0)
    inputs = torch.randn(16, 8)
    reference = torch.matmul(inputs, weight.T)

    fully_quantized = torch.matmul(inputs, quantizer.fake_quantize(weight).T)
    protected = weight[:, :1]
    remaining = quantizer.fake_quantize(weight[:, 1:])
    split_quantized = torch.matmul(inputs[:, :1], protected.T)
    split_quantized += torch.matmul(inputs[:, 1:], remaining.T)

    full_error = (reference - fully_quantized).abs().mean()
    split_error = (reference - split_quantized).abs().mean()
    assert split_error < full_error
    assert hadamard_mean_outlier_ratio(weight) > 1
