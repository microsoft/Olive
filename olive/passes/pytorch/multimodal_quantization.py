# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared calibration and analysis primitives for multimodal quantization."""

# pylint: disable=not-callable

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

VISION_MASK_KEY = "vision_mask"
ANSWER_MASK_KEY = "answer_mask"
_METADATA_KEYS = {VISION_MASK_KEY, ANSWER_MASK_KEY}


@dataclass(frozen=True)
class MultimodalCalibrationMasks:
    """Decoder-coordinate token masks carried alongside model inputs."""

    vision: torch.Tensor
    answer: torch.Tensor

    def to(self, device: str | torch.device) -> MultimodalCalibrationMasks:
        return MultimodalCalibrationMasks(self.vision.to(device), self.answer.to(device))


def split_multimodal_calibration_batch(
    batch: dict[str, torch.Tensor],
    *,
    vision_mask_key: str = VISION_MASK_KEY,
    answer_mask_key: str = ANSWER_MASK_KEY,
    ignore_index: int = -100,
) -> tuple[dict[str, torch.Tensor], MultimodalCalibrationMasks]:
    """Separate model inputs from decoder-coordinate quantization metadata."""
    if "input_ids" not in batch:
        raise ValueError("Multimodal calibration batches must contain input_ids.")
    if vision_mask_key not in batch:
        raise ValueError(f"Multimodal calibration batches must contain {vision_mask_key!r}.")

    vision_mask = batch[vision_mask_key].bool()
    if answer_mask_key in batch:
        answer_mask = batch[answer_mask_key].bool()
    elif "labels" in batch:
        # MBQ's reference implementation uses the unshifted labels mask. The causal
        # loss performs its own next-token shift independently.
        answer_mask = batch["labels"].ne(ignore_index)
    else:
        raise ValueError(f"Multimodal calibration batches must contain {answer_mask_key!r} or labels to derive it.")

    input_ids = batch["input_ids"]
    expected_shape = input_ids.shape[:2]
    if vision_mask.shape != expected_shape or answer_mask.shape != expected_shape:
        raise ValueError(
            "vision and answer masks must use decoder token coordinates and match input_ids shape "
            f"{tuple(expected_shape)}; got {tuple(vision_mask.shape)} and {tuple(answer_mask.shape)}."
        )
    if torch.any(vision_mask & answer_mask):
        raise ValueError("vision and answer masks must not overlap.")

    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        if attention_mask.shape != expected_shape:
            raise ValueError("attention_mask must match input_ids shape for multimodal calibration.")
        padding_mask = attention_mask.eq(0)
        if torch.any((vision_mask | answer_mask) & padding_mask):
            raise ValueError("vision and answer masks must not select padded tokens.")

    metadata_keys = _METADATA_KEYS | {vision_mask_key, answer_mask_key}
    model_inputs = {key: value for key, value in batch.items() if key not in metadata_keys}
    return model_inputs, MultimodalCalibrationMasks(vision=vision_mask, answer=answer_mask)


def validate_masks_for_activations(
    masks: MultimodalCalibrationMasks,
    activations: torch.Tensor,
) -> None:
    """Ensure masks address the batch and sequence dimensions of decoder activations."""
    if activations.ndim < 3:
        raise ValueError(f"Expected decoder activations with at least 3 dimensions, got {activations.ndim}.")
    activation_shape = activations.shape[:2]
    if masks.vision.shape != activation_shape or masks.answer.shape != activation_shape:
        raise ValueError(
            "Calibration masks do not match the captured decoder activation coordinates: "
            f"masks={tuple(masks.vision.shape)}, activations={tuple(activation_shape)}. "
            "Provide masks after multimodal token expansion."
        )


def modality_balanced_reconstruction_loss(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    masks: MultimodalCalibrationMasks,
    vision_weight: float,
) -> torch.Tensor:
    """Compute MBQ's answer-plus-weighted-vision reconstruction MAE."""
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate outputs must have identical shapes.")
    validate_masks_for_activations(masks, reference)
    if vision_weight < 0:
        raise ValueError("vision_weight must be non-negative.")

    error = (reference.float() - candidate.float()).abs()
    answer = masks.answer.unsqueeze(-1).expand_as(error)
    vision = masks.vision.unsqueeze(-1).expand_as(error)
    denominator = answer.sum() + vision.sum()
    if denominator == 0:
        raise ValueError("At least one vision or answer token is required for MBQ reconstruction.")
    return (error[answer].sum() + vision_weight * error[vision].sum()) / denominator


class ActivationRangeObserver:
    """Streaming per-tensor activation range observer for multimodal diagnostics."""

    def __init__(self, bits: int = 8, symmetric: bool = True):
        if bits < 2 or bits > 16:
            raise ValueError("bits must be between 2 and 16.")
        self.bits = bits
        self.symmetric = symmetric
        self.minimum = math.inf
        self.maximum = -math.inf
        self.numel = 0
        self.num_batches = 0

    @torch.no_grad()
    def update(self, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            return
        finite = tensor.detach().float()
        finite = finite[torch.isfinite(finite)]
        if finite.numel() == 0:
            raise ValueError("Activation range calibration received no finite values.")
        self.minimum = min(self.minimum, finite.min().item())
        self.maximum = max(self.maximum, finite.max().item())
        self.numel += finite.numel()
        self.num_batches += 1

    def qparams(self) -> dict[str, int | float | bool]:
        if self.numel == 0:
            raise ValueError("Activation range observer has no samples.")

        if self.symmetric:
            quant_max = (1 << (self.bits - 1)) - 1
            bound = max(abs(self.minimum), abs(self.maximum))
            scale = bound / quant_max if bound else 1.0
            zero_point = 0
            quant_min = -(1 << (self.bits - 1))
        else:
            quant_min = 0
            quant_max = (1 << self.bits) - 1
            width = self.maximum - self.minimum
            scale = width / (quant_max - quant_min) if width else 1.0
            zero_point = round(quant_min - self.minimum / scale)
            zero_point = min(max(zero_point, quant_min), quant_max)

        return {
            "bits": self.bits,
            "symmetric": self.symmetric,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "scale": scale,
            "zero_point": zero_point,
            "quant_min": quant_min,
            "quant_max": quant_max,
            "numel": self.numel,
            "num_batches": self.num_batches,
        }


def hadamard_mean_outlier_ratio(weight: torch.Tensor) -> float:
    """Return the MQuant Eq. 9 channel-mean outlier ratio for an online Hadamard input."""
    if weight.ndim != 2:
        raise ValueError("Hadamard outlier analysis expects a 2D linear weight.")
    original_max = weight.detach().float().abs().max()
    if original_max == 0:
        return 0.0
    rotated_first_channel = math.sqrt(weight.shape[1]) * weight.detach().float().mean(dim=1).abs().max()
    return (rotated_first_channel / original_max).item()


class ProtectedInputChannelLinear(nn.Module):
    """Reference GEMV-plus-GEMM split used to study a protected activation channel.

    This module is intentionally not an export contract. Mobius/ORT support is
    required before an MQuant RMS deployment pass can use this structure.
    """

    def __init__(self, linear: nn.Linear, channel: int = 0):
        super().__init__()
        if not 0 <= channel < linear.in_features:
            raise ValueError(f"channel must be in [0, {linear.in_features}), got {channel}.")
        self.channel = channel
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.protected_weight = nn.Parameter(linear.weight[:, channel : channel + 1].detach().clone())
        remaining = torch.cat((linear.weight[:, :channel], linear.weight[:, channel + 1 :]), dim=1)
        self.remaining_weight = nn.Parameter(remaining.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        protected = inputs[..., self.channel : self.channel + 1]
        remaining = torch.cat((inputs[..., : self.channel], inputs[..., self.channel + 1 :]), dim=-1)
        return F.linear(protected, self.protected_weight) + F.linear(remaining, self.remaining_weight, self.bias)
