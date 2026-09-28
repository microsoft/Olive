# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

import torch

from olive.common.quant.tensor import QuantTensor
from olive.common.quant.utils import WeightQuantizer
from olive.model.utils.path_utils import normalize_path_suffix
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec

logger = logging.getLogger(__name__)


def _kquant_like_group_qparams(weight: torch.Tensor, bits: int, group_size: int, symmetric: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-group qparams using a lightweight K-quant-style refinement.

    This is intentionally a focused subset of the ONNX K-quant search: it operates on the
    actual weight tensor, refines scale/zero-point per group, and keeps the result compatible
    with Olive's existing ``QuantTensor`` encoding.
    """
    if weight.dim() != 2:
        raise ValueError(f"RnntGptq expects a 2D weight tensor, got {tuple(weight.shape)}")

    if weight.shape[-1] % group_size != 0:
        raise ValueError(
            f"Weight last dimension {weight.shape[-1]} must be divisible by group_size={group_size} for RNNT K-quant refinement."
        )

    q = 1 << bits
    maxq = q - 1
    minq = 0
    if symmetric:
        midq = (maxq + minq + 1) // 2

    reshaped = weight.reshape(-1, group_size).float()
    q_weight = reshaped.clone()
    scales = torch.ones((reshaped.shape[0],), device=weight.device, dtype=torch.float32)
    zero_points = torch.full((reshaped.shape[0],), midq if symmetric else 0, device=weight.device, dtype=torch.int32)

    if symmetric:
        for row_idx in range(reshaped.shape[0]):
            data = reshaped[row_idx]
            rmin = data.min()
            rmax = data.max()
            best_scale = 1.0
            best_mse = torch.mean((data - data.round()) ** 2)
            if rmin == rmax:
                scales[row_idx] = 1.0
                zero_points[row_idx] = midq
                continue

            candidates = []
            for factor in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
                scale = factor * (rmax - rmin) / maxq
                if scale <= 0:
                    continue
                qv = torch.clamp(torch.round((data / scale) + midq), minq, maxq)
                deq = (qv - midq) * scale
                err = torch.mean((data - deq) ** 2)
                candidates.append((err.item(), scale, qv))

            if candidates:
                best_err, best_scale, _ = min(candidates, key=lambda x: x[0])
                scales[row_idx] = float(best_scale)
                zero_points[row_idx] = midq
                q_weight[row_idx] = torch.clamp(torch.round(data / best_scale + midq), minq, maxq)
                best_mse = best_err

            if best_mse > 0:
                pass
    else:
        for row_idx in range(reshaped.shape[0]):
            data = reshaped[row_idx]
            rmin = data.min()
            rmax = data.max()
            if rmin == rmax:
                scales[row_idx] = 1.0
                zero_points[row_idx] = 0
                continue

            best_err = float("inf")
            best_scale = 1.0
            best_zp = 0
            for z in range(0, maxq + 1):
                scale = (rmax - rmin) / maxq
                if scale <= 0:
                    continue
                qv = torch.clamp(torch.round(data / scale + z), minq, maxq)
                deq = (qv - z) * scale
                err = torch.mean((data - deq) ** 2)
                if err.item() < best_err:
                    best_err = err.item()
                    best_scale = float(scale)
                    best_zp = int(z)

            scales[row_idx] = best_scale
            zero_points[row_idx] = best_zp
            q_weight[row_idx] = torch.clamp(torch.round(data / best_scale + best_zp), minq, maxq)

    return scales.reshape(-1, 1), zero_points.reshape(-1, 1)


class RnntGptq(Pass):
    """Apply GPTQ-style weight quantization to RNNT linear layers without requiring HF transformer metadata."""

    @classmethod
    def _default_config(cls, accelerator_spec: "AcceleratorSpec") -> dict[str, PassConfigParam]:
        return {
            "bits": PassConfigParam(type_=int, default_value=4, description="Quantization bits for RNNT layer weights."),
            "group_size": PassConfigParam(type_=int, default_value=64, description="Quantization group size."),
            "sym": PassConfigParam(type_=bool, default_value=False, description="Whether to use symmetric quantization."),
        }

    @classmethod
    def validate_config(cls, config: type[BasePassConfig], accelerator_spec: "AcceleratorSpec") -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        if config.bits not in (2, 4, 8):
            logger.error("RnntGptq bits must be one of {2, 4, 8}.")
            return False

        if config.group_size <= 0:
            logger.error("RnntGptq group_size must be greater than 0.")
            return False

        return True

    def _run_for_config(self, model: Any, config: type[BasePassConfig], output_model_path: str) -> Any:
        base_model = model.load_model(cache_model=False) if hasattr(model, "load_model") else model
        if not isinstance(base_model, torch.nn.Module):
            raise TypeError("RnntGptq expects a torch.nn.Module or PyTorchModelHandler instance.")

        mp_info = None
        if hasattr(model, "model_attributes"):
            mp_info = model.model_attributes.get("mixed_precision_info") if model.model_attributes else None

        default_bits = int(getattr(config, "bits", 4))
        default_group_size = int(getattr(config, "group_size", 64))
        default_sym = bool(getattr(config, "sym", False))
        qparam_strategy = getattr(config, "qparam_strategy", "rtn")

        if mp_info is not None:
            default_cfg = mp_info.get("default", {})
            default_bits = int(default_cfg.get("bits", default_bits))
            default_group_size = int(default_cfg.get("group_size", default_group_size))
            qparam_strategy = default_cfg.get("qparam_strategy", qparam_strategy)

        # Some NeMo RNNT model internals include object-proxy state that cannot be deep-copied.
        # Quantization needs to mutate the loaded model in place for this architecture.
        quantized_model = base_model

        for module in quantized_model.modules():
            if isinstance(module, torch.nn.Linear):
                in_features = module.weight.shape[-1]
                effective_group_size = min(default_group_size, in_features)
                if effective_group_size <= 0:
                    effective_group_size = in_features

                quantizer = WeightQuantizer(
                    bits=default_bits,
                    symmetric=default_sym,
                    group_size=effective_group_size,
                    signed=False,
                )
                if str(qparam_strategy).lower() == "kquant":
                    scales, zero_points = _kquant_like_group_qparams(
                        module.weight.detach(),
                        bits=default_bits,
                        group_size=effective_group_size,
                        symmetric=default_sym,
                    )
                    qweight = QuantTensor.from_float(
                        module.weight.detach(),
                        bits=default_bits,
                        symmetric=default_sym,
                        group_size=effective_group_size,
                        scales=scales.to(module.weight.device),
                        zero_points=zero_points.to(module.weight.device),
                    )
                else:
                    qweight = QuantTensor.from_float(
                        module.weight.detach(),
                        bits=default_bits,
                        symmetric=default_sym,
                        group_size=effective_group_size,
                    )
                module.weight = torch.nn.Parameter(qweight, requires_grad=False)

        resolved_output_path = normalize_path_suffix(output_model_path, "model.pt") if output_model_path else None

        if hasattr(model, "load_model"):
            output_model = model
            output_model.model = quantized_model
            if resolved_output_path:
                torch.save(quantized_model.state_dict(), resolved_output_path)
            return output_model

        class _QuantizedRnntResult:
            def __init__(self, model):
                self.model = model

        output_model = _QuantizedRnntResult(quantized_model)

        if resolved_output_path:
            torch.save(quantized_model.state_dict(), resolved_output_path)
        return output_model
