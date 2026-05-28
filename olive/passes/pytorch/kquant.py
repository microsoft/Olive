# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# K-quant weight-only quantization for PyTorch models.
#
# Algorithm originates from llama.cpp's ggml k-quants
# (``make_qkx2_quants`` for the asymmetric variant and ``make_qx_quants`` for
# the symmetric variant):
# https://github.com/ggml-org/llama.cpp/blob/64eda5deb9859e87a020e56bab5d2f9ca956f1de/ggml/src/ggml-quants.c
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import torch

from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam
from olive.passes.pytorch.quant_utils import finalize, get_quantizer_config, prepare_model

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import HfModelHandler
    from olive.passes.pass_config import BasePassConfig


logger = logging.getLogger(__name__)


_ASYM_NSTEP = 20
_ASYM_RDELTA = 0.1
_ASYM_RRMIN = -1.0

_SYM_STEPS = tuple(s for s in range(-9, 10) if s != 0)
_SYM_STEP_DELTA = 0.1


RefineFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]


def _safe_reciprocal(x: torch.Tensor) -> torch.Tensor:
    safe = torch.where(x != 0, x, torch.ones_like(x))
    return torch.where(x != 0, 1.0 / safe, torch.ones_like(x))


def _refine_asymmetric(
    data: torch.Tensor,
    weights: torch.Tensor,
    quant_l: torch.Tensor,
    iscale: torch.Tensor,
    quant_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sum_w = torch.sum(weights, dim=1, keepdim=True)
    sum_x = torch.sum(weights * data, dim=1, keepdim=True)
    wq = weights * quant_l
    sum_l = torch.sum(wq, dim=1, keepdim=True)
    sum_l2 = torch.sum(wq * quant_l, dim=1, keepdim=True)
    sum_xl = torch.sum(wq * data, dim=1, keepdim=True)

    det = sum_w * sum_l2 - sum_l * sum_l
    valid = det > 0
    det_safe = torch.where(valid, det, torch.ones_like(det))

    scale_lsq = (sum_w * sum_xl - sum_x * sum_l) / det_safe
    offset_lsq = (sum_l2 * sum_x - sum_l * sum_xl) / det_safe

    scale = torch.where(valid, scale_lsq, _safe_reciprocal(iscale))
    offset = torch.where(valid, offset_lsq, quant_offset)
    return scale, offset


def _refine_symmetric(
    data: torch.Tensor,
    weights: torch.Tensor,
    quant_l: torch.Tensor,
    iscale: torch.Tensor,
    quant_offset: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    wq = weights * quant_l
    sum_l2 = torch.sum(wq * quant_l, dim=1, keepdim=True)
    sum_xl = torch.sum(wq * data, dim=1, keepdim=True)

    valid = sum_l2 > 0
    sum_l2_safe = torch.where(valid, sum_l2, torch.ones_like(sum_l2))
    scale = torch.where(valid, sum_xl / sum_l2_safe, _safe_reciprocal(iscale))
    offset = torch.zeros_like(scale)
    return scale, offset


def _kquant_search(
    data: torch.Tensor,
    weights: torch.Tensor,
    normalizer: torch.Tensor,
    quant_offset: torch.Tensor,
    l_min: float,
    l_max: float,
    initial_factor: float,
    candidate_factors: tuple[float, ...],
    refine_fn: RefineFn,
) -> tuple[torch.Tensor, torch.Tensor]:
    nontrivial = normalizer != 0
    ones = torch.ones_like(normalizer)
    norm_safe = torch.where(nontrivial, normalizer, ones)

    def quantize(factor: float) -> tuple[torch.Tensor, torch.Tensor]:
        iscale = torch.where(nontrivial, factor / norm_safe, ones)
        quant_l = torch.clamp(torch.round(iscale * (data - quant_offset)), l_min, l_max)
        return iscale, quant_l

    def mad_of(scale: torch.Tensor, offset: torch.Tensor, quant_l: torch.Tensor) -> torch.Tensor:
        diff = scale * quant_l + offset - data
        return torch.sum(weights * diff * diff, dim=1, keepdim=True)

    iscale, quant_l = quantize(initial_factor)
    scale = _safe_reciprocal(iscale)
    offset = quant_offset.expand_as(scale).clone()
    best_mad = mad_of(scale, offset, quant_l)

    for factor in candidate_factors:
        iscale_c, quant_l_c = quantize(factor)
        scale_c, offset_c = refine_fn(data, weights, quant_l_c, iscale_c, quant_offset)
        mad = mad_of(scale_c, offset_c, quant_l_c)
        accept = mad < best_mad
        scale = torch.where(accept, scale_c, scale)
        offset = torch.where(accept, offset_c, offset)
        best_mad = torch.where(accept, mad, best_mad)

    return scale, offset


@torch.no_grad()
def kquant_find_qparams(
    weight: torch.Tensor,
    group_size: int,
    maxq: int,
    minq: int,
    symmetric: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute k-quant per-group scale and zero point for a 2D weight tensor.

    Args:
        weight: 2D tensor of shape ``(out_features, in_features)``. For embeddings
            this is ``(num_embeddings, embedding_dim)``.
        group_size: Group size along the last dimension. Must be > 0 and evenly
            divide ``weight.shape[-1]``.
        maxq: Inclusive maximum integer code (as produced by ``get_maxq_minq``).
        minq: Inclusive minimum integer code (as produced by ``get_maxq_minq``).
        symmetric: When True, run the symmetric variant (zero point fixed to the
            midpoint of the quantization range); otherwise run the asymmetric
            variant that solves for both per-group scale and zero point.

    Returns:
        Tuple ``(scales, zero_points)`` matching ``WeightQuantizer.find_qparams``:
            * ``scales``: shape ``(out_features, num_groups)``, dtype matches the
              input weight dtype.
            * ``zero_points``: shape ``(out_features, num_groups)``, dtype
              ``int32``, values in ``[minq, maxq]``.

    """
    if maxq <= minq:
        raise ValueError(f"k-quant requires maxq > minq, got maxq={maxq}, minq={minq}.")
    if group_size <= 0:
        raise ValueError(f"k-quant requires group_size > 0, got {group_size}.")
    if weight.dim() != 2:
        raise ValueError(f"Expected a 2D weight tensor, got shape {tuple(weight.shape)}.")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({group_size}) for k-quant.")

    orig_dtype = weight.dtype
    data = weight.detach().to(torch.float32).reshape(-1, group_size)

    sum_x2 = torch.sum(data * data, dim=1, keepdim=True)
    av_x = torch.sqrt(sum_x2 / group_size)
    weights = av_x + torch.abs(data)

    midq = (maxq + minq + 1) // 2
    qrange = maxq - minq

    if symmetric:
        l_min, l_max = float(minq - midq), float(maxq - midq)
        normalizer = torch.max(torch.abs(data), dim=1, keepdim=True).values
        quant_offset = torch.zeros_like(normalizer)
        nmax = l_max
        candidate_factors = tuple(nmax + _SYM_STEP_DELTA * s for s in _SYM_STEPS)
        scale, _ = _kquant_search(
            data,
            weights,
            normalizer,
            quant_offset,
            l_min,
            l_max,
            initial_factor=nmax,
            candidate_factors=candidate_factors,
            refine_fn=_refine_symmetric,
        )
        zero_point = torch.full_like(scale, float(midq)).to(torch.int32)
    else:
        l_min, l_max = float(minq), float(maxq)
        rmin = torch.min(data, dim=1, keepdim=True).values
        rmax = torch.max(data, dim=1, keepdim=True).values
        normalizer = rmax - rmin
        candidate_factors = tuple(_ASYM_RRMIN + _ASYM_RDELTA * s + qrange for s in range(_ASYM_NSTEP))
        scale, offset = _kquant_search(
            data,
            weights,
            normalizer,
            rmin,
            l_min,
            l_max,
            initial_factor=float(qrange),
            candidate_factors=candidate_factors,
            refine_fn=_refine_asymmetric,
        )
        zero_point = torch.clamp(torch.round(float(minq) - offset / scale), l_min, l_max).to(torch.int32)

    num_groups = in_features // group_size
    scales = scale.reshape(out_features, num_groups).to(orig_dtype).contiguous()
    zero_points = zero_point.reshape(out_features, num_groups).contiguous()
    return scales, zero_points


class KQuant(Pass):
    """K-quant weight-only quantization (PyTorch-native).

    Per-group weight quantization using the iterative weighted-least-squares
    search from llama.cpp's ggml k-quants. Supports both asymmetric (scale and
    zero point) and symmetric (scale only) variants for 4- and 8-bit weights of
    ``nn.Linear`` and ``nn.Embedding`` modules.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        config = get_quantizer_config(allow_embeds=True)
        config["group_size"] = PassConfigParam(
            type_=int,
            default_value=32,
            description="Group size for k-quant quantization. Must be > 0. Default value is 32.",
        )
        return config

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        if config.group_size <= 0 and config.group_size != -1:
            logger.info("group_size must be -1 or greater than 0")
            return False

        return True

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        """Run k-quant quantization on the model.

        Args:
            model: The HuggingFace model to quantize.
            config: Configuration object containing quantization parameters.
            output_model_path: Path where the quantized model will be saved.

        Returns:
            HfModelHandler for the quantized model.

        """
        wrapper, qcfg, retie_word_embeddings = prepare_model(model, config, allow_quantized=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for _name, module in wrapper.model.named_modules():
            if not hasattr(module, "quant_info"):
                continue
            quantizer = module.quant_info.quantizer

            weight = module.weight.data.to(device)
            effective_group_size = quantizer.group_size if quantizer.group_size > 0 else weight.shape[1]
            scales, zero_points = kquant_find_qparams(
                weight,
                group_size=effective_group_size,
                maxq=quantizer.maxq,
                minq=quantizer.minq,
                symmetric=quantizer.symmetric,
            )
            module.quant_info.scales = scales.to("cpu")
            module.quant_info.zero_points = zero_points.to("cpu")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return finalize(model, output_model_path, wrapper, qcfg, device, retie_word_embeddings=retie_word_embeddings)
