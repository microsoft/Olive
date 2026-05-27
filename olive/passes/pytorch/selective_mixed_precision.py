# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from olive.common.hf.wrapper import ModelWrapper
from olive.common.quant.hf_utils import replace_matching_submodules, sort_layers_by_name
from olive.common.quant.utils import WeightQuantizer
from olive.common.utils import StrEnumBase, get_attr
from olive.constants import PrecisionBits
from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.pytorch.quant_utils import get_qkv_quantization_groups
from olive.passes.pytorch.train_utils import get_calibration_dataset, kl_div_loss, load_hf_base_model
from olive.search.search_parameter import Categorical

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from olive.hardware.accelerator import AcceleratorSpec


logger = logging.getLogger(__name__)

_EPS = 1e-12


class KldMemoryMode(StrEnumBase):
    """Memory mode for KL Divergence gradient based selection.

    - ``auto``: pick one of the modes below based on the model size and free device memory.
    - ``full``: keep a per-layer fp32 gradient accumulator (legacy behavior, highest peak memory).
    - ``multi_gpu``: same algorithm as ``full`` but shard teacher and student across all visible
      CUDA devices via ``accelerate``. Used when the model does not fit on a single GPU but fits
      across multiple GPUs. Falls back to ``low_memory`` if ``accelerate`` is not installed.
    - ``low_memory``: stream the alignment to a scalar accumulator per layer; teacher and student
      stay on device.
    - ``offload``: also keep teacher and student off device when not in use; lowest peak memory.
    """

    AUTO = "auto"
    FULL = "full"
    MULTI_GPU = "multi_gpu"
    LOW_MEMORY = "low_memory"
    OFFLOAD = "offload"


# ---------------------------------------------------------------------------
# Sensitivity scoring strategies
# ---------------------------------------------------------------------------
#
# Every score-based algorithm is a self-contained ``ScoringStrategy`` that owns
# the full pipeline from model + calibration inputs to per-unit scalar scores:
#
#   1. ``compute_module_stats`` produces ``{module_name: stats_dict}`` for every
#      scored module, where ``stats_dict`` is a plain ``dict[str, float]`` whose
#      schema is private to the strategy.
#   2. ``combine_stats`` merges per-module stats across the members of a fusion
#      group (q/k/v projections that ONNX export fuses into one matmul).
#   3. ``score`` turns one stats record + total numel into a scalar sensitivity
#      score (lower == more sensitive).
#
# The shared ``compute_unit_scores`` glue calls (1) then aggregates QKV groups
# into single selection units (members forced to share precision) and leaves
# every other module as a singleton unit; the resulting unit-keyed scalar
# scores are consumed by the algorithm-agnostic ``get_overrides_from_scores``.
#
# QKV exactness: ``WeightQuantizer`` groups along the input-channel dim, so
# concatenating q/k/v along the output dim preserves every quant group within
# a single member. Therefore aggregating per-member stats is bit-equivalent to
# scoring the fused [Q|K|V] matmul.


class ScoringStrategy(ABC):
    """Compute per-(fusion-)unit sensitivity scores for a single algorithm."""

    def compute_unit_scores(
        self,
        handler: HfModelHandler | None,
        model_wrapper: ModelWrapper,
        device: str,
        qkv_groups: Iterable[Sequence[str]],
    ) -> tuple[dict[tuple[str, ...], int], dict[tuple[str, ...], float]]:
        """Return ``(unit_numels, unit_scores)`` keyed by tuples of co-promoted module names."""
        module_numels, module_stats = self.compute_module_stats(handler, model_wrapper, device)
        return self._aggregate(module_numels, module_stats, qkv_groups)

    def _aggregate(
        self,
        module_numels: dict[str, int],
        module_stats: dict[str, dict[str, float]],
        qkv_groups: Iterable[Sequence[str]],
    ) -> tuple[dict[tuple[str, ...], int], dict[tuple[str, ...], float]]:
        # QKV members must share the same precision because ONNX export fuses q/k/v into a
        # single MatMul. For each algorithm, combining per-member intermediate stats and then
        # scoring is bit-equivalent (up to float-summation order) to scoring the row-concatenated
        # weight ``F = cat([Q, K, V], dim=0)`` because: (a) ``WeightQuantizer`` is row/group-wise
        # along dim=1, so ``Q(F) = cat(Q(Q_w), Q(K_w), Q(V_w))`` -- the per-row/per-group quant
        # parameters are unchanged by row concatenation; (b) Frobenius-squared norms decompose
        # additively over row concatenation (justifies SNR/SNR_RELATIVE sum aggregation);
        # (c) max-over-rows of a row-concatenation equals max of per-block maxes (justifies
        # IQE/IQE_RELATIVE max aggregation); (d) gradient-vector inner products decompose
        # additively when parameters are concatenated (justifies KLD sum aggregation).
        # Caveat: this equivalence requires per-row/per-group quantization (``group_size != 0``).
        # With per-tensor quantization (``group_size == 0``) the fused tensor would use a single
        # shared scale, which the per-member aggregation cannot replicate.
        grouped: set[str] = set()
        unit_numels: dict[tuple[str, ...], int] = {}
        unit_scores: dict[tuple[str, ...], float] = {}

        for group in qkv_groups:
            unit = tuple(group)
            if not unit or not all(name in module_stats for name in unit):
                continue
            numel = sum(module_numels[name] for name in unit)
            combined = self.combine_stats([module_stats[name] for name in unit])
            unit_numels[unit] = numel
            unit_scores[unit] = self.score(combined, numel)
            grouped.update(unit)

        for name, stats in module_stats.items():
            if name in grouped:
                continue
            unit_numels[(name,)] = module_numels[name]
            unit_scores[(name,)] = self.score(stats, module_numels[name])

        return unit_numels, unit_scores

    @abstractmethod
    def compute_module_stats(
        self,
        handler: HfModelHandler | None,
        model_wrapper: ModelWrapper,
        device: str,
    ) -> tuple[dict[str, int], dict[str, dict[str, float]]]:
        """Return ``(module_numels, module_stats)`` for every scored module."""

    @abstractmethod
    def combine_stats(self, stats_list: list[dict[str, float]]) -> dict[str, float]:
        """Merge per-module stats across the members of a fusion (QKV) group."""

    @abstractmethod
    def score(self, stats: dict[str, float], numel: int) -> float:
        """Reduce one stats record + total numel to a scalar; lower == more sensitive."""


# ---------------------------------------------------------------------------
# SNR / IQE family
# ---------------------------------------------------------------------------


def _snr_squared_norms(x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    """Return ``(||x||^2, ||x-y||^2)`` as Python floats."""
    x = x.flatten().float()
    y = y.flatten().float()
    signal_sq = torch.dot(x, x).item()
    noise_diff = x - y
    noise_sq = torch.dot(noise_diff, noise_diff).item()
    return signal_sq, noise_sq


def _snr_db(signal_sq: float, noise_sq: float) -> float:
    return 10 * math.log10(max(signal_sq, _EPS * _EPS) / max(noise_sq, _EPS * _EPS))


def _iqe_raw(x: torch.Tensor, y: torch.Tensor) -> float:
    """Return ``max_row(mean_last((x-y)^2))`` as a Python float."""
    # based on nncf implementation at
    # https://github.com/openvinotoolkit/nncf/blob/develop/src/nncf/quantization/algorithms/weight_compression/weight_lowering.py
    return torch.pow(x.float() - y.float(), 2).mean(-1).max().item()


class _LinearScanStrategy(ScoringStrategy):
    """Shared linear-module scan used by all SNR/IQE strategies."""

    def __init__(self, quantizer: WeightQuantizer, high_quantizer: WeightQuantizer):
        self.quantizer = quantizer
        self.high_quantizer = high_quantizer

    def compute_module_stats(self, handler, model_wrapper, device):
        module_numels: dict[str, int] = {}
        module_stats: dict[str, dict[str, float]] = {}

        @torch.no_grad()
        def process(module: torch.nn.Module, module_name: str):
            module_numels[module_name] = module.weight.numel()
            module.to(device)
            module_stats[module_name] = self._stats_for_weight(module.weight)
            return module.cpu()

        replace_matching_submodules(
            model_wrapper.model,
            lambda module, _: isinstance(module, torch.nn.Linear),
            process,
            description="Computing sensitivity scores",
        )
        return module_numels, module_stats

    @abstractmethod
    def _stats_for_weight(self, weight: torch.Tensor) -> dict[str, float]:
        """Compute the per-module stats record for ``weight``."""


class _SnrStrategy(_LinearScanStrategy):
    def _stats_for_weight(self, weight):
        signal_sq, noise_sq = _snr_squared_norms(weight, self.quantizer.fake_quantize(weight))
        return {"signal_sq": signal_sq, "noise_sq": noise_sq}

    def combine_stats(self, stats_list):
        return {
            "signal_sq": sum(s["signal_sq"] for s in stats_list),
            "noise_sq": sum(s["noise_sq"] for s in stats_list),
        }

    def score(self, stats, numel):
        return _snr_db(stats["signal_sq"], stats["noise_sq"])


class _SnrRelativeStrategy(_SnrStrategy):
    def _stats_for_weight(self, weight):
        stats = super()._stats_for_weight(weight)
        _, stats["high_noise_sq"] = _snr_squared_norms(weight, self.high_quantizer.fake_quantize(weight))
        return stats

    def combine_stats(self, stats_list):
        combined = super().combine_stats(stats_list)
        combined["high_noise_sq"] = sum(s["high_noise_sq"] for s in stats_list)
        return combined

    def score(self, stats, numel):
        return _snr_db(stats["signal_sq"], stats["noise_sq"]) - _snr_db(stats["signal_sq"], stats["high_noise_sq"])


class _IqeStrategy(_LinearScanStrategy):
    def _stats_for_weight(self, weight):
        return {"iqe_raw": _iqe_raw(weight, self.quantizer.fake_quantize(weight))}

    def combine_stats(self, stats_list):
        # max-of-rows over the concatenated [Q|K|V] equals the max of per-member maxes.
        return {"iqe_raw": max(s["iqe_raw"] for s in stats_list)}

    def score(self, stats, numel):
        return 1.0 / (stats["iqe_raw"] + _EPS)


class _IqeRelativeStrategy(_IqeStrategy):
    def _stats_for_weight(self, weight):
        return {
            "iqe_raw": _iqe_raw(weight, self.quantizer.fake_quantize(weight)),
            "high_iqe_raw": _iqe_raw(weight, self.high_quantizer.fake_quantize(weight)),
        }

    def combine_stats(self, stats_list):
        return {
            "iqe_raw": max(s["iqe_raw"] for s in stats_list),
            "high_iqe_raw": max(s["high_iqe_raw"] for s in stats_list),
        }

    def score(self, stats, numel):
        # (1/iqe_raw) / (1/high_iqe_raw); equivalent per member, max-aggregated.
        return (1.0 / (stats["iqe_raw"] + _EPS)) / (1.0 / (stats["high_iqe_raw"] + _EPS))


# ---------------------------------------------------------------------------
# KL-divergence gradient
# ---------------------------------------------------------------------------


class _KldGradientStrategy(ScoringStrategy):
    """KL-divergence gradient based sensitivity (mlx-lm style).

    Based on mlx-lm implementation at https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/quant/dynamic_quant.py
    """

    # Memory budgeting constants for AUTO mode selection.
    _ACTIVATION_BUDGET_RATIO = 0.2  # headroom budgeted for activations as a fraction of param bytes
    _MEMORY_SAFETY_FACTOR = 1.2  # multiplicative safety factor for allocator fragmentation
    _FP32_BYTES_PER_ELEMENT = 4
    _FREE_MEMORY_BUDGET_RATIO = 0.85  # leave ~15% headroom in free-memory budget

    def __init__(
        self,
        quantizer: WeightQuantizer,
        high_quantizer: WeightQuantizer,
        memory_mode: KldMemoryMode = KldMemoryMode.AUTO,
    ):
        self.quantizer = quantizer
        self.high_quantizer = high_quantizer
        self.memory_mode = memory_mode

    def combine_stats(self, stats_list):
        return {"alignment": sum(s["alignment"] for s in stats_list)}

    def score(self, stats, numel):
        # Negate so lower == more sensitive; normalize per million parameters.
        return -stats["alignment"] / (max(numel, 1) / 1e6)

    # -- Memory mode resolution -------------------------------------------------

    @classmethod
    def _estimate_memory_bytes(cls, model: torch.nn.Module) -> tuple[int, int, int]:
        """Return ``(param_bytes, full_estimate, low_estimate)`` peak-memory estimates."""
        param_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        linear_grad_bytes = sum(
            module.weight.numel() * cls._FP32_BYTES_PER_ELEMENT
            for module in model.modules()
            if isinstance(module, torch.nn.Linear)
        )
        activation_budget = int(cls._ACTIVATION_BUDGET_RATIO * param_bytes)

        full_estimate = int((2 * param_bytes + linear_grad_bytes + activation_budget) * cls._MEMORY_SAFETY_FACTOR)
        low_estimate = int((2 * param_bytes + activation_budget) * cls._MEMORY_SAFETY_FACTOR)
        return param_bytes, full_estimate, low_estimate

    @classmethod
    def _memory_budget(cls, free_bytes: int) -> int:
        return int(free_bytes * cls._FREE_MEMORY_BUDGET_RATIO)

    @classmethod
    def _multi_gpu_max_memory(cls, model: torch.nn.Module, free_per_gpu: list[int]) -> dict[int, int]:
        """Per-GPU model-copy limits that leave room for FULL-mode KLD memory."""
        param_bytes, full_estimate, _ = cls._estimate_memory_bytes(model)
        # Cap each GPU at the parameter share of the full estimate so the remainder of the budget
        # stays free for the second model copy, the fp32 grad accumulator, and activations.
        per_model_memory_fraction = param_bytes / full_estimate if full_estimate else 1.0
        return {
            device_idx: int(cls._memory_budget(free_bytes) * per_model_memory_fraction)
            for device_idx, free_bytes in enumerate(free_per_gpu)
        }

    @classmethod
    def resolve_memory_mode(
        cls,
        model: torch.nn.Module,
        device: str,
        memory_mode: KldMemoryMode,
    ) -> KldMemoryMode:
        """Resolve ``KldMemoryMode.AUTO`` to a concrete mode for ``model`` on ``device``."""
        if memory_mode != KldMemoryMode.AUTO:
            return memory_mode

        if not device.startswith("cuda") or not torch.cuda.is_available():
            logger.info("KLD memory mode auto-selected: full (non-CUDA device %s).", device)
            return KldMemoryMode.FULL

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            logger.warning("CUDA reports available but no devices visible; defaulting to offload.")
            return KldMemoryMode.OFFLOAD

        try:
            free_per_gpu = [torch.cuda.mem_get_info(i)[0] for i in range(gpu_count)]
        except Exception as exc:  # pragma: no cover - depends on driver/runtime
            logger.warning("Failed to query free CUDA memory (%s); defaulting to offload.", exc)
            return KldMemoryMode.OFFLOAD

        _, full_estimate, low_estimate = cls._estimate_memory_bytes(model)
        single_gpu_budget = cls._memory_budget(free_per_gpu[0])
        multi_gpu_budget = sum(cls._memory_budget(free_bytes) for free_bytes in free_per_gpu)

        if full_estimate <= single_gpu_budget:
            chosen = KldMemoryMode.FULL
        elif gpu_count > 1 and full_estimate <= multi_gpu_budget:
            chosen = KldMemoryMode.MULTI_GPU
        elif low_estimate <= single_gpu_budget:
            chosen = KldMemoryMode.LOW_MEMORY
        else:
            chosen = KldMemoryMode.OFFLOAD

        logger.info(
            "KLD memory mode auto-selected: %s (gpus=%d, full=%.2f GB, low=%.2f GB,"
            " single_budget=%.2f GB, multi_budget=%.2f GB).",
            chosen,
            gpu_count,
            full_estimate / 1e9,
            low_estimate / 1e9,
            single_gpu_budget / 1e9,
            multi_gpu_budget / 1e9,
        )
        return chosen

    # -- Module-stats pipeline --------------------------------------------------

    def compute_module_stats(self, handler, model_wrapper, device):
        # based on mlx-lm implementation at https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/quant/dynamic_quant.py
        from tqdm.auto import tqdm

        model = model_wrapper.model
        # TODO(jambayk): make data_config configurable
        data = get_calibration_dataset(handler, max_seq_len=512, max_samples=256)

        resolved_mode = self.resolve_memory_mode(model, device, self.memory_mode)
        if resolved_mode == KldMemoryMode.MULTI_GPU:
            if not (device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.device_count() > 1):
                logger.warning(
                    "kld_memory_mode=multi_gpu requires at least two visible CUDA devices; falling back to low_memory."
                )
                resolved_mode = KldMemoryMode.LOW_MEMORY
            else:
                import importlib.util

                if importlib.util.find_spec("accelerate") is None:
                    logger.warning(
                        "kld_memory_mode=multi_gpu requires the 'accelerate' package; falling back to low_memory."
                    )
                    resolved_mode = KldMemoryMode.LOW_MEMORY

        # Offloading between host and device is only meaningful on a non-CPU device; on CPU the
        # transfers degenerate to no-ops, so we keep the low-memory path to avoid redundant work.
        offload = resolved_mode == KldMemoryMode.OFFLOAD and device != "cpu"
        multi_gpu = resolved_mode == KldMemoryMode.MULTI_GPU
        # MULTI_GPU runs the same per-layer fp32 grad accumulator algorithm as FULL, just sharded.
        full_memory = resolved_mode == KldMemoryMode.FULL or multi_gpu

        if multi_gpu:
            from accelerate import dispatch_model, infer_auto_device_map

            # Keep both copies on CPU before dispatching so deepcopy is safe and the device map
            # can be inferred once on the un-dispatched model.
            model.to("cpu").eval()
            q_model = deepcopy(model).eval()
            no_split = getattr(model, "_no_split_modules", None) or []

            free_per_gpu = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
            max_memory = self._multi_gpu_max_memory(model, free_per_gpu)

            device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split)
            # Coalesce any sub-decoder-layer placements onto a single device so accelerate hooks
            # do not need to cross device boundaries inside a transformer block (which breaks
            # pointwise ops like the MLP gate*up product). Use the model-type-aware layer prefix
            # (e.g. ``model.layers``, ``transformer.h``, ``gpt_neox.layers``) instead of hard-
            # coding ``model.layers``.
            layer_prefix = model_wrapper.get_layers(return_name=True)[1]
            prefix_parts = layer_prefix.split(".")
            layer_key_len = len(prefix_parts) + 1  # prefix + layer index
            coalesced_map: dict[str, object] = {}
            layer_devices: dict[str, object] = {}
            for module_name, mapped_device in device_map.items():
                parts = module_name.split(".")
                if len(parts) >= layer_key_len and parts[: len(prefix_parts)] == prefix_parts:
                    layer_key = ".".join(parts[:layer_key_len])
                    layer_devices.setdefault(layer_key, mapped_device)
                    coalesced_map[module_name] = layer_devices[layer_key]
                else:
                    coalesced_map[module_name] = mapped_device
            device_map = coalesced_map
            if any(str(mapped_device) in {"cpu", "disk"} for mapped_device in device_map.values()):
                logger.warning(
                    "Unable to place kld_memory_mode=multi_gpu fully on CUDA devices; falling back to low_memory."
                )
                multi_gpu = False
                full_memory = False
            else:
                # Verify no decoder layer's submodules are spread across devices.
                layer_groups: dict[str, set] = {}
                for module_name, mapped_device in device_map.items():
                    parts = module_name.split(".")
                    if len(parts) >= layer_key_len and parts[: len(prefix_parts)] == prefix_parts:
                        layer_groups.setdefault(parts[len(prefix_parts)], set()).add(str(mapped_device))
                split_layers = [layer for layer, devices in layer_groups.items() if len(devices) > 1]
                if split_layers:
                    logger.warning(
                        "kld_memory_mode=multi_gpu device_map split decoder layer(s) %s across "
                        "devices; falling back to low_memory.",
                        split_layers[:5],
                    )
                    multi_gpu = False
                    full_memory = False
                else:
                    layer_device_counts: dict[str, int] = {}
                    for devices in layer_groups.values():
                        # layer_groups[layer] is a singleton set after coalescing succeeded above.
                        (dev,) = devices
                        layer_device_counts[dev] = layer_device_counts.get(dev, 0) + 1
                    logger.info(
                        "kld_memory_mode=multi_gpu device_map: %d decoder layers across %s (total %d module entries).",
                        len(layer_groups),
                        layer_device_counts,
                        len(device_map),
                    )
                    model = dispatch_model(model, device_map=device_map).eval()
                    q_model = dispatch_model(q_model, device_map=device_map).eval()
        if not multi_gpu:
            model.to("cpu" if offload else device).eval()
            q_model = deepcopy(model).to(device).eval()

        # freeze all parameters
        for param in q_model.parameters():
            param.requires_grad = False
        # enable gradient checkpointing
        q_model.gradient_checkpointing_enable()

        # replace the weights in qmodel with low-bit quantized weights
        module_numels: dict[str, int] = {}
        q_layers: dict[str, torch.nn.Module] = {}
        sensitivity_sums: dict[str, float] = {}
        grad_accum: dict[str, torch.Tensor] = {}

        @torch.no_grad()
        def process_module(module: torch.nn.Module, module_name: str):
            module_numels[module_name] = module.weight.numel()
            low_w = self.quantizer.fake_quantize(module.weight.data)
            module.weight.data = low_w
            module.weight.requires_grad = True
            q_layers[module_name] = module
            sensitivity_sums[module_name] = 0.0
            if full_memory:
                grad_accum[module_name] = torch.zeros_like(module.weight.data, dtype=torch.float32)
            return module

        replace_matching_submodules(
            q_model,
            lambda module, _: isinstance(module, torch.nn.Linear),
            process_module,
            description="Preparing for sensitivity estimation",
        )

        def empty_device_cache():
            if not (device.startswith("cuda") and torch.cuda.is_available()):
                return
            if multi_gpu:
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
            else:
                torch.cuda.empty_cache()

        if offload:
            q_model.to("cpu")
            empty_device_cache()

        @torch.no_grad()
        def get_teacher_logits(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
            if offload:
                model.to(device)
            teacher_logits = model(**inputs).logits
            if offload:
                model.to("cpu")
                empty_device_cache()
            return teacher_logits

        @torch.no_grad()
        def accumulate_full_grads():
            for module_name, layer in q_layers.items():
                if layer.weight.grad is None:
                    raise ValueError(f"Missing gradient for {module_name} while estimating KLD sensitivity.")
                grad_accum[module_name] += layer.weight.grad.data.detach().float()

        @torch.no_grad()
        def accumulate_streaming_sensitivities():
            for module_name, layer in q_layers.items():
                if layer.weight.grad is None:
                    raise ValueError(f"Missing gradient for {module_name} while estimating KLD sensitivity.")

                source_weight = get_attr(model, module_name).weight.data.to(layer.weight.device)
                high_w = self.high_quantizer.fake_quantize(source_weight)
                alignment = layer.weight.grad.data.detach().float() * (layer.weight.data - high_w).float()
                sensitivity_sums[module_name] += alignment.sum().item()
                del source_weight, high_w

        for batch in tqdm(data, desc="Estimating sensitivities"):
            inputs = {k: v.to(device) for k, v in batch.items()}

            teacher_logits = get_teacher_logits(inputs)

            if offload:
                q_model.to(device)
            student_logits = q_model(**inputs).logits
            loss = kl_div_loss(student_logits, teacher_logits).mean()
            loss.backward()

            if full_memory:
                accumulate_full_grads()
            else:
                accumulate_streaming_sensitivities()
            q_model.zero_grad(set_to_none=True)
            if offload:
                q_model.to("cpu")
                empty_device_cache()
            del teacher_logits, student_logits, loss

        if full_memory:
            with torch.no_grad():
                for module_name, layer in q_layers.items():
                    avg_grad = grad_accum[module_name] / len(data)
                    high_w = self.high_quantizer.fake_quantize(get_attr(model, module_name).weight.data)
                    sensitivity_sums[module_name] = (avg_grad * (layer.weight.data - high_w)).sum().item() * len(data)

        module_stats = {name: {"alignment": sensitivity_sums[name] / len(data)} for name in q_layers}
        return module_numels, module_stats


_SCORING_STRATEGIES: dict[str, type[ScoringStrategy]] = {}


def _register_strategy(algorithm: str, strategy_cls: type[ScoringStrategy]) -> None:
    _SCORING_STRATEGIES[algorithm] = strategy_cls


class SelectiveMixedPrecision(Pass):
    """Annotate the model with mixed precision information.

    This pass is used to annotate the model with mixed precision information, which can be used by other passes
    to quantize the model. The pass will add a model attribute `mixed_precision_info` that contains
    information about the default precision along with overrides for specific layers.

    The supported algorithms are:
    - Layer id based heuristic:
        - k_quant_last: LM head in high precision.
        - k_quant_down: LM head + Down projection from first 1/8 and last 1/8 layers, and every 3rd layer in between in high precision.
        - k_quant_mixed: LM head + QKV and Down projection from first 1/8 and last 1/8 layers, and every 3rd layer in between in high precision.
    - Sensitivity score based:
        - snr: Signal-to-Noise Ratio based selection.
        - snr_relative: Relative SNR (between low and high precision) based selection.
        - iqe: Inverse of Integer Quantization Error based selection.
        - iqe_relative: Relative IQE (between low and high precision) based selection.
        - kld_gradient: KL Divergence gradient based selection.

    For ``kld_gradient`` the peak memory required for KL Divergence scoring can be tuned via
    ``kld_memory_mode``, which supports ``auto`` (default; picks based on the model size and free
    device memory), ``full``, ``multi_gpu`` (shards the scoring forward across all visible CUDA
    devices with ``accelerate``), ``low_memory``, and ``offload``.

    The override map produced by this pass groups Q/K/V projections in the same attention block so
    they always share precision, which is required for ModelBuilder's GQA fusion: ONNX export
    fuses q/k/v into a single matmul, so the score-based algorithms aggregate per-projection
    stats into the score of the fused matmul before deciding which units to promote.
    """

    class Algorithm(StrEnumBase):
        """The algorithm to use for mixed precision."""

        IQE = "iqe"
        IQE_RELATIVE = "iqe_relative"
        K_QUANT_DOWN = "k_quant_down"
        K_QUANT_MIXED = "k_quant_mixed"
        K_QUANT_LAST = "k_quant_last"
        KLD_GRADIENT = "kld_gradient"
        SNR = "snr"
        SNR_RELATIVE = "snr_relative"

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "algorithm": PassConfigParam(
                type_=SelectiveMixedPrecision.Algorithm,
                required=False,
                search_defaults=Categorical(
                    [
                        SelectiveMixedPrecision.Algorithm.K_QUANT_DOWN,
                        SelectiveMixedPrecision.Algorithm.K_QUANT_MIXED,
                        SelectiveMixedPrecision.Algorithm.K_QUANT_LAST,
                    ]
                ),
                description="The algorithm to use for mixed precision.",
            ),
            "bits": PassConfigParam(
                type_=PrecisionBits,
                default_value=PrecisionBits.BITS4,
                description="The default precision bits.",
            ),
            "group_size": PassConfigParam(
                type_=int,
                default_value=128,
                description="The default group size. Only used for snr, iqe and kld_gradient algorithms.",
            ),
            "sym": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether to use symmetric quantization by default. Only used for snr, iqe and kld_gradient"
                    " algorithms."
                ),
            ),
            "high_bits": PassConfigParam(
                type_=PrecisionBits,
                default_value=PrecisionBits.BITS8,
                description="The high precision bits for selected layers.",
            ),
            "high_group_size": PassConfigParam(
                type_=int,
                default_value=None,
                description=(
                    "The group size for high precision layers. Only used for snr, iqe and kld_gradient algorithms. If"
                    " None, use group_size."
                ),
            ),
            "high_sym": PassConfigParam(
                type_=bool,
                default_value=None,
                description=(
                    "Whether to use symmetric quantization for high precision layers. Only used for snr, iqe and"
                    " kld_gradient algorithms. If None, use sym."
                ),
            ),
            "ratio": PassConfigParam(
                type_=float,
                default_value=None,
                description=(
                    "The ratio of default precision parameters to total parameters. Only used for snr, iqe and"
                    " kld_gradient algorithms. Must be provided when using these algorithms."
                ),
            ),
            "kld_memory_mode": PassConfigParam(
                type_=KldMemoryMode,
                default_value=KldMemoryMode.AUTO,
                description=(
                    "Memory mode for kld_gradient. ``auto`` (default) picks among ``full``, ``multi_gpu``,"
                    " ``low_memory`` and ``offload`` based on the model size and free device memory."
                    " ``full`` keeps a per-layer fp32 gradient accumulator (legacy behavior)."
                    " ``multi_gpu`` runs the ``full`` algorithm with teacher and student sharded across all"
                    " visible CUDA devices via ``accelerate``."
                    " ``low_memory`` streams the alignment to a scalar per layer."
                    " ``offload`` also keeps teacher and student off device when not in use."
                ),
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        if not config.algorithm.startswith("k_quant") and (config.ratio is None or not (0 < config.ratio < 1)):
            logger.error("When using %s algorithm, ratio must be provided and between 0 and 1.", config.algorithm)
            return False

        return True

    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        """Run the selective mixed precision pass."""
        if not isinstance(model, HfModelHandler):
            raise ValueError("SelectiveMixedPrecision pass currently only supports HfModelHandler.")

        # clear cached model
        model.model = None
        model_wrapper = ModelWrapper.from_model(load_hf_base_model(model))

        if config.algorithm.startswith("k_quant"):
            default, overrides = self.get_k_quant_config(model_wrapper, config.algorithm, config.bits, config.high_bits)
        else:
            default, overrides = self.get_scored_config(
                model,
                model_wrapper,
                config.algorithm,
                config.bits,
                config.group_size,
                config.sym,
                config.high_bits,
                config.high_group_size if config.high_group_size is not None else config.group_size,
                config.high_sym if config.high_sym is not None else config.sym,
                config.ratio,
                config.kld_memory_mode,
            )

        lm_head_name = model_wrapper.get_lm_head()[1]
        if model_wrapper.config.tie_word_embeddings and lm_head_name in overrides:
            overrides[model_wrapper.get_embeds()[1][0]] = overrides[lm_head_name]

        # sort the overrides for better readability
        overrides = sort_layers_by_name(overrides)

        # Create output model with mixed precision info as model attribute
        # deepcopy is okay since loaded model is not cached
        output_model = deepcopy(model)
        output_model.model_attributes = output_model.model_attributes or {}
        output_model.model_attributes["mixed_precision_info"] = {
            "default": default,
            "overrides": overrides,
        }
        return output_model

    @staticmethod
    def get_k_quant_config(
        model_wrapper: ModelWrapper,
        algorithm: SelectiveMixedPrecision.Algorithm,
        bits: PrecisionBits,
        high_bits: PrecisionBits,
    ) -> tuple[dict, dict[str, dict]]:
        """Get mixed precision config for k-quant algorithms."""
        override_config = {"bits": high_bits}
        overrides = {model_wrapper.get_lm_head()[1]: override_config}

        if algorithm != SelectiveMixedPrecision.Algorithm.K_QUANT_LAST:
            layer_prefix = model_wrapper.get_layers()[1]
            num_layers = model_wrapper.num_hidden_layers
            for layer_idx, layer_wrapper in enumerate(model_wrapper.get_layer_wrappers()):
                if not (
                    layer_idx < num_layers / 8
                    or layer_idx >= 7 * num_layers / 8
                    or ((layer_idx - num_layers // 8) % 3 == 2)
                ):
                    continue

                # Add qkv
                if algorithm == SelectiveMixedPrecision.Algorithm.K_QUANT_MIXED:
                    for attn_input_name in layer_wrapper.get_attention_inputs(return_name=True)[1]:
                        overrides[f"{layer_prefix}.{layer_idx}.{attn_input_name}"] = override_config

                # Add down_proj
                for attn_output_name in layer_wrapper.get_mlp_outputs(return_name=True)[1]:
                    overrides[f"{layer_prefix}.{layer_idx}.{attn_output_name}"] = override_config

        return {"bits": bits}, overrides

    @staticmethod
    def get_overrides_from_scores(
        unit_numels: dict[tuple[str, ...], int],
        unit_scores: dict[tuple[str, ...], float],
        high_override_config: dict,
        ratio: float,
    ) -> tuple[dict[str, dict], int]:
        """Greedily promote selection units to high precision until ``ratio`` is met.

        A selection unit is a tuple of one or more module names that must share precision
        (singletons for standalone modules; q/k/v projections grouped into one). Units are
        ranked by ``unit_scores`` ascending (lower == more sensitive) and promoted in order
        until the cumulative promoted numel reaches ``(1 - ratio)`` of the total.
        """
        threshold = sum(unit_numels.values()) * (1 - ratio)
        overrides: dict[str, dict] = {}
        high_precision_numels = 0
        for unit in sorted(unit_scores, key=unit_scores.get):
            high_precision_numels += unit_numels[unit]
            for module_name in unit:
                overrides[module_name] = high_override_config.copy()
            if high_precision_numels >= threshold:
                break

        return overrides, high_precision_numels

    @staticmethod
    def get_scored_config(
        handler: HfModelHandler,
        model_wrapper: ModelWrapper,
        algorithm: SelectiveMixedPrecision.Algorithm,
        bits: PrecisionBits,
        group_size: int,
        symmetric: bool,
        high_bits: PrecisionBits,
        high_group_size: int,
        high_symmetric: bool,
        ratio: float,
        kld_memory_mode: KldMemoryMode = KldMemoryMode.AUTO,
    ) -> tuple[dict, dict[str, dict]]:
        """Get mixed precision config based on sensitivity scores."""
        quantizer = WeightQuantizer(bits=bits, group_size=group_size, symmetric=symmetric)
        high_quantizer = WeightQuantizer(bits=high_bits, group_size=high_group_size, symmetric=high_symmetric)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        strategy = _make_scoring_strategy(algorithm, quantizer, high_quantizer, kld_memory_mode=kld_memory_mode)
        # ONNX export fuses q/k/v into one matmul, so the q/k/v projections of each attention
        # block always share precision. Aggregating per-member stats into the fused matmul's
        # score is exact (see WeightQuantizer grouping note above).
        qkv_groups = get_qkv_quantization_groups(model_wrapper)
        # Per-member aggregation is bit-equivalent to scoring the fused [Q|K|V] matmul only
        # when each row keeps its own scale (``group_size != 0``). Per-tensor quantization
        # collapses to one scale across the whole tensor, which the per-member sum cannot
        # replicate, so the selection scores would not reflect what the fused matmul sees.
        if qkv_groups and (group_size == 0 or high_group_size == 0):
            raise ValueError(
                "Score-based selective mixed precision does not support per-tensor "
                "quantization (group_size=0) for models with grouped QKV projections; "
                "use group_size=-1 (per-channel) or a positive group size instead."
            )

        unit_numels, unit_scores = strategy.compute_unit_scores(handler, model_wrapper, device, qkv_groups)

        high_override_config = {"bits": high_bits, "group_size": high_group_size, "symmetric": high_symmetric}
        overrides, high_precision_numels = SelectiveMixedPrecision.get_overrides_from_scores(
            unit_numels,
            unit_scores,
            high_override_config,
            ratio,
        )
        logger.info(
            "Selected %d modules for high precision out of %d units. Ratio of low precision: %.4f",
            len(overrides),
            len(unit_numels),
            1 - high_precision_numels / sum(unit_numels.values()),
        )

        return {"bits": bits, "group_size": group_size, "symmetric": symmetric}, overrides


_register_strategy(SelectiveMixedPrecision.Algorithm.SNR, _SnrStrategy)
_register_strategy(SelectiveMixedPrecision.Algorithm.SNR_RELATIVE, _SnrRelativeStrategy)
_register_strategy(SelectiveMixedPrecision.Algorithm.IQE, _IqeStrategy)
_register_strategy(SelectiveMixedPrecision.Algorithm.IQE_RELATIVE, _IqeRelativeStrategy)
_register_strategy(SelectiveMixedPrecision.Algorithm.KLD_GRADIENT, _KldGradientStrategy)


def _make_scoring_strategy(
    algorithm: SelectiveMixedPrecision.Algorithm,
    quantizer: WeightQuantizer,
    high_quantizer: WeightQuantizer,
    *,
    kld_memory_mode: KldMemoryMode = KldMemoryMode.AUTO,
) -> ScoringStrategy:
    """Build the strategy instance for ``algorithm``."""
    strategy_cls = _SCORING_STRATEGIES[algorithm]
    if strategy_cls is _KldGradientStrategy:
        return strategy_cls(quantizer, high_quantizer, memory_mode=kld_memory_mode)
    return strategy_cls(quantizer, high_quantizer)
