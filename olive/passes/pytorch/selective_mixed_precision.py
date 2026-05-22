# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
import math
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
    from olive.hardware.accelerator import AcceleratorSpec


logger = logging.getLogger(__name__)


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

    class KldMemoryMode(StrEnumBase):
        """Memory mode for KL Divergence gradient based selection.

        - ``auto``: pick one of the modes below based on the model size and free device memory.
        - ``full``: keep a per-layer fp32 gradient accumulator (legacy behaviour, highest peak memory).
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
                type_=SelectiveMixedPrecision.KldMemoryMode,
                default_value=SelectiveMixedPrecision.KldMemoryMode.AUTO,
                description=(
                    "Memory mode for kld_gradient. ``auto`` (default) picks among ``full``, ``multi_gpu``,"
                    " ``low_memory`` and ``offload`` based on the model size and free device memory."
                    " ``full`` keeps a per-layer fp32 gradient accumulator (legacy behaviour)."
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
        model_wrapper: ModelWrapper,
        module_numels: dict[str, int],
        module_scores: dict[str, float],
        high_override_config: dict,
        ratio: float,
    ) -> tuple[dict[str, dict], int]:
        """Get high precision overrides from sensitivity scores."""
        qkv_groups = get_qkv_quantization_groups(model_wrapper, set(module_scores))
        grouped_modules = {module_name for group in qkv_groups for module_name in group}

        scored_items = [
            (
                group,
                sum(module_numels[module_name] for module_name in group),
                min(module_scores[name] for name in group),
            )
            for group in qkv_groups
        ]
        scored_items.extend(
            ((module_name,), module_numels[module_name], score)
            for module_name, score in module_scores.items()
            if module_name not in grouped_modules
        )

        threshold = sum(module_numels.values()) * (1 - ratio)
        overrides = {}
        high_precision_numels = 0
        for module_names, numels, _ in sorted(scored_items, key=lambda item: item[2]):
            high_precision_numels += numels
            overrides.update({module_name: high_override_config.copy() for module_name in module_names})
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
    ):
        """Get mixed precision config based on sensitivity scores."""
        quantizer = WeightQuantizer(bits=bits, group_size=group_size, symmetric=symmetric)
        high_quantizer = WeightQuantizer(
            bits=high_bits,
            group_size=high_group_size,
            symmetric=high_symmetric,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if algorithm == SelectiveMixedPrecision.Algorithm.KLD_GRADIENT:
            module_numels, module_scores = SelectiveMixedPrecision.get_kld_scores(
                handler,
                model_wrapper.model,
                algorithm,
                quantizer,
                high_quantizer,
                device,
                kld_memory_mode,
            )
        else:
            module_numels, module_scores = SelectiveMixedPrecision.get_snr_iqe_scores(
                handler,
                model_wrapper.model,
                algorithm,
                quantizer,
                high_quantizer,
                device,
            )

        high_override_config = {"bits": high_bits, "group_size": high_group_size, "symmetric": high_symmetric}
        overrides, high_precision_numels = SelectiveMixedPrecision.get_overrides_from_scores(
            model_wrapper,
            module_numels,
            module_scores,
            high_override_config,
            ratio,
        )
        logger.info(
            "Selected %d modules for high precision out of %d modules. Ratio of low precision: %.4f",
            len(overrides),
            len(module_numels),
            1 - high_precision_numels / sum(module_numels.values()),
        )

        return {"bits": bits, "group_size": group_size, "symmetric": symmetric}, overrides

    @staticmethod
    def get_snr_iqe_scores(
        handler: HfModelHandler,
        model: torch.nn.Module,
        algorithm: SelectiveMixedPrecision.Algorithm,
        quantizer: WeightQuantizer,
        high_quantizer: WeightQuantizer,
        device: str,
    ) -> tuple[dict[str, int], dict[str, float]]:
        """Compute SNR or IQE based sensitivity scores."""
        score_func = (
            SelectiveMixedPrecision.compute_snr if algorithm.startswith("snr") else SelectiveMixedPrecision.compute_iqe
        )

        module_numels = {}
        module_scores = {}

        def should_include(module, _):
            return isinstance(module, torch.nn.Linear)

        @torch.no_grad()
        def process_module(module, module_name):
            module_numels[module_name] = module.weight.numel()
            module.to(device)

            score = score_func(module.weight, quantizer.fake_quantize(module.weight))
            if algorithm.endswith("_relative"):
                high_score = score_func(module.weight, high_quantizer.fake_quantize(module.weight))
                if algorithm.startswith("snr"):
                    # SNR is in dB, so we subtract
                    score -= high_score
                else:
                    # IQE is inverted, so we divide
                    score /= high_score
            module_scores[module_name] = score
            return module.cpu()

        replace_matching_submodules(model, should_include, process_module, description="Computing SNR/IQE scores")
        return module_numels, module_scores

    @staticmethod
    def _estimate_kld_memory_bytes(model: torch.nn.Module) -> tuple[int, int, int]:
        """Estimate parameter bytes and peak KLD memory for FULL and LOW_MEMORY modes."""
        # Activations are bounded by gradient checkpointing; budget this fraction of model bytes as headroom.
        activation_budget_ratio = 0.2
        # Multiplicative safety factor applied to absorb allocator fragmentation and short-lived temporaries.
        memory_safety_factor = 1.2
        # Bytes per element for the fp32 gradient accumulator held by the FULL mode.
        fp32_bytes_per_element = 4

        param_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        linear_grad_bytes = sum(
            module.weight.numel() * fp32_bytes_per_element
            for module in model.modules()
            if isinstance(module, torch.nn.Linear)
        )
        activation_budget = int(activation_budget_ratio * param_bytes)

        full_estimate = int((2 * param_bytes + linear_grad_bytes + activation_budget) * memory_safety_factor)
        low_estimate = int((2 * param_bytes + activation_budget) * memory_safety_factor)
        return param_bytes, full_estimate, low_estimate

    @staticmethod
    def _get_kld_memory_budget(free_bytes: int) -> int:
        """Return the usable free-memory budget for KLD mode selection."""
        # Leave ~15% headroom for allocator fragmentation and underestimated activation peaks.
        free_memory_budget_ratio = 0.85
        return int(free_bytes * free_memory_budget_ratio)

    @staticmethod
    def _get_kld_multi_gpu_max_memory(model: torch.nn.Module, free_per_gpu: list[int]) -> dict[int, int]:
        """Return per-GPU model-copy limits that leave room for FULL-mode KLD memory."""
        param_bytes, full_estimate, _ = SelectiveMixedPrecision._estimate_kld_memory_bytes(model)
        # Cap each GPU at the parameter share of the full estimate so the remainder of the budget
        # stays free for the second model copy, the fp32 grad accumulator, and activations.
        per_model_memory_fraction = param_bytes / full_estimate if full_estimate else 1.0
        return {
            device_idx: int(SelectiveMixedPrecision._get_kld_memory_budget(free_bytes) * per_model_memory_fraction)
            for device_idx, free_bytes in enumerate(free_per_gpu)
        }

    @staticmethod
    def resolve_kld_memory_mode(
        model: torch.nn.Module,
        device: str,
        kld_memory_mode: KldMemoryMode,
    ) -> KldMemoryMode:
        """Resolve ``KldMemoryMode.AUTO`` to a concrete mode for ``model`` on ``device``.

        On CPU we always prefer the ``full`` legacy path since host memory is usually ample.
        On CUDA we estimate the peak device memory for each mode and pick the most accurate
        mode whose estimate fits in free device memory with safety headroom.
        """
        if kld_memory_mode != SelectiveMixedPrecision.KldMemoryMode.AUTO:
            return kld_memory_mode

        if not device.startswith("cuda") or not torch.cuda.is_available():
            logger.info("KLD memory mode auto-selected: full (non-CUDA device %s).", device)
            return SelectiveMixedPrecision.KldMemoryMode.FULL

        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            logger.warning("CUDA reports available but no devices visible; defaulting to offload.")
            return SelectiveMixedPrecision.KldMemoryMode.OFFLOAD

        try:
            free_per_gpu = [torch.cuda.mem_get_info(i)[0] for i in range(gpu_count)]
        except Exception as exc:  # pragma: no cover - depends on driver/runtime
            logger.warning("Failed to query free CUDA memory (%s); defaulting to offload.", exc)
            return SelectiveMixedPrecision.KldMemoryMode.OFFLOAD

        _, full_estimate, low_estimate = SelectiveMixedPrecision._estimate_kld_memory_bytes(model)
        single_gpu_budget = SelectiveMixedPrecision._get_kld_memory_budget(free_per_gpu[0])
        multi_gpu_budget = sum(
            SelectiveMixedPrecision._get_kld_memory_budget(free_bytes) for free_bytes in free_per_gpu
        )

        if full_estimate <= single_gpu_budget:
            chosen = SelectiveMixedPrecision.KldMemoryMode.FULL
        elif gpu_count > 1 and full_estimate <= multi_gpu_budget:
            chosen = SelectiveMixedPrecision.KldMemoryMode.MULTI_GPU
        elif low_estimate <= single_gpu_budget:
            chosen = SelectiveMixedPrecision.KldMemoryMode.LOW_MEMORY
        else:
            chosen = SelectiveMixedPrecision.KldMemoryMode.OFFLOAD

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

    @staticmethod
    def get_kld_scores(
        handler: HfModelHandler,
        model: torch.nn.Module,
        algorithm: SelectiveMixedPrecision.Algorithm,
        quantizer: WeightQuantizer,
        high_quantizer: WeightQuantizer,
        device: str,
        kld_memory_mode: KldMemoryMode = KldMemoryMode.AUTO,
    ) -> tuple[dict[str, int], dict[str, float]]:
        """Compute KL Divergence gradient based sensitivity scores.

        The gradients are computed using a calibration dataset and the KL Divergence loss between
        the outputs of the original model and the fake quantized model.
        """
        # based on mlx-lm implementation at https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/quant/dynamic_quant.py
        from tqdm.auto import tqdm

        # TODO(jambayk): make data_config configurable
        data = get_calibration_dataset(handler, max_seq_len=512, max_samples=256)

        resolved_mode = SelectiveMixedPrecision.resolve_kld_memory_mode(model, device, kld_memory_mode)
        if resolved_mode == SelectiveMixedPrecision.KldMemoryMode.MULTI_GPU:
            if not (device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.device_count() > 1):
                logger.warning(
                    "kld_memory_mode=multi_gpu requires at least two visible CUDA devices; falling back to low_memory."
                )
                resolved_mode = SelectiveMixedPrecision.KldMemoryMode.LOW_MEMORY
            else:
                import importlib.util

                if importlib.util.find_spec("accelerate") is None:
                    logger.warning(
                        "kld_memory_mode=multi_gpu requires the 'accelerate' package; falling back to low_memory."
                    )
                    resolved_mode = SelectiveMixedPrecision.KldMemoryMode.LOW_MEMORY
        # Offloading between host and device is only meaningful on a non-CPU device; on CPU the
        # transfers degenerate to no-ops, so we keep the low-memory path to avoid redundant work.
        offload = resolved_mode == SelectiveMixedPrecision.KldMemoryMode.OFFLOAD and device != "cpu"
        multi_gpu = resolved_mode == SelectiveMixedPrecision.KldMemoryMode.MULTI_GPU
        # MULTI_GPU runs the same per-layer fp32 grad accumulator algorithm as FULL, just sharded.
        full_memory = resolved_mode == SelectiveMixedPrecision.KldMemoryMode.FULL or multi_gpu
        if multi_gpu:
            from accelerate import dispatch_model, infer_auto_device_map

            # Keep both copies on CPU before dispatching so deepcopy is safe and the device map
            # can be inferred once on the un-dispatched model.
            model.to("cpu").eval()
            q_model = deepcopy(model).eval()
            no_split = getattr(model, "_no_split_modules", None) or []

            free_per_gpu = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
            max_memory = SelectiveMixedPrecision._get_kld_multi_gpu_max_memory(model, free_per_gpu)

            device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split)
            # Coalesce any sub-decoder-layer placements onto a single device so accelerate hooks do
            # not need to cross device boundaries inside a transformer block (which breaks pointwise
            # ops like the MLP gate*up product).
            coalesced_map: dict[str, object] = {}
            layer_devices: dict[str, object] = {}
            for module_name, mapped_device in device_map.items():
                parts = module_name.split(".")
                if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                    layer_key = ".".join(parts[:3])
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
                    if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                        layer_groups.setdefault(parts[2], set()).add(str(mapped_device))
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
                    device_counts: dict[str, int] = {}
                    for mapped_device in device_map.values():
                        device_counts[str(mapped_device)] = device_counts.get(str(mapped_device), 0) + 1
                    logger.info(
                        "kld_memory_mode=multi_gpu device_map: %d entries across %s.",
                        len(device_map),
                        device_counts,
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
        module_numels = {}
        q_layers: dict[str, torch.nn.Module] = {}
        sensitivity_sums: dict[str, float] = {}
        grad_accum: dict[str, torch.Tensor] = {}

        def should_include(module, _):
            return isinstance(module, torch.nn.Linear)

        @torch.no_grad()
        def process_module(module, module_name):
            module_numels[module_name] = module.weight.numel()
            low_w = quantizer.fake_quantize(module.weight.data)
            module.weight.data = low_w
            module.weight.requires_grad = True
            q_layers[module_name] = module
            sensitivity_sums[module_name] = 0.0
            if full_memory:
                grad_accum[module_name] = torch.zeros_like(module.weight.data, dtype=torch.float32)
            return module

        replace_matching_submodules(
            q_model, should_include, process_module, description="Preparing for sensitivity estimation"
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
                high_w = high_quantizer.fake_quantize(source_weight)
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
                    high_w = high_quantizer.fake_quantize(get_attr(model, module_name).weight.data)
                    sensitivity_sums[module_name] = (avg_grad * (layer.weight.data - high_w)).sum().item() * len(data)

        # negative sensitivity because lower is more sensitive
        return module_numels, {
            name: -(sensitivity_sums[name] / len(data)) / (module_numels[name] / 1e6) for name in q_layers
        }

    @staticmethod
    def compute_snr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
        """Compute signal-to-noise ratio in dB.

        Args:
            x (torch.Tensor): Original tensor.
            y (torch.Tensor): Quantized tensor.
            eps (float): Small value to avoid division by zero.

        Returns:
            float: SNR value in dB.

        """
        x = x.flatten().float()
        y = y.flatten().float()
        signal_norm = max(torch.norm(x).item(), eps)
        noise_norm = max(torch.norm(x - y).item(), eps)
        return 20 * math.log10(signal_norm / noise_norm)

    @staticmethod
    def compute_iqe(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
        """Compute the inverse of integer quantization error.

        Error is computed as max(mean((x - y)^2)) over the last dimension.

        Args:
            x (torch.Tensor): Original tensor.
            y (torch.Tensor): Quantized tensor.
            eps (float): Small value to avoid division by zero.

        Returns:
            float: Inverse of IQE score.

        """
        # based on nncf implementation at
        # https://github.com/openvinotoolkit/nncf/blob/develop/src/nncf/quantization/algorithms/weight_compression/weight_lowering.py
        iqe = torch.pow(x.float() - y.float(), 2).mean(-1).max().item()
        # invert so that lower score means more sensitive
        return 1 / (iqe + eps)
