# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Based on the MBQ implementation at
# https://github.com/thu-nics/MBQ/tree/a4d460dfb4b1c07b5d1f3ddda6e86d1c90d6e7f1
# --------------------------------------------------------------------------
"""Modality-Balanced Quantization scale reparameterization."""

# pylint: disable=not-callable

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import torch
from torch.nn import functional as F

from olive.common.quant.utils import WeightQuantizer
from olive.common.utils import tensor_data_to_device
from olive.constants import PrecisionBits
from olive.data.config import DataConfig
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf
from olive.passes.pytorch.multimodal_quantization import (
    MultimodalCalibrationMasks,
    modality_balanced_reconstruction_loss,
    split_multimodal_calibration_batch,
    validate_masks_for_activations,
)
from olive.passes.pytorch.train_utils import get_calibration_dataset, load_hf_base_model

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import HfModelHandler

logger = logging.getLogger(__name__)

_SUPPORTED_LAYER_TYPES = {
    "Qwen2DecoderLayer",
    "Qwen2_5_VLDecoderLayer",
    "Qwen2VLDecoderLayer",
}
_DECODER_LAYER_PATHS = (
    "model.language_model.layers",
    "model.layers",
    "language_model.layers",
)


@dataclass(frozen=True)
class _ScaleGroup:
    name: str
    previous: torch.nn.Module
    linears: tuple[torch.nn.Linear, ...]
    capture: torch.nn.Linear
    block_type: str


class _GradientMeans:
    def __init__(self):
        self.vision_sum = 0.0
        self.vision_batches = 0
        self.answer_sum = 0.0
        self.answer_batches = 0

    def update(self, gradient: torch.Tensor, masks: MultimodalCalibrationMasks) -> None:
        validate_masks_for_activations(masks, gradient)
        magnitude = gradient.detach().float().abs()
        if masks.vision.any():
            self.vision_sum += magnitude[masks.vision].mean().item()
            self.vision_batches += 1
        if masks.answer.any():
            self.answer_sum += magnitude[masks.answer].mean().item()
            self.answer_batches += 1

    def ratio(self) -> float:
        if not self.vision_batches or not self.answer_batches:
            raise ValueError("MBQ requires at least one non-empty vision mask and answer mask for every decoder layer.")
        vision_mean = self.vision_sum / self.vision_batches
        answer_mean = self.answer_sum / self.answer_batches
        if answer_mean <= 0:
            raise ValueError("MBQ observed a zero answer-token gradient and cannot compute modality reweighting.")
        return vision_mean / answer_mean


class Mbq(Pass):
    """Apply MBQ's modality-balanced AWQ-style scale folding to a language decoder.

    The pass implements all four scale groups used by MBQ for Qwen2/Qwen2-VL
    decoder layers: norm->QKV, V->O (when dimensions permit), norm->gate/up,
    and up->down.
    Vision and answer masks must already use the decoder sequence coordinates
    produced after multimodal token expansion.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "bits": PassConfigParam(
                type_=PrecisionBits,
                default_value=PrecisionBits.BITS4,
                description="Weight bits used during MBQ reconstruction search and required for downstream quantization.",
            ),
            "group_size": PassConfigParam(
                type_=int,
                default_value=128,
                description="Weight quantization group size used during scale search.",
            ),
            "sym": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether reconstruction search uses symmetric weight quantization.",
            ),
            "n_grid": PassConfigParam(
                type_=int,
                default_value=20,
                description="Number of AWQ exponent candidates in [0, 1).",
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, dict],
                default_value=None,
                description=(
                    "Required multimodal calibration data. Batches must contain input_ids, labels, vision_mask, "
                    "and all processor outputs needed by the model."
                ),
            ),
            "vision_mask_key": PassConfigParam(
                type_=str,
                default_value="vision_mask",
                description="Batch key containing a decoder-coordinate boolean vision-token mask.",
            ),
            "answer_mask_key": PassConfigParam(
                type_=str,
                default_value="answer_mask",
                description="Optional batch key containing the answer-token mask; labels != -100 is the fallback.",
            ),
            "decoder_layer_path": PassConfigParam(
                type_=Optional[str],
                default_value=None,
                description="Explicit dotted decoder ModuleList path. Required when architecture discovery is ambiguous.",
            ),
            "device": PassConfigParam(
                type_=Optional[str],
                default_value=None,
                description="Calibration/search device. Defaults to CUDA when available, otherwise CPU.",
            ),
            "save_processor": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Save the input model's tokenizer and multimodal processor with the checkpoint.",
            ),
        }

    @classmethod
    def validate_config(cls, config: type[BasePassConfig], accelerator_spec: AcceleratorSpec) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False
        bits = config.bits.value if hasattr(config.bits, "value") else int(config.bits)
        if bits not in (2, 4, 8):
            logger.info("MBQ bits must be 2, 4, or 8.")
            return False
        if config.group_size <= 0 and config.group_size != -1:
            logger.info("MBQ group_size must be -1 or greater than 0.")
            return False
        if config.n_grid <= 0:
            logger.info("MBQ n_grid must be greater than 0.")
            return False
        if config.data_config is None:
            logger.info("MBQ requires data_config.")
            return False
        return True

    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        if model.adapter_path:
            raise ValueError("MBQ does not currently support models with adapters.")

        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        pytorch_model = load_hf_base_model(model, torch_dtype=model.get_load_kwargs().get("torch_dtype") or "auto")
        pytorch_model.eval().to(device)
        layers, layer_path = self._resolve_decoder_layers(pytorch_model, config.decoder_layer_path)
        groups = [self._get_scale_groups(layer, layer_idx) for layer_idx, layer in enumerate(layers)]

        calibration_data = get_calibration_dataset(
            model,
            config.data_config,
            include_labels=True,
        )
        if not calibration_data:
            raise ValueError("MBQ calibration data is empty.")

        gradient_means = {
            (layer_idx, block_type): _GradientMeans()
            for layer_idx in range(len(layers))
            for block_type in ("attn", "mlp")
        }
        hidden_states, layer_args, layer_kwargs, calibration_masks = self._collect_calibration(
            pytorch_model,
            layers,
            calibration_data,
            gradient_means,
            device,
            config.vision_mask_key,
            config.answer_mask_key,
        )

        ratios = self._finalize_reweight_ratios(gradient_means, len(layers))
        bits = config.bits.value if hasattr(config.bits, "value") else int(config.bits)
        quantizer = WeightQuantizer(bits=bits, group_size=config.group_size, symmetric=config.sym)

        pytorch_model.to("cpu")
        for layer_idx, (layer, layer_groups) in enumerate(zip(layers, groups)):
            layer.to(device)
            samples = self._capture_layer_samples(
                layer,
                layer_groups,
                hidden_states,
                layer_args,
                layer_kwargs,
                calibration_masks,
                device,
            )
            for group in layer_groups:
                scale = self._search_scale(
                    group.linears,
                    samples[group.name],
                    ratios[(layer_idx, group.block_type)],
                    quantizer,
                    config.n_grid,
                    device,
                )
                self._apply_scale(group.previous, group.linears, scale)
            hidden_states = self._run_layer_samples(
                layer,
                hidden_states,
                layer_args,
                layer_kwargs,
                device,
            )
            layer.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        mbq_config = {
            "algorithm": "mbq",
            "reference": "arXiv:2412.19509",
            "decoder_layer_path": layer_path,
            "bits": bits,
            "group_size": config.group_size,
            "symmetric": config.sym,
            "n_grid": config.n_grid,
            "mask_coordinates": "expanded_decoder_tokens",
        }
        pytorch_model.config.mbq_config = mbq_config
        pytorch_model.save_pretrained(output_model_path)
        if config.save_processor:
            model.save_metadata(output_model_path)
            self._save_processor(model, output_model_path)

        output = inherit_hf_from_hf(model, output_model_path, adapter_path=None)
        output.model_attributes = output.model_attributes or {}
        output.model_attributes["mbq_config"] = mbq_config
        return output

    @staticmethod
    def _resolve_decoder_layers(
        model: torch.nn.Module,
        explicit_path: str | None,
    ) -> tuple[torch.nn.ModuleList, str]:
        candidates = (explicit_path,) if explicit_path else _DECODER_LAYER_PATHS
        matches = []
        for path in candidates:
            try:
                layers = model.get_submodule(path)
            except AttributeError:
                continue
            if isinstance(layers, torch.nn.ModuleList) and layers:
                matches.append((layers, path))

        if not matches:
            raise ValueError(
                "Unable to find a supported language decoder ModuleList. Set decoder_layer_path explicitly."
            )
        if len(matches) > 1 and explicit_path is None:
            supported_matches = [
                match for match in matches if all(type(layer).__name__ in _SUPPORTED_LAYER_TYPES for layer in match[0])
            ]
            if len(supported_matches) != 1:
                raise ValueError(
                    f"Ambiguous decoder layer paths {[path for _, path in matches]}; set decoder_layer_path explicitly."
                )
            matches = supported_matches

        layers, path = matches[0]
        unsupported = sorted({type(layer).__name__ for layer in layers} - _SUPPORTED_LAYER_TYPES)
        if unsupported:
            raise ValueError(
                f"MBQ does not have a verified scale-group adapter for decoder layer types {unsupported}. "
                f"Supported types: {sorted(_SUPPORTED_LAYER_TYPES)}."
            )
        return layers, path

    @staticmethod
    def _get_scale_groups(layer: torch.nn.Module, layer_idx: int) -> list[_ScaleGroup]:
        attention = layer.self_attn
        mlp = layer.mlp
        groups = [
            _ScaleGroup(
                f"layers.{layer_idx}.attn_input",
                layer.input_layernorm,
                (attention.q_proj, attention.k_proj, attention.v_proj),
                attention.q_proj,
                "attn",
            ),
            _ScaleGroup(
                f"layers.{layer_idx}.mlp_input",
                layer.post_attention_layernorm,
                (mlp.gate_proj, mlp.up_proj),
                mlp.gate_proj,
                "mlp",
            ),
            _ScaleGroup(
                f"layers.{layer_idx}.mlp_output",
                mlp.up_proj,
                (mlp.down_proj,),
                mlp.down_proj,
                "mlp",
            ),
        ]
        if attention.v_proj.weight.shape == attention.o_proj.weight.shape:
            groups.insert(
                1,
                _ScaleGroup(
                    f"layers.{layer_idx}.attn_output",
                    attention.v_proj,
                    (attention.o_proj,),
                    attention.o_proj,
                    "attn",
                ),
            )
        return groups

    @classmethod
    def _collect_calibration(
        cls,
        model: torch.nn.Module,
        layers: torch.nn.ModuleList,
        calibration_data: list[dict[str, torch.Tensor]],
        gradient_means: dict[tuple[int, str], _GradientMeans],
        device: str,
        vision_mask_key: str,
        answer_mask_key: str,
    ) -> tuple[list[torch.Tensor], list[tuple], list[dict], list[MultimodalCalibrationMasks]]:
        ratio_modules = {}
        for layer_idx, layer in enumerate(layers):
            ratio_modules[layer.self_attn.o_proj] = (layer_idx, "attn")
            ratio_modules[layer.mlp.down_proj] = (layer_idx, "mlp")

        current_outputs: dict[torch.nn.Module, torch.Tensor] = {}
        hidden_states: list[torch.Tensor] = []
        layer_args: list[tuple] = []
        layer_kwargs: list[dict] = []
        calibration_masks: list[MultimodalCalibrationMasks] = []
        active_masks = None
        handles = []

        def capture_ratio_output(module, _, output):
            if not output.requires_grad:
                output.requires_grad_(True)
            current_outputs[module] = output
            return output

        def capture_first_layer_input(_, args, kwargs):
            nonlocal active_masks
            hidden = kwargs.get("hidden_states")
            if hidden is None:
                if not args:
                    raise ValueError("MBQ could not capture the first decoder layer hidden states.")
                hidden = args[0]
                extra_args = args[1:]
            else:
                extra_args = args
            extra_kwargs = {key: value for key, value in kwargs.items() if key != "hidden_states"}
            validate_masks_for_activations(active_masks, hidden)
            hidden_states.append(cls._detach_to_cpu(hidden))
            layer_args.append(cls._detach_to_cpu(extra_args))
            layer_kwargs.append(cls._detach_to_cpu(extra_kwargs))
            calibration_masks.append(cls._detach_to_cpu(active_masks))

        handles.append(layers[0].register_forward_pre_hook(capture_first_layer_input, with_kwargs=True))
        handles.extend(module.register_forward_hook(capture_ratio_output) for module in ratio_modules)

        original_use_cache = getattr(model.config, "use_cache", None)
        if original_use_cache is not None:
            model.config.use_cache = False
        try:
            for batch in calibration_data:
                current_outputs.clear()
                model_inputs, masks = split_multimodal_calibration_batch(
                    batch,
                    vision_mask_key=vision_mask_key,
                    answer_mask_key=answer_mask_key,
                )
                if "labels" not in model_inputs:
                    raise ValueError("MBQ calibration requires labels to compute modality gradients.")
                active_masks = masks.to(device)
                outputs = model(**tensor_data_to_device(model_inputs, device))
                loss = getattr(outputs, "loss", None)
                if loss is None or loss.ndim != 0:
                    raise ValueError(
                        "MBQ requires the model forward to return a scalar .loss when labels are provided."
                    )

                missing = ratio_modules.keys() - current_outputs.keys()
                if missing:
                    raise ValueError(f"MBQ calibration did not execute {len(missing)} gradient target modules.")

                ratio_targets = list(ratio_modules)
                gradients = torch.autograd.grad(loss, [current_outputs[module] for module in ratio_targets])
                for module, gradient in zip(ratio_targets, gradients):
                    gradient_means[ratio_modules[module]].update(gradient, active_masks)
        finally:
            for handle in handles:
                handle.remove()
            if original_use_cache is not None:
                model.config.use_cache = original_use_cache
        return hidden_states, layer_args, layer_kwargs, calibration_masks

    @staticmethod
    def _detach_to_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, MultimodalCalibrationMasks):
            return MultimodalCalibrationMasks(
                Mbq._detach_to_cpu(value.vision),
                Mbq._detach_to_cpu(value.answer),
            )
        if isinstance(value, tuple):
            return tuple(Mbq._detach_to_cpu(item) for item in value)
        if isinstance(value, list):
            return [Mbq._detach_to_cpu(item) for item in value]
        if isinstance(value, dict):
            return {key: Mbq._detach_to_cpu(item) for key, item in value.items()}
        return value

    @staticmethod
    @torch.no_grad()
    def _capture_layer_samples(
        layer: torch.nn.Module,
        groups: list[_ScaleGroup],
        hidden_states: list[torch.Tensor],
        layer_args: list[tuple],
        layer_kwargs: list[dict],
        calibration_masks: list[MultimodalCalibrationMasks],
        device: str,
    ) -> dict[str, list[tuple[torch.Tensor, MultimodalCalibrationMasks]]]:
        samples = {group.name: [] for group in groups}
        group_by_capture = {group.capture: group for group in groups}
        current_inputs = {}

        def capture_input(module, args):
            current_inputs[module] = args[0]

        handles = [module.register_forward_pre_hook(capture_input) for module in group_by_capture]
        try:
            for hidden, args, kwargs, masks in zip(hidden_states, layer_args, layer_kwargs, calibration_masks):
                current_inputs.clear()
                layer(
                    hidden.to(device),
                    *tensor_data_to_device(args, device),
                    **tensor_data_to_device(kwargs, device),
                )
                missing = group_by_capture.keys() - current_inputs.keys()
                if missing:
                    raise ValueError(f"MBQ layer replay did not execute {len(missing)} scale-group inputs.")
                for module, group in group_by_capture.items():
                    activations = current_inputs[module]
                    device_masks = masks.to(device)
                    validate_masks_for_activations(device_masks, activations)
                    samples[group.name].append((activations.detach().cpu(), masks))
        finally:
            for handle in handles:
                handle.remove()
        return samples

    @staticmethod
    @torch.no_grad()
    def _run_layer_samples(
        layer: torch.nn.Module,
        hidden_states: list[torch.Tensor],
        layer_args: list[tuple],
        layer_kwargs: list[dict],
        device: str,
    ) -> list[torch.Tensor]:
        outputs = []
        for hidden, args, kwargs in zip(hidden_states, layer_args, layer_kwargs):
            output = layer(
                hidden.to(device),
                *tensor_data_to_device(args, device),
                **tensor_data_to_device(kwargs, device),
            )
            if isinstance(output, tuple):
                output = output[0]
            outputs.append(output.detach().cpu())
        return outputs

    @staticmethod
    def _finalize_reweight_ratios(
        gradient_means: dict[tuple[int, str], _GradientMeans],
        num_layers: int,
    ) -> dict[tuple[int, str], float]:
        raw = {key: means.ratio() for key, means in gradient_means.items()}
        medians = {
            block_type: torch.tensor(
                [raw[(layer_idx, block_type)] for layer_idx in range(num_layers)], dtype=torch.float64
            )
            .median()
            .item()
            for block_type in ("attn", "mlp")
        }
        return {
            (layer_idx, block_type): max(raw[(layer_idx, block_type)], medians[block_type])
            for layer_idx in range(num_layers)
            for block_type in ("attn", "mlp")
        }

    @staticmethod
    @torch.no_grad()
    def _search_scale(
        linears: tuple[torch.nn.Linear, ...],
        samples: list[tuple[torch.Tensor, MultimodalCalibrationMasks]],
        vision_weight: float,
        quantizer: WeightQuantizer,
        n_grid: int,
        device: str,
    ) -> torch.Tensor:
        if not samples:
            raise ValueError("MBQ scale search received no activation samples.")
        in_features = linears[0].in_features
        if any(linear.in_features != in_features for linear in linears):
            raise ValueError("All MBQ scale-group linears must share in_features.")
        if quantizer.group_size > 0 and in_features % quantizer.group_size:
            raise ValueError(
                f"Linear in_features {in_features} must be divisible by MBQ group_size {quantizer.group_size}."
            )

        activation_sum = torch.zeros(in_features, dtype=torch.float64, device=device)
        activation_count = 0
        device_samples = []
        for sample_inputs, sample_masks in samples:
            device_inputs = sample_inputs.to(device)
            device_masks = sample_masks.to(device)
            activation_sum += device_inputs.detach().abs().reshape(-1, in_features).double().sum(dim=0)
            activation_count += device_inputs.numel() // in_features
            device_samples.append((device_inputs, device_masks))
        activation_mean = (activation_sum / activation_count).float().clamp(min=1e-4)

        best_loss = math.inf
        best_scale = None
        for grid_idx in range(n_grid):
            exponent = grid_idx / n_grid
            scale = activation_mean.pow(exponent).clamp(min=1e-4)
            scale = scale / torch.sqrt(scale.max() * scale.min())
            loss = 0.0
            for linear in linears:
                weight = linear.weight.detach()
                candidate_weight = quantizer.fake_quantize(weight * scale.unsqueeze(0)) / scale.unsqueeze(0)
                candidate_weight = candidate_weight.to(weight.dtype)
                for inputs, masks in device_samples:
                    reference = F.linear(inputs, weight, linear.bias)
                    candidate = F.linear(inputs, candidate_weight, linear.bias)
                    loss += modality_balanced_reconstruction_loss(
                        reference,
                        candidate,
                        masks,
                        vision_weight,
                    ).item()
            if loss < best_loss:
                best_loss = loss
                best_scale = scale.detach().clone()

        if best_scale is None:
            raise RuntimeError("MBQ scale search failed to select a scale.")
        return best_scale

    @staticmethod
    @torch.no_grad()
    def _apply_scale(
        previous: torch.nn.Module,
        linears: tuple[torch.nn.Linear, ...],
        scale: torch.Tensor,
    ) -> None:
        scale = scale.to(linears[0].weight.device, dtype=linears[0].weight.dtype)
        if isinstance(previous, torch.nn.Linear):
            width = scale.numel()
            if previous.out_features < width:
                raise ValueError("Previous linear does not have enough output channels for MBQ scale folding.")
            previous.weight[-width:].div_(scale.unsqueeze(1))
            if previous.bias is not None:
                previous.bias[-width:].div_(scale)
        elif hasattr(previous, "weight") and previous.weight.numel() == scale.numel():
            previous.weight.div_(scale.to(previous.weight.dtype))
            if getattr(previous, "bias", None) is not None:
                previous.bias.div_(scale.to(previous.bias.dtype))
        else:
            raise ValueError(f"Unsupported MBQ previous operator type {type(previous).__name__}.")

        for linear in linears:
            linear.weight.mul_(scale.unsqueeze(0))

    @staticmethod
    def _save_processor(model: HfModelHandler, output_model_path: str) -> None:
        from transformers import AutoProcessor

        trust_remote_code = model.get_load_kwargs().get("trust_remote_code", False)
        processor = AutoProcessor.from_pretrained(model.model_name_or_path, trust_remote_code=trust_remote_code)
        processor.save_pretrained(Path(output_model_path))
