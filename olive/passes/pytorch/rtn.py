# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from dataclasses import dataclass
from typing import Optional

import torch

from olive.common.hf.wrapper import ModelWrapper
from olive.common.quant.hf_utils import OliveHfQuantizationConfig, replace_matching_submodules
from olive.common.quant.linear import QuantLinear
from olive.common.quant.utils import WeightQuantizer
from olive.constants import PrecisionBits
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf
from olive.passes.pytorch.train_utils import (
    load_hf_base_model,
)

logger = logging.getLogger(__name__)

# ruff: noqa: N806


class Rtn(Pass):
    """Round-to-nearest quantization."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "bits": PassConfigParam(
                type_=PrecisionBits,
                default_value=PrecisionBits.BITS4,
                description="quantization bits. Default value is 4",
            ),
            "group_size": PassConfigParam(
                type_=int,
                default_value=128,
                description="Block size for quantization. Default value is 128.",
            ),
            "sym": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Symmetric quantization. Default value is False.",
            ),
        }

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        """Run GPTQ quantization on the model.

        Args:
            model: The HuggingFace model to quantize.
            config: Configuration object containing quantization parameters.
            output_model_path: Path where the quantized model will be saved.

        Returns:
            HfModelHandler for the quantized model.

        """
        wrapper = ModelWrapper.from_model(load_hf_base_model(model, torch_dtype="auto"))
        wrapper.model.eval()

        quant_config = self.get_quant_config(model, config)

        self.prepare_model(wrapper, quant_config)

        # get the inputs for the first layer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # finalize the quantization
        self.finalize(wrapper, quant_config, device)

        # save the quantized model
        wrapper.model.save_pretrained(output_model_path)
        model.save_metadata(output_model_path)

        return inherit_hf_from_hf(model, output_model_path, adapter_path=model.adapter_path)

    def get_quant_config(self, model: HfModelHandler, config: type[BasePassConfig]) -> OliveHfQuantizationConfig:
        """Get quantization configuration with mixed precision support.

        Args:
            model: The HuggingFace model to get configuration for.
            config: Configuration object containing quantization parameters.

        Returns:
            OliveHfQuantizationConfig object with quantization settings.

        """
        quant_config = {
            "bits": config.bits,
            "symmetric": config.sym,
            "group_size": config.group_size,
        }
        if mp_info := (model.model_attributes or {}).get("mixed_precision_info"):
            for k, v in quant_config.items():
                if mp_info["default"].get(k) is not None and v != mp_info["default"][k]:
                    logger.debug("Overriding %s with mixed precision info: %s", k, mp_info["default"][k])
                    quant_config[k] = mp_info["default"][k]
            quant_config["overrides"] = mp_info.get("overrides")
        return OliveHfQuantizationConfig(**quant_config)

    def prepare_model(self, wrapper: ModelWrapper, quant_config: OliveHfQuantizationConfig) -> None:
        """Prepare the model for quantization by adding quant_info to linear layers.

        Args:
            wrapper: ModelWrapper containing the model to prepare.
            quant_config: Quantization configuration to use.

        """
        # TODO(jambayk): make lm head quantization configurable
        lm_head_name = wrapper.get_lm_head()[1]

        def should_quantize(module: torch.nn.Module, name: str) -> bool:
            return isinstance(module, torch.nn.Linear) and name != lm_head_name

        def add_quant_info(module: torch.nn.Module, name: str) -> torch.nn.Module:
            # TODO(jambayk): validate that the module and config are compatible
            module.quant_info = QuantInfo(quantizer=WeightQuantizer(**quant_config.get_qlinear_init_args(name)))
            return module

        replace_matching_submodules(
            wrapper.model,
            should_quantize,
            add_quant_info,
            description="Preparing model for quantization",
        )

    def finalize(self, wrapper: ModelWrapper, quant_config: OliveHfQuantizationConfig, device: str) -> None:
        """Finalize quantization by replacing linear layers with quantized versions.

        Args:
            wrapper: ModelWrapper containing the model to finalize.
            quant_config: Quantization configuration to use.
            device: Device to perform quantization on.

        """

        def should_quantize(module: torch.nn.Module, _: str) -> bool:
            return hasattr(module, "quant_info")

        def quantize_and_pack(module: torch.nn.Module, _: str) -> QuantLinear:
            module.to(device)
            return QuantLinear.from_linear(
                module.to(device),
                bits=module.quant_info.quantizer.bits,
                symmetric=module.quant_info.quantizer.symmetric,
                group_size=module.quant_info.quantizer.group_size,
            ).to("cpu")  # move the original module to CPU

        replace_matching_submodules(
            wrapper.model,
            should_quantize,
            quantize_and_pack,
            description="Quantizing and packing linear layers",
        )

        wrapper.model.quantization_method = quant_config.quant_method
        wrapper.model.config.quantization_config = quant_config


@dataclass
class QuantInfo:
    """Class to hold quantization information for GPTQ.

    This class stores all the necessary information for quantizing a layer,
    including the quantizer, computed scales and zero points, and calibration data.

    Attributes:
        quantizer: The weight quantizer used for quantization.
        scales: Computed scales for quantization. Set after processing.
        zero_points: Computed zero points for quantization. Set after processing.
        data: Calibration data including Hessian matrix and sample count.
              Format: {"H": torch.Tensor, "N": int} or None.

    """

    quantizer: WeightQuantizer
    scales: Optional[torch.Tensor] = None
    zero_points: Optional[torch.Tensor] = None
    data: Optional[dict] = None
