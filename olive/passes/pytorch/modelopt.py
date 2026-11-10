# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Union

import torch

from olive.common.utils import tensor_data_to_device
from olive.data.config import DataConfig
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf
from olive.passes.pytorch.train_utils import get_calibration_dataset

logger = logging.getLogger(__name__)


class ModelOptQuantization(Pass):
    """Quantize Hugging Face PyTorch models using NVIDIA ModelOpt's native presets or configs."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "quantization_config": PassConfigParam(
                type_=Union[str, dict[str, Any]],
                default_value="fp8",
                description=(
                    "ModelOpt model preset name (e.g. fp8, int8_smoothquant, int4_awq, nvfp4, mxfp4, mxfp6, "
                    "mxfp8, mxint8) or a complete native QuantizeConfig dictionary. All installed ModelOpt "
                    "model presets are supported, including mixed-precision and weight-only variants."
                ),
            ),
            "data_config": PassConfigParam(
                type_=Union[DataConfig, dict],
                default_value=None,
                description=(
                    "Tokenized calibration data. Defaults to Olive's WikiText calibration data when the "
                    "quantization config requires calibration. Not loaded for calibration-free configs."
                ),
            ),
            "device": PassConfigParam(
                type_=str,
                default_value="cuda",
                description="PyTorch device used when input HfModel load_kwargs do not specify device_map.",
            ),
            "export_format": PassConfigParam(
                type_=Literal["hf", "modelopt"],
                default_value="hf",
                description=(
                    "'hf' exports packed deployment weights using ModelOpt's export_hf_checkpoint. "
                    "'modelopt' saves a Hugging Face checkpoint with ModelOpt quantizer state, supporting "
                    "all native quantization formats without requiring packed HF export support."
                ),
            ),
        }

    @classmethod
    def validate_config(cls, config: type[BasePassConfig], accelerator_spec: AcceleratorSpec) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False
        if not config.quantization_config or (
            isinstance(config.quantization_config, dict) and "quant_cfg" not in config.quantization_config
        ):
            logger.error("'quantization_config' must be a ModelOpt model preset name or a dictionary with 'quant_cfg'.")
            return False
        try:
            device = torch.device(config.device)
        except (RuntimeError, ValueError):
            logger.exception("'device' must be a valid PyTorch CPU or CUDA device.")
            return False
        if device.type not in ("cpu", "cuda"):
            logger.error("'device' must be a CPU or CUDA device.")
            return False
        return True

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        if not isinstance(model, HfModelHandler):
            raise ValueError("ModelOptQuantization requires an HfModel input.")
        if model.adapter_path:
            raise ValueError("Merge the Hugging Face model's adapters before ModelOpt quantization.")

        try:
            import modelopt.torch.opt as mto
            import modelopt.torch.quantization as mtq
            from modelopt.recipe.presets import QUANT_CFG_CHOICES
            from modelopt.torch.export import export_hf_checkpoint
            from modelopt.torch.quantization.config import need_calibration
        except ImportError as exc:
            raise ImportError(
                "ModelOptQuantization requires 'nvidia-modelopt[torch]>=0.48.0'. "
                "Install it with 'pip install \"nvidia-modelopt[torch]>=0.48.0\"'."
            ) from exc

        quant_config = config.quantization_config
        if isinstance(quant_config, str):
            if quant_config not in QUANT_CFG_CHOICES:
                raise ValueError(
                    f"Unknown ModelOpt quantization preset '{quant_config}'. "
                    f"Available presets: {', '.join(sorted(QUANT_CFG_CHOICES))}."
                )
            quant_config = QUANT_CFG_CHOICES[quant_config]
        quant_config = deepcopy(quant_config)

        device_map = model.get_load_kwargs().get("device_map")
        if device_map is None:
            device = torch.device(config.device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable. Use a CUDA-enabled environment or set 'device' to 'cpu'.")
        if config.export_format == "modelopt":
            mto.enable_huggingface_checkpointing()
        torch_model = model.load_model(cache_model=False).eval()
        if device_map is None:
            torch_model = torch_model.to(device)

        forward_loop = None
        if need_calibration(quant_config):
            dataset = get_calibration_dataset(model, config.data_config)
            input_device = torch_model.get_input_embeddings().weight.device
            if input_device.type == "meta":
                raise ValueError(
                    "Calibration requires materialized input embeddings. Update the HfModel's device_map "
                    "to keep input embeddings on a CPU or CUDA device instead of offloading them."
                )

            def calibrate(calibration_model):
                for batch in dataset:
                    calibration_model(**tensor_data_to_device(batch, input_device))

            forward_loop = calibrate

        logger.info("Quantizing Hugging Face model with ModelOpt: %s.", config.quantization_config)
        original_use_cache = getattr(torch_model.config, "use_cache", None)
        if original_use_cache is not None:
            torch_model.config.use_cache = False
        try:
            quantized_model = mtq.quantize(torch_model, quant_config, forward_loop=forward_loop)
        finally:
            if original_use_cache is not None:
                torch_model.config.use_cache = original_use_cache
        Path(output_model_path).mkdir(parents=True, exist_ok=True)
        if config.export_format == "hf":
            export_hf_checkpoint(quantized_model, export_dir=output_model_path)
        else:
            quantized_model.save_pretrained(output_model_path, save_modelopt_state=True)
        model.save_metadata(output_model_path)
        return inherit_hf_from_hf(model, output_model_path)
