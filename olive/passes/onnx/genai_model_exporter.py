# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Export a PyTorch model using the onnxruntime-genai package.
# --------------------------------------------------------------------------
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict

from onnx import TensorProto

from olive.hardware.accelerator import AcceleratorLookup, AcceleratorSpec, Device
from olive.model import ONNXModelHandler, PyTorchModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam

logger = logging.getLogger(__name__)


class GenAIModelExporter(Pass):
    """Converts a Huggingface generative PyTorch model to ONNX model using the Generative AI builder.

    See https://github.com/microsoft/onnxruntime-genai
    """

    class Precision(str, Enum):
        FP32 = "fp32"
        FP16 = "fp16"
        INT4 = "int4"

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "precision": PassConfigParam(
                type_=GenAIModelExporter.Precision,
                required=True,
                description="Precision of model.",
            )
        }

    def _run_for_config(
        self, model: PyTorchModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        from onnxruntime_genai.models.builder import create_model

        if not model.hf_config:
            raise ValueError(
                "GenAIModelExporter pass only supports exporting HF models i.e. PyTorchModelHandler with hf_config."
            )

        Path(output_model_path).mkdir(parents=True, exist_ok=True)
        output_model_filepath = Path(resolve_onnx_path(output_model_path))

        precision = config["precision"]
        device = (
            Device.CPU
            if self.accelerator_spec.execution_provider
            in AcceleratorLookup.get_execution_providers_for_device(Device.CPU)
            else Device.GPU
        )

        logger.info(
            "Valid precision + execution provider combinations are: FP32 CPU, FP32 CUDA, FP16 CUDA, INT4 CPU, INT4 CUDA"
        )

        # Set input/output precision of ONNX model
        io_dtype = (
            TensorProto.FLOAT
            if precision in {"int8", "fp32"} or (precision == "int4" and device == Device.CPU)
            else TensorProto.FLOAT16
        )

        # Select cache location based on priority
        # HF_CACHE (HF >= v5) -> TRANSFORMERS_CACHE (HF < v5) -> local dir
        cache_dir = os.environ.get("HF_HOME", None)
        if not cache_dir:
            cache_dir = os.environ.get("TRANSFORMERS_CACHE", None)
        if not cache_dir:
            cache_dir = output_model_filepath.parent / "genai_cache_dir"

        create_model(
            model_name_or_path=str(model.hf_config.model_name),
            output_dir=str(output_model_filepath.parent),
            precision=str(precision),
            execution_provider=str(device),
            cache_dir=str(cache_dir),
            io_dtype=io_dtype,
            filename=str(output_model_filepath.name),
        )

        return ONNXModelHandler(output_model_filepath.parent, onnx_file_name=output_model_filepath.name)
