# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Export a PyTorch model using the onnxruntime-genai package.
# --------------------------------------------------------------------------
import logging
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict

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

        def __str__(self) -> str:
            return self.value

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "precision": PassConfigParam(
                type_=GenAIModelExporter.Precision,
                required=True,
                description="Precision of model.",
            )
        }

    def validate_search_point(
        self, search_point: Dict[str, Any], accelerator_spec: AcceleratorSpec, with_fixed_value: bool = False
    ) -> bool:
        if with_fixed_value:
            search_point = self.config_at_search_point(search_point or {})
        precision = search_point.get("precision")
        device = (
            Device.CPU
            if self.accelerator_spec.execution_provider
            in AcceleratorLookup.get_execution_providers_for_device(Device.CPU)
            else Device.GPU
        )
        if precision == GenAIModelExporter.Precision.FP16 and device == Device.CPU:
            logger.info(
                "FP16 is not supported on CPU. Valid precision + execution"
                "provider combinations are: FP32 CPU, FP32 CUDA, FP16 CUDA, INT4 CPU, INT4 CUDA"
            )
            return False
        return True

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        return False

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
        target_execution_provider = (
            "cpu"
            if self.accelerator_spec.execution_provider
            in AcceleratorLookup.get_execution_providers_for_device(Device.CPU)
            else "cuda"
        )
        # Select cache location based on priority
        # HF_CACHE (HF >= v5) -> TRANSFORMERS_CACHE (HF < v5) -> local dir
        cache_dir = os.environ.get("HF_HOME", None)
        if not cache_dir:
            cache_dir = os.environ.get("TRANSFORMERS_CACHE", None)
        if not cache_dir:
            # please do not clean up the cache dir as it contains huggingface model
            # snapshots that can be reused for future runs
            cache_dir = str(Path(tempfile.gettempdir()) / "hf_cache")

        # currently we only support regular hf models so we can pass the name_or_path directly to model_name
        # could also check if it is a locally saved model and pass the path to input_path but it is not necessary
        create_model(
            model_name=str(model.model_path or model.hf_config.model_name),
            input_path="",  # empty string for now
            output_dir=str(output_model_filepath.parent),
            precision=precision,
            execution_provider=target_execution_provider,
            cache_dir=cache_dir,
            filename=str(output_model_filepath.name),
        )

        return ONNXModelHandler(output_model_filepath.parent, onnx_file_name=output_model_filepath.name)
