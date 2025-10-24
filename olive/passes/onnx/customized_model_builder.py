# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Export a PyTorch model using the onnxruntime-genai package.
# --------------------------------------------------------------------------
import copy
import json
import logging
from enum import IntEnum
from pathlib import Path
from typing import Any, ClassVar, Union

import onnx
import transformers
from packaging import version

from olive.constants import Precision
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.hardware.constants import ExecutionProvider
from olive.model import CompositeModelHandler, HfModelHandler, ONNXModelHandler 
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.olive_pass import PassConfigParam
from olive.passes.pass_config import BasePassConfig

logger = logging.getLogger(__name__)

def precision_to_string(p: Precision):
    """Convert precision to string."""
    precision_mapping = {
        Precision.INT4: "int4",
        Precision.INT8: "int8",
        Precision.FP16: "fp16",
        Precision.FP32: "fp32",
    }
    return precision_mapping.get(p)


class CustomizedModelBuilder(Pass):
    """Converts a Huggingface generative PyTorch model to ONNX model using the customized ONNX conversion.

    See whisper for example, https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/models/whisper/README.md
    """

    EP_MAP: ClassVar[dict[ExecutionProvider, str]] = {
        ExecutionProvider.CPUExecutionProvider: "cpu",
        ExecutionProvider.CUDAExecutionProvider: "cuda",
    }

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "precision": PassConfigParam(
                type_=Precision,
                default_value=Precision.FP32,
                required=True,
                description="Precision of model.",
            ),
            "metadata_only": PassConfigParam(
                type_=bool,
                default_value=False,
                required=False,
                description="Whether to export the model or generate required metadata only.",
            ),
            "search": PassConfigParam(
                type_=dict[str, Any], required=False, description="Search options to use for generate loop."
            ),
            "int4_is_symmetric": PassConfigParam(
                type_=bool,
                required=False,
                description="Specify whether symmetric or asymmetric INT4 quantization needs to be used.",
            ),
            "include_hidden_states": PassConfigParam(
                type_=bool,
                required=False,
                description="Specify whether to have the hidden states as an output from your ONNX model.",
            ),
            "enable_cuda_graph": PassConfigParam(
                type_=bool,
                required=False,
                description=(
                    "The model can use CUDA graph capture for CUDA execution provider. "
                    "If enabled, all nodes being placed on the CUDA EP is the prerequisite "
                    "for the CUDA graph to be used correctly."
                ),
            ),
            "extra_options": PassConfigParam(
                type_=dict[str, Any],
                required=False,
                description="Extra key-value pairs options to pass to the model builder.",
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

        # if device is GPU, but user choose CPU EP, the is_cpu should be True
        if (config.precision == Precision.FP16) and not (
            accelerator_spec.accelerator_type == Device.GPU
            and accelerator_spec.execution_provider != ExecutionProvider.CPUExecutionProvider
        ):
            logger.info("FP16 is not supported on CPU.")
            return False

        if (
            config.precision == Precision.BF16
            and accelerator_spec.execution_provider != ExecutionProvider.CUDAExecutionProvider
        ):
            logger.info("BF16 is only supported on CUDA execution provider.")
            return False

        # Support for limited precision types
        return config.precision in {Precision.FP32, Precision.FP16, Precision.BF16, Precision.INT8, Precision.INT4}

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        return False

    def _run_for_config(
        self,
        model: Union[HfModelHandler, ONNXModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Union[ONNXModelHandler, CompositeModelHandler]:
        try:
            from onnxruntime.transformers.models.whisper.convert_to_onnx import main as run_whisper
        except ImportError:
            raise ImportError(
                "onnxruntime package is required to run CustomizedModelBuilder pass. Please install the package"
                " corresponding to your onnxruntime installation using pip. cpu: onnxruntime, cuda:"
                " onnxruntime-gpu"
            ) from None
        self.maybe_patch_quant()

        precision = config.precision
        metadata_only = config.metadata_only

        if metadata_only:
            if not isinstance(model, ONNXModelHandler):
                raise ValueError("metadata_only option is available only with ONNXModel as input.")
        elif not isinstance(model, HfModelHandler):
            raise ValueError("model building is available only with HfModel as input.")

        Path(output_model_path).mkdir(parents=True, exist_ok=True)
        output_model_filepath = (
            Path(resolve_onnx_path(output_model_path))
            if not metadata_only
            else Path(resolve_onnx_path(output_model_path, model.onnx_file_name))
        )

        target_execution_provider = self.EP_MAP.get(self.accelerator_spec.execution_provider, "cpu")

        extra_args = {"filename": str(output_model_filepath.name)}
        if metadata_only:
            extra_args["config_only"] = True
            model_path = None
            input_path = str(model.get_resource("model_path"))
        else:
            model_path = model.model_name_or_path
            # provide the model path as input path, model builder uses input_path for quantized models
            input_path = model_path
            if model.adapter_path:
                extra_args["adapter_path"] = model.adapter_path

        extra_args.update(
            {
                key: value.value if isinstance(value, IntEnum) else value
                for key, value in config.dict().items()
                if value is not None and key not in {"precision", "metadata_only", "search", "extra_options"}
            }
        )

        # Override extra options with user provided in extra_options parameter
        if config.extra_options:
            extra_args.update(config.extra_options)

        model_attributes = copy.deepcopy(model.model_attributes or {})
        print(f"model_attributes={model_attributes}")

        try:
            logger.debug("Building model with the following args: %s", extra_args)
            print(f"Building model with the following args: {extra_args}")
            model_name = model_path
            output_dir = str(output_model_filepath.parent)
            arguments = [
                "-m",
                model_name,
                "--output",
                output_dir,
                "--precision",
                precision_to_string(precision),
                "--provider",
                target_execution_provider,
                "--use_external_data_format",
                "--optimize_onnx",
                "--no_beam_search_op",
                "--output_cross_qk",
            ]
            if target_execution_provider == "cuda":
                arguments += ["--use_gpu"]
            if "int" in precision_to_string(precision):
                arguments += ["--quantize_symmetric"]

            print(f"output_model_filepath={output_model_filepath}", flush=True)
            run_whisper(arguments)
        except Exception:
            # if model building fails, clean up the intermediate files in the cache_dir
            cache_dir = Path(transformers.utils.TRANSFORMERS_CACHE)
            if cache_dir.is_dir():
                for file in cache_dir.iterdir():
                    if file.suffix == ".bin":
                        file.unlink()
            raise

        if metadata_only:
            output_model = copy.copy(model)
            output_model.model_attributes = model_attributes
        else:
            onnx_files = list(output_model_filepath.parent.glob("*.onnx"))
            onnx_file_names = [f.name for f in onnx_files]
            component_models = []
            component_names = []
            for name in onnx_file_names:
                component_models.append(ONNXModelHandler(
                    output_model_filepath.parent,
                    onnx_file_name=name,
                    model_attributes=model_attributes,
                ))
                component_names.append(name)
            output_model = CompositeModelHandler(
                component_models,
                component_names,
                model_path=output_model_filepath.parent,                
                model_attributes=model_attributes,
            )
        return output_model

    # TODO(jambayk): Remove this once version 0.9.1 with olive quant changes is released
    @staticmethod
    def maybe_patch_quant():
        """Patch onnxruntime-genai olive quant model to disable offset handling for qzeros in version 0.9.0+."""
        from onnxruntime_genai import __version__ as genai_version

        if version.parse(genai_version) < version.parse("0.9.0"):
            return

        from onnxruntime_genai.models.quantized_model import OliveModel

        if getattr(OliveModel.handle_qzeros, "__name__", "") == "_noop_handle_qzeros":
            return

        def _noop_handle_qzeros(self, module):
            pass

        OliveModel.handle_qzeros = _noop_handle_qzeros
