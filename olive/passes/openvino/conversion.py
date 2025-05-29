# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Callable, Union

import torch

from olive.constants import Framework
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, ONNXModelHandler, OpenVINOModelHandler, PyTorchModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam, get_user_script_data_config


class OpenVINOConversion(Pass):
    """Converts PyTorch, ONNX or TensorFlow Model to OpenVino Model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            "input_shapes": PassConfigParam(
                type_=Union[Callable, str, list],
                required=False,
                description=(
                    "Set or override shapes for model inputs."
                    "It configures dynamic and static dimensions in model inputs"
                    "depending on your inference requirements."
                    "Static parameter is required if static models are required."
                ),
            ),
            "example_input_func": PassConfigParam(
                type_=Union[Callable, str],
                required=False,
                description=(
                    "Function/function name to generate sample of model input in original framework."
                    "For PyTorch it can be torch.Tensor."
                    "For Tensorflow it can be tf.Tensor or numpy.ndarray."
                    "By default a pytorch float tensor is created."
                ),
            ),
            "compress_to_fp16": PassConfigParam(
                type_=bool,
                default_value=True,
                required=False,
                description="Compress weights in output OpenVINO model to FP16. Default is True.",
            ),
            "extra_configs": PassConfigParam(
                type_=dict,
                default_value=None,
                required=False,
                description=(
                    "Extra configurations for OpenVINO model conversion. extra_config can be set by "
                    "passing a dictionary where key is the parameter name, and the value is the parameter value. "
                    "Please check Conversion Parameters documentation for more details: "
                    "https://docs.openvino.ai/2023.3/openvino_docs_OV_Converter_UG_Conversion_Options.html"
                ),
            ),
            "model_name": PassConfigParam(
                type_=str,
                default_value="ov_model",
                required=False,
                description=("Name of output openVINO model."),
            ),
            "static": PassConfigParam(
                type_=bool,
                default_value=True,
                required=False,
                description=("Create a static model instead of a dynamic model.Enabled by default."),
            ),
        }

    def _run_for_config(
        self,
        model: Union[HfModelHandler, PyTorchModelHandler, ONNXModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> OpenVINOModelHandler:
        try:
            import openvino as ov
        except ImportError:
            raise ImportError("Please install olive-ai[openvino] to use OpenVINO model") from None

        if model.framework == Framework.SNPE:
            raise ValueError("OpenVINO conversion is not supported for SNPE model.")

        input_model = model.model_path
        if model.framework == Framework.PYTORCH:
            input_model = model.load_model()

        example_input = []
        if config.example_input_func:
            example_input = self._user_module_loader.call_object(config.example_input_func)
        elif config.input_shapes and model.framework == Framework.PYTORCH and isinstance(config.input_shapes, list):
            for i in config.input_shapes:
                example_input.append(torch.rand(i))
        else:
            example_input = None

        input_shapes = None
        if config.input_shapes:
            config_input = config.input_shapes
            if isinstance(config_input, list):
                input_shapes = config_input
            else:
                input_shapes = self._user_module_loader.call_object(config_input)

        extra_configs = config.extra_configs or {}
        args = {
            "input_model": input_model,
            "example_input": example_input,
            **extra_configs,
        }

        if config.static:
            args["input"] = input_shapes

        try:
            ov_model = ov.convert_model(**args)
        except Exception:
            msg = "Invalid config file. Please recheck parameters"
            raise ValueError(msg) from None

        output_dir = Path(output_model_path) / config.model_name

        # Save as ov model
        ov.save_model(ov_model, output_model=output_dir.with_suffix(".xml"), compress_to_fp16=config.compress_to_fp16)
        return OpenVINOModelHandler(model_path=output_model_path)
