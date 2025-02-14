# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Callable, Dict, List, Type, Union

from olive.constants import Framework
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import HfModelHandler, ONNXModelHandler, OpenVINOModelHandler, PyTorchModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam, get_user_script_data_config


class OpenVINOConversion(Pass):
    """Converts PyTorch, ONNX or TensorFlow Model to OpenVino Model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            **get_user_script_data_config(),
            "input": PassConfigParam(
                type_=Union[Callable, str, List],
                required=False,
                description=(
                    "Set or override shapes for model inputs. "
                    "It configures dynamic and static dimensions in model inputs "
                    "depending on your inference requirements."
                ),
            ),
            "example_input_func": PassConfigParam(
                type_=Union[Callable, str],
                required=False,
                description=(
                    "Function/function name to generate sample of model input in original framework. "
                    "For PyTorch it can be torch.Tensor. "
                    "For Tensorflow it can be tf.Tensor or numpy.ndarray. "
                ),
            ),
            "compress_to_fp16": PassConfigParam(
                type_=bool,
                default_value=True,
                required=False,
                description="Compress weights in output OpenVINO model to FP16. Default is True.",
            ),
            "extra_configs": PassConfigParam(
                type_=Dict,
                default_value=None,
                required=False,
                description=(
                    "Extra configurations for OpenVINO model conversion. extra_config can be set by "
                    "passing a dictionary where key is the parameter name, and the value is the parameter value. "
                    "Please check Conversion Parameters documentation for more details: "
                    "https://docs.openvino.ai/2023.3/openvino_docs_OV_Converter_UG_Conversion_Options.html"
                ),
            ),
            "output_model": PassConfigParam(
                type_=str,
                default_value="ov_model",
                required=False,
                description="Name of the output OpenVINO model.",
            ),
        }

    def _run_for_config(
        self,
        model: Union[HfModelHandler, PyTorchModelHandler, ONNXModelHandler],
        config: Type[BasePassConfig],
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

        example_input = None
        if config.example_input_func:
            example_input = self._user_module_loader.call_object(config.example_input_func)

        input_shape = None
        if config.input:
            config_input = config.input
            if isinstance(config_input, List):
                input_shape = config_input
            else:
                input_shape = self._user_module_loader.call_object(config_input)

        extra_configs = config.extra_configs or {}
        args = {
            "input_model": input_model,
            "input": input_shape,
            "example_input": example_input,
            **extra_configs,
        }

        ov_model = ov.convert_model(**args)

        model_name = "ov_model"
        output_dir = Path(output_model_path) / (config.output_model or model_name)

        # Save as ov model
        ov.save_model(ov_model, output_model=output_dir.with_suffix(".xml"), compress_to_fp16=config.compress_to_fp16)
        return OpenVINOModelHandler(model_path=output_model_path)
