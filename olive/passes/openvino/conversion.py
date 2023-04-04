# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from olive.constants import Framework
from olive.model import ONNXModel, OpenVINOModel, PyTorchModel
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam


class OpenVINOConversion(Pass):
    """
    Converts PyTorch, ONNX or TensorFlow Model to OpenVino Model.
    """

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        return {
            "input": PassConfigParam(
                type_=List[Tuple],
                required=False,
                description=(
                    "Input can be set by passing a list of tuples. "
                    "Each tuple should contain input name and optionally input type or input shape."
                ),
            ),
            "input_shape": PassConfigParam(
                type_=List[int],
                required=False,
                description=(
                    "Input shape(s) that should be fed to an input node(s) of the model."
                    " Shape is defined as a comma-separated list of integer numbers"
                    " enclosed in parentheses or square brackets, for example [1,3,227,227]."
                ),
            ),
            "extra_config": PassConfigParam(
                type_=Dict,
                default_value=None,
                required=False,
                description=(
                    "Extra configurations for OpenVINO model conversion. extra_config can be set by"
                    " passing a dictionary where key is the parameter name, and the value is the parameter value."
                    " Please check 'mo' command usage instruction for available parameters:"
                    " https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html"
                ),
            ),
        }

    def _run_for_config(
        self, model: Union[PyTorchModel, ONNXModel], config: Dict[str, Any], output_model_path: str
    ) -> OpenVINOModel:
        import torch

        try:
            from openvino.runtime import serialize
            from openvino.tools.mo import convert_model
        except ImportError:
            raise ImportError("Please install olive[openvino] to use OpenVINO model")

        model_name = model.name if model.name else "ov_model"

        model_input_param = "input_model"
        input_model = model.model_path
        if model.framework == Framework.TENSORFLOW and not model.is_file:
            model_input_param = "saved_model_dir"
        if model.framework == Framework.PYTORCH:
            input_model = model.load_model()

        extra_config = config["extra_config"] or {}
        if not extra_config.get("example_input") and config.get("input_shape"):
            extra_config["example_input"] = torch.randn(config["input_shape"])
        args = {model_input_param: input_model, "model_name": model_name, **extra_config}

        # OpenVINO Python API document: https://docs.openvino.ai/latest/openvino_docs_MO_DG_Python_API.html
        ov_model = convert_model(**args)
        output_dir = Path(output_model_path) / model_name

        # Save as ov model
        serialize(ov_model, xml_path=str(output_dir.with_suffix(".xml")), bin_path=str(output_dir.with_suffix(".bin")))
        openvino_model = OpenVINOModel(model_path=output_model_path, name=model_name)

        return openvino_model
