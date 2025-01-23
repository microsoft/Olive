# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Callable, Dict, List, Type, Union

from olive.common.pydantic_v1 import validator
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler, SNPEModelHandler, TensorFlowModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.platform_sdk.qualcomm.constants import InputLayout, InputType
from olive.platform_sdk.qualcomm.snpe.tools.dev import get_dlc_io_config, to_dlc
from olive.resource_path import LocalFile


def _validate_input_types_layouts(v, values, field):
    if v is None:
        return v

    if "input_names" not in values:
        raise ValueError("Invalid input_names")
    if len(v) != len(values["input_names"]):
        raise ValueError(f"{field.name} must be the same length as input_names")
    for value in v:
        # input_types: None and "default" are the same
        # input_layouts: If None, it will use the default based on the Source Framework,
        # shape of input and input encoding
        valid_values = [None]
        valid_values += (
            [input_type.value for input_type in InputType]
            if field.name == "input_types"
            else [layout.value for layout in InputLayout]
        )
        if value not in valid_values:
            raise ValueError(f"Invalid value: {value}. Valid values are: {valid_values}")
    return v


class SNPEConversion(Pass):
    """Convert ONNX or TensorFlow model to SNPE DLC.

    Uses snpe-tensorflow-to-dlc or snpe-onnx-to-dlc tools from the SNPE SDK.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "input_names": PassConfigParam(type_=List[str], required=True, description="List of input names."),
            "input_shapes": PassConfigParam(
                type_=List[List[int]],
                required=True,
                description="List of input shapes. Must be the same length as input_names.",
            ),
            "output_names": PassConfigParam(type_=List[str], required=True, description="List of output names."),
            "input_types": PassConfigParam(
                type_=List[Union[str, None]],
                default_value=None,
                description=(
                    "List of input types. If not None, it must be a list of the same length as input_names. List"
                    " members can be None to use default value. Refer to"
                    " olive.platform_sdk.qualcomm.constants.InputType for valid values."
                ),
            ),
            "input_layouts": PassConfigParam(
                type_=List[Union[str, None]],
                default_value=None,
                description=(
                    "List of input layouts. If not None, it must be a list of the same length as input_names. List"
                    " members can be None to use inferred value."
                    " Refer to olive.platform_sdk.qualcomm.constants.InputLayout for valid values."
                ),
            ),
            "extra_args": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Extra arguments to pass to snpe conversion tool. Refer to snpe-onnx-to-dlc and"
                    " snpe-tensorflow-to-dlc at https://developer.qualcomm.com/sites/default/files/docs/snpe/tools.html"
                    " for more additional arguments. The value is a string that will be passed as is to the tool."
                    " e.g.: --enable_cpu_fallback --priority_hint low"
                ),
            ),
        }

    @classmethod
    def _validators(cls) -> Dict[str, Callable]:
        return {
            "validate_input_types_layouts": validator("input_types", "input_layouts", allow_reuse=True)(
                _validate_input_types_layouts
            )
        }

    def _run_for_config(
        self,
        model: Union[ONNXModelHandler, TensorFlowModelHandler],
        config: Type[BasePassConfig],
        output_model_path: str,
    ) -> SNPEModelHandler:
        if Path(output_model_path).suffix != ".dlc":
            output_model_path += ".dlc"

        to_dlc(model.model_path, model.framework, config, output_model_path)
        io_config = get_dlc_io_config(output_model_path, config.input_names, config.output_names)
        return SNPEModelHandler(model_path=LocalFile({"path": output_model_path}), **io_config)
