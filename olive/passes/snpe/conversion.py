# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from pydantic import validator

from olive.model import ONNXModel, SNPEModel, TensorFlowModel
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import PassConfigParam
from olive.snpe.constants import InputLayout, InputType
from olive.snpe.tools.dev import get_dlc_io_config, to_dlc


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
            [type.value for type in InputType]
            if field.name == "input_types"
            else [layout.value for layout in InputLayout]
        )
        if value not in valid_values:
            raise ValueError(f"Invalid value: {value}. Valid values are: {valid_values}")
    return v


class SNPEConversion(Pass):
    """
    Convert ONNX or TensorFlow model to SNPE DLC.
    Uses snpe-tensorflow-to-dlc or snpe-onnx-to-dlc tools from the SNPE SDK.
    """

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        return {
            "input_names": PassConfigParam(type_=List[str], required=True, description="List of input names."),
            "input_shapes": PassConfigParam(
                type_=List[List[int]],
                required=True,
                description="List of input shapes. Must be the same length as input_names.",
            ),
            "output_names": PassConfigParam(type_=List[str], required=True, description="List of output names."),
            "output_shapes": PassConfigParam(
                type_=List[List[int]],
                required=True,
                description="List of output shapes. Must be the same length as output_names.",
            ),
            "input_types": PassConfigParam(
                type_=List[Union[str, None]],
                default_value=None,
                description=(
                    "List of input types. If not None, it must be a list of the same length as input_names. List"
                    " members can be None to use default value. Refer to olive.snpe.constants.InputType for valid"
                    " values."
                ),
            ),
            "input_layouts": PassConfigParam(
                type_=List[Union[str, None]],
                default_value=None,
                description=(
                    "List of input layouts. If not None, it must be a list of the same length as input_names. List"
                    " members can be None to use infered value. Refer to olive.snpe.constants.InputLayout for valid"
                    " values."
                ),
            ),
            "extra_args": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Extra arguments to pass to snpe conversion tool. Refer to snpe-onnx-to-dlc and"
                    " snpe-tensorflow-to-dlc at https://developer.qualcomm.com/sites/default/files/docs/snpe/tools.html"
                    " for more additional arguments. Must be a dictionary of the form: {'arg_name': 'arg_value'}."
                ),
            ),
        }

    @staticmethod
    def _validators() -> Dict[str, Callable]:
        return {
            "validate_input_types_layouts": validator("input_types", "input_layouts", allow_reuse=True)(
                _validate_input_types_layouts
            )
        }

    def _run_for_config(
        self, model: Union[ONNXModel, TensorFlowModel], config: Dict[str, Any], output_model_path: str
    ) -> SNPEModel:
        config = self._config_class(**config)

        if Path(output_model_path).suffix != ".dlc":
            output_model_path += ".dlc"

        to_dlc(model.model_path, model.framework, config.dict(), output_model_path)
        io_config = get_dlc_io_config(output_model_path, config.input_names, config.output_names)
        return SNPEModel(model_path=output_model_path, name=model.name, **io_config)
