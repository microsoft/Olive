# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from typing import Any, Callable, Dict, List, Type

from olive.common.pydantic_v1 import root_validator
from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes.olive_pass import Pass
from olive.passes.onnx.common import (
    fix_dim_params,
    fix_input_shapes,
    get_external_data_config,
    model_proto_to_olive_model,
)
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class DynamicToFixedShape(Pass):
    """Convert dynamic shape to fixed shape for ONNX model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "dim_param": PassConfigParam(
                type_=List[str],
                default_value=None,
                required=False,
                description="Symbolic parameter name. Provide dim_value if specified.",
            ),
            "dim_value": PassConfigParam(
                type_=List[int],
                default_value=None,
                required=False,
                description="Value to replace dim_param with in the model. Must be > 0.",
            ),
            "input_name": PassConfigParam(
                type_=List[str],
                default_value=None,
                required=False,
                description="Model input name to replace shape of. Provide input_shape if specified.",
            ),
            "input_shape": PassConfigParam(
                type_=List[List[int]],
                default_value=None,
                required=False,
                description=(
                    "Shape to use for input_shape. Provide comma separated list for the shape. "
                    "All values must be > 0. e.g. [1,3,256,256]"
                ),
            ),
        }
        config.update(get_external_data_config())
        return config

    @classmethod
    def _validators(cls) -> Dict[str, Callable[..., Any]]:
        return {
            "validate_configs": root_validator(allow_reuse=True)(_jointly_validate_configs),
        }

    def _run_for_config(
        self,
        model: ONNXModelHandler,
        config: Type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:

        onnx_model = model.load_model()
        output_model_path = resolve_onnx_path(output_model_path)

        if config.dim_param:
            fix_dim_params(onnx_model, config.dim_param, config.dim_value)
        elif config.input_name:
            fix_input_shapes(onnx_model, config.input_name, config.input_shape)

        return model_proto_to_olive_model(onnx_model, output_model_path, config)


def _jointly_validate_configs(cls, values):
    if values.get("input_name") and values.get("dim_param"):
        raise ValueError("Cannot set both dim_param and input_name at the same time.")
    if not values.get("input_name") and not values.get("dim_param"):
        raise ValueError("dim_param and input_name cannot be both empty.")

    # cannot use if values["dim_param"] ^ values["dim_value"] because the value could be list
    # and list cannot be used in xor operation
    if (not values["dim_param"]) ^ (not values["dim_value"]):
        raise ValueError("dim_param and dim_value must be both provided or both None.")
    if (not values["input_name"]) ^ (not values["input_shape"]):
        raise ValueError("input_name and input_shape must be both provided or both None.")

    if values["dim_param"] and values["dim_value"]:
        if len(values["dim_param"]) != len(values["dim_value"]):
            raise ValueError("dim_param and dim_value must have the same number of elements.")
        if any(i < 0 for i in values["dim_value"]):
            raise ValueError("dim_value must be all >= 0 when dim_param is provided.")

    if values["input_name"] and values["input_shape"]:
        if len(values["input_name"]) != len(values["input_shape"]):
            raise ValueError("input_name and input_shape must have the same number of elements.")
        if any(any(i <= 0 for i in shape) for shape in values["input_shape"]):
            raise ValueError("input_shape must be all > 0 when input_name is provided.")
    return values
