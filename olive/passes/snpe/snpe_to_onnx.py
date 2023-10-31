# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Callable, Dict

from pydantic import validator

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModel, SNPEModel
from olive.passes.olive_pass import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam
from olive.snpe import SNPEDevice
from olive.snpe.tools.dev import dlc_to_onnx


def _validate_target_device(v):
    valid_devices = [d.value for d in SNPEDevice]
    if v not in valid_devices:
        raise ValueError(f"Invalid target device: {v}. Valid values are: {valid_devices}")
    return v


class SNPEtoONNXConversion(Pass):
    """Convert a SNPE DLC to ONNX to use with SNPE Execution Provider.

    Creates a ONNX graph with the SNPE DLC as a node.
    """

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "target_device": PassConfigParam(
                type_=str,
                default_value="cpu",
                description="Target device for the ONNX model. Refer to olive.snpe.SNPEDevice for valid values.",
            ),
            "target_opset": PassConfigParam(type_=int, default_value=12, description="Target ONNX opset version."),
        }
        config.update(get_external_data_config())
        return config

    @staticmethod
    def _validators() -> Dict[str, Callable]:
        return {
            "validate_target_device": validator("target_device", allow_reuse=True)(_validate_target_device),
        }

    def _run_for_config(
        self, model: SNPEModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModel:
        config = self._config_class(**config)

        output_model_path = ONNXModel.resolve_path(output_model_path)

        # create a onnx model that wraps the dlc binary in a node
        onnx_model = dlc_to_onnx(model.model_path, config.dict(), **model.io_config)

        # save the model to the output path and return the model
        return model_proto_to_olive_model(onnx_model, output_model_path, config.dict())
