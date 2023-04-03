# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Callable, Dict

from pydantic import validator

from olive.model import ONNXModel, SNPEModel
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import PassConfigParam
from olive.snpe import SNPEDevice
from olive.snpe.tools.dev import dlc_to_onnx


def _validate_target_device(v):
    valid_devices = [d.value for d in SNPEDevice]
    if v not in valid_devices:
        raise ValueError(f"Invalid target device: {v}. Valid values are: {valid_devices}")
    return v


class SNPEtoONNXConversion(Pass):
    """
    Convert a SNPE DLC to ONNX to use with SNPE Execution Provider.
    Creates a ONNX graph with the SNPE DLC as a node.
    """

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        return {
            "target_device": PassConfigParam(
                type_=str,
                default_value="cpu",
                description="Target device for the ONNX model. Refer to olive.snpe.SNPEDevice for valid values.",
            ),
            "target_opset": PassConfigParam(type_=int, default_value=12, description="Target ONNX opset version."),
        }

    @staticmethod
    def _validators() -> Dict[str, Callable]:
        return {
            "validate_target_device": validator("target_device", allow_reuse=True)(_validate_target_device),
        }

    def _run_for_config(self, model: SNPEModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        config = self._config_class(**config)

        if Path(output_model_path).suffix != ".onnx":
            output_model_path += ".onnx"

        dlc_to_onnx(model.model_path, config.dict(), output_model_path, **model.io_config)
        return ONNXModel(output_model_path, name=model.name)
