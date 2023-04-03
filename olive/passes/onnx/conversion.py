# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import torch

from olive.model import ONNXModel, PyTorchModel
from olive.passes import Pass
from olive.passes.pass_config import PassConfigParam


class TraceModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *input_data, **input_dict):
        if isinstance(self.model(*input_data, **input_dict), dict):
            return [val for val in self.model(*input_data, **input_dict).values()]
        return self.model(*input_data, **input_dict)


class OnnxConversion(Pass):
    """Convert a PyTorch model to ONNX model using torch.onnx.export."""

    _requires_user_script = True

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        return {
            "input_names": PassConfigParam(type_=List[str], required=True, description="List of input names."),
            # required for if input_tensors_func is not provided
            "input_shapes": PassConfigParam(
                type_=List[List[int]],
                default_value=None,
                description=(
                    "List of input shapes. Must be provided if input_tensor_func is not provided. It is used to create"
                    " dummy inputs for the model during onnx export."
                ),
            ),
            "input_types": PassConfigParam(
                type_=List[str],
                default_value=None,
                description=(
                    "List of input types. If provided, must be the same length as input_shapes. Otherwise, defaults to"
                    " float32 for all inputs. Used with input_shapes to create dummy inputs for the model during onnx"
                    " export."
                ),
            ),
            "input_tensor_func": PassConfigParam(
                type_=Union[Callable, str],
                default_value=None,
                is_object=True,
                description=(
                    "Function (no input) to create dummy inputs for the model. Can be a function (local use) or name of"
                    " a function to be imported from user script. If provided, input_shapes and input_types will be"
                    " ignored. Refer to 'args' at https://pytorch.org/docs/stable/onnx.html#torch.onnx.export for more"
                    " details."
                ),
            ),
            "output_names": PassConfigParam(type_=List[str], required=True, description="List of output names."),
            "dynamic_axes": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "Dynamic axes for the model. Refer to 'dynamic_axes' at"
                    " https://pytorch.org/docs/stable/onnx.html#torch.onnx.export for more details."
                ),
            ),
            "target_opset": PassConfigParam(
                type_=int, default_value=14, description="The version of the default (ai.onnx) opset to target."
            ),
        }

    def _initialize(self):
        # input shapes
        self._fixed_params["input_shapes"] = self._fixed_params["input_shapes"] or []

        # input types
        str_to_type = {"float32": torch.float32, "float16": torch.float16, "int32": torch.int32, "int64": torch.int64}
        input_types = []
        if self._fixed_params["input_types"] is not None:
            for input_type in self._fixed_params["input_types"]:
                input_types.append(str_to_type[input_type])
        else:
            input_types = [str_to_type["float32"] for _ in self._fixed_params["input_shapes"]]

        assert not (
            self._fixed_params["input_tensor_func"] and self._fixed_params["input_shapes"]
        ), "Either input_tensor_func or input_shapes must be provided."

        # dummy inputs
        self._dummy_inputs = []
        if self._fixed_params["input_tensor_func"] is not None:
            self._dummy_inputs = self._user_module_loader.call_object(self._fixed_params["input_tensor_func"])
        else:
            for input_shape, input_type in zip(self._fixed_params["input_shapes"], input_types):
                self._dummy_inputs.append(torch.zeros(input_shape, dtype=input_type))
            self._dummy_inputs = tuple(self._dummy_inputs) if len(self._dummy_inputs) > 1 else self._dummy_inputs[0]

        # dynamic axes
        self._dynamic_axes = {}
        if self._fixed_params["dynamic_axes"] is not None:
            for name in self._fixed_params["dynamic_axes"]:
                self._dynamic_axes[name] = {
                    int(key): value for key, value in self._fixed_params["dynamic_axes"][name].items()
                }
        else:
            self._dynamic_axes = None

    def _run_for_config(self, model: PyTorchModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        pytorch_model = model.load_model()
        pytorch_model.eval()
        if isinstance(pytorch_model, torch.jit.RecursiveScriptModule):
            pytorch_model = TraceModelWrapper(pytorch_model)

        if Path(output_model_path).suffix != ".onnx":
            output_model_path += ".onnx"

        torch.onnx.export(
            pytorch_model,
            # TODO: support custom input tensor. Will implement once we decide how to handle non-hashable config params
            self._dummy_inputs,
            output_model_path,
            export_params=True,
            opset_version=config["target_opset"],
            input_names=config["input_names"],
            output_names=config["output_names"],
            dynamic_axes=self._dynamic_axes,
        )

        model = ONNXModel(output_model_path, model.name)
        return model
