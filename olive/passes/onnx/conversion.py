# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Union

import onnx
import torch

from olive.common.config_utils import validate_config
from olive.common.utils import tensor_data_to_device
from olive.hardware import AcceleratorSpec, Device
from olive.model import CompositeOnnxModel, ONNXModel, PyTorchModel
from olive.model.hf_utils import get_hf_model_io_config
from olive.model.model_config import IOConfig
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class TraceModelWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *input_data, **input_dict):
        if isinstance(self.model(*input_data, **input_dict), dict):
            return [val for val in self.model(*input_data, **input_dict).values()]
        return self.model(*input_data, **input_dict)


class OnnxConversion(Pass):
    """Convert a PyTorch model to ONNX model using torch.onnx.export on CPU."""

    _requires_user_script = True

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "target_opset": PassConfigParam(
                type_=int, default_value=13, description="The version of the default (ai.onnx) opset to target."
            ),
            "use_dynamo_exporter": PassConfigParam(
                type_=bool, default_value=False, description="Whether to use dynamo_export API to export ONNX model."
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: PyTorchModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> Union[ONNXModel, CompositeOnnxModel]:
        return self._convert_model_on_device(model, data_root, config, output_model_path, "cpu")

    def _convert_model_on_device(
        self,
        model: PyTorchModel,
        data_root: str,
        config: Dict[str, Any],
        output_model_path: str,
        device: str,
    ):
        # check if the model has components
        if model.components:
            onnx_models = []
            component_names = []
            for component_name in model.components:
                component_model = model.get_component(component_name)
                component_output_path = Path(output_model_path).with_suffix("") / component_name
                output_model_component = self._run_for_config(
                    component_model, data_root, config, str(component_output_path)
                )
                output_model_component.model_attributes = (
                    output_model_component.model_attributes or model.model_attributes
                )
                onnx_models.append(output_model_component)
                component_names.append(component_name)
            return CompositeOnnxModel(onnx_models, component_names)

        # get dummy inputs
        dummy_inputs = model.get_dummy_inputs()

        # convert the model
        pytorch_model = model.load_model()
        pytorch_model.eval()

        # TODO: add e2e test for model on cpu but data on gpu; model on gpu but data on cpu
        # put pytorch_model and dummy_inputs at the same device
        pytorch_model.to(device)
        dummy_inputs = tensor_data_to_device(dummy_inputs, device)
        if isinstance(pytorch_model, torch.jit.RecursiveScriptModule):
            pytorch_model = TraceModelWrapper(pytorch_model)

        onnx_model = None
        if config["use_dynamo_exporter"]:
            # TODO: remove this import check once torch.onnx.dynamo_export is available in stable pytorch
            try:
                from torch.onnx import dynamo_export
            except ImportError:
                raise ImportError(
                    "torch.onnx.dynamo_export is not available. Please upgrade your pytorch version to nightly build."
                )
            exported = dynamo_export(
                pytorch_model,
                *dummy_inputs,
                export_options=torch.onnx.ExportOptions(opset_version=config["target_opset"], dynamic_shapes=True),
            )
            onnx_model = exported.model_proto
        else:
            # Standard ONNX export

            # get input and output names, and dynamic axes
            # priority: model.io_config > auto-generated io_config for HF models
            io_config = model.io_config
            if not io_config and (model.hf_config and not model.hf_config.components):
                logger.debug("Using hf config to get io_config for the model.")
                io_config = get_hf_model_io_config(
                    model.hf_config.model_name, model.hf_config.task, model.hf_config.feature
                )
            assert io_config, "Cannot get io_config for the model. Please specify io_config or hf_config for the model."
            io_config = validate_config(io_config, IOConfig)
            input_names = io_config.input_names
            output_names = io_config.output_names
            dynamic_axes = io_config.dynamic_axes

            # some dummy inputs might not be used in the model, so we need to remove them
            # this can happen when we are using an hf dataset to generate dummy inputs
            # only handle dict for now since we cannot get the name of the input from a list/tuple
            if isinstance(dummy_inputs, dict):
                dummy_input_keys = set(dummy_inputs.keys())

                # handle dummy inputs for hf model with past, which has past_key_values
                # match input names in `past_key_values.(hidden_layer_num).(key|value)` pattern
                from transformers.modeling_utils import PreTrainedModel

                if issubclass(type(pytorch_model), PreTrainedModel):
                    for name, input in dummy_inputs.items():
                        if isinstance(input, list):
                            key_value_names = set(
                                [f"{name}.{idx}.key" for idx in range(len(input))]
                                + [f"{name}.{idx}.value" for idx in range(len(input))]
                            )
                            if key_value_names.issubset(set(input_names)):
                                dummy_input_keys.discard(name)

                unused_keys = dummy_input_keys - set(input_names)

                if unused_keys:
                    logger.debug(f"Removing unused dummy inputs: {unused_keys}")
                for key in unused_keys:
                    del dummy_inputs[key]

            output_model_path = ONNXModel.resolve_path(output_model_path)

            # there might be multiple files created during export, so we need to track the dir
            # if there are other processes writing to the same dir, we might end up deleting files created by
            # other processes
            tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp")
            tmp_dir_path = Path(tmp_dir.name)
            tmp_model_path = str(tmp_dir_path / Path(output_model_path).name)

            torch.onnx.export(
                pytorch_model,
                dummy_inputs,
                tmp_model_path,
                export_params=True,
                opset_version=config["target_opset"],
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
            onnx_model = onnx.load(tmp_model_path)

            # the model is loaded into memory, so it's safe to delete previously exported file(s)
            tmp_dir.cleanup()

            # Workaround as described under IOConfig.string_to_int_dim_params: change numeric dim_param to dim_value
            if io_config.string_to_int_dim_params:
                for tensor in onnx_model.graph.output:
                    for dim_proto in tensor.type.tensor_type.shape.dim:
                        if (
                            dim_proto.HasField("dim_param")
                            and dim_proto.dim_param in io_config.string_to_int_dim_params
                        ):
                            dim_value = int(dim_proto.dim_param)
                            dim_proto.Clear()
                            dim_proto.dim_value = dim_value

        # Reset to CPU so the resource consumed on GPU could be free.
        if device != "cpu":
            pytorch_model.to("cpu")
        # save the model to the output path and return the model
        return model_proto_to_olive_model(onnx_model, output_model_path, config)


class DeviceSpecificOnnxConversion(OnnxConversion):
    """Convert a PyTorch model to ONNX model using torch.onnx.export by using specific hardware device."""

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        return False

    def _run_for_config(
        self, model: PyTorchModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> Union[ONNXModel, CompositeOnnxModel]:
        accel_type = self.accelerator_spec.accelerator_type
        device = torch.device("cuda") if accel_type == Device.GPU else torch.device(accel_type)
        return self._convert_model_on_device(model, data_root, config, output_model_path, device)
