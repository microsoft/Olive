# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Type, Union

import onnx
from packaging import version

from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import Field, validator
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.onnx.pipeline import TENSOR_TYPE_MAP
from olive.passes.pass_config import BasePassConfig, PassConfigParam


class PrePostProcessorInput(ConfigBase):
    name: str = Field(..., description="Name of the input.")
    data_type: str = Field(..., description="Data type of the input.")
    shape: List[Union[str, int]] = Field(..., description="Shape of the input.")

    @validator("data_type", pre=True)
    def validate_data_type(cls, v):
        if v not in TENSOR_TYPE_MAP:
            raise ValueError(f"Data type {v} is not supported.")
        return v

    @validator("shape", pre=True)
    def validate_shape(cls, v):
        if not isinstance(v, list):
            raise ValueError(f"Shape {v} must be a list.")
        if not all(isinstance(i, (str, int)) for i in v):
            raise ValueError(f"Shape {v} must be a list of strings or integers.")

        return v


class AppendPrePostProcessingOps(Pass):
    """Add Pre/Post nodes to the input model."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, Dict[str, Any]]:
        config = {
            "pre": PassConfigParam(
                type_=List[Dict[str, Any]],
                default_value=None,
                description="List of pre-processing commands to add.",
            ),
            "post": PassConfigParam(
                type_=List[Dict[str, Any]],
                default_value=None,
                description="List of post-processing commands to add.",
            ),
            "tool_command": PassConfigParam(
                type_=str,
                default_value=None,
                description="Composited tool commands to invoke.",
            ),
            "tool_command_args": PassConfigParam(
                type_=Union[Dict[str, Any], List[PrePostProcessorInput]],
                default_value=None,
                description="""Arguments to pass to tool command or to PrePostProcessor.
                If it is used for PrePostProcessor, the schema would like:
                {
                    "name": "image",
                    "data_type": "uint8",
                    "shape": ["num_bytes"],
                """,
            ),
            "target_opset": PassConfigParam(
                type_=int, default_value=16, description="The version of the default (ai.onnx) opset to target."
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        from onnxruntime import __version__ as OrtVersion

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        tool_command = config.tool_command
        if tool_command:
            if tool_command == "whisper":
                from onnxruntime_extensions import __version__ as ortext_version

                assert version.parse(ortext_version) >= version.parse(
                    "0.9.0"
                ), "Whisper pre-post processing requires onnxruntime_extensions>=0.9.0"

                from olive.passes.utils.whisper_prepost import add_pre_post_processing_to_model

                tool_command_args = config.tool_command_args or {}
                onnx_model = add_pre_post_processing_to_model(
                    model.load_model(), config.target_opset, **tool_command_args
                )
            else:
                # Use the pre-defined helper to add pre/post processing to model.
                from onnxruntime_extensions.tools import add_pre_post_processing_to_model as add_ppp

                # ORT 1.14 and later support ONNX opset 18, which added antialiasing to the Resize operator.
                # Results are much better when that can be used. Minimum opset is 16.
                onnx_opset = config.target_opset

                if version.parse(OrtVersion) >= version.parse("1.14.0"):
                    onnx_opset = 18

                if isinstance(tool_command, str):
                    try:
                        tool_command = getattr(add_ppp, tool_command)
                    except AttributeError:
                        raise AttributeError(f"{tool_command} is not found in onnxruntime_extensions.tools") from None
                elif not isinstance(tool_command, Callable):
                    raise ValueError(
                        "tool_command must be a callable or a string defined in onnxruntime_extensions.tools"
                    ) from None

                kwargs = config.tool_command_args or {}
                kwargs["onnx_opset"] = onnx_opset

                # add the processing commands to the model
                # save the model to a temporary directory, will save it to the output path later with
                # external data config
                with tempfile.TemporaryDirectory(prefix="olive_tmp") as tmp_dir:
                    tmp_dir_path = Path(tmp_dir)
                    tmp_model_path = tmp_dir_path / Path(output_model_path).name
                    tool_command(Path(model.model_path), Path(tmp_model_path), **kwargs)
                    onnx_model = onnx.load(tmp_model_path)
        else:
            # Handle args pre and post
            new_model_proto = self._run_prepost_pipeline(model, config)
            onnx_model = new_model_proto

        olive_model = model_proto_to_olive_model(onnx_model, output_model_path, config)
        olive_model.use_ort_extensions = True
        return olive_model

    def _run_prepost_pipeline(self, model: ONNXModelHandler, config: Type[BasePassConfig]):
        from onnxruntime_extensions.tools.pre_post_processing import PrePostProcessor

        from olive.passes.onnx.pipeline.step_utils import create_named_value, parse_steps

        # Initialize pre/post step instance list
        pre_steps = []
        pre = config.pre
        model_proto = model.load_model()
        if pre:
            steps = parse_steps(model_proto, pre)
            pre_steps = [self.create_step_from_config(step_name, step_param) for step_name, step_param in steps]

        post_steps = []
        post = config.post
        if post:
            steps = parse_steps(model_proto, post)
            post_steps = [self.create_step_from_config(step_name, step_param) for step_name, step_param in steps]

        # Initialize PrePostProcessor instance
        input_param = config.tool_command_args
        assert isinstance(input_param, list)
        assert all(isinstance(i, PrePostProcessorInput) for i in input_param)
        inputs = [create_named_value(i.name, TENSOR_TYPE_MAP[i.data_type], i.shape) for i in input_param]
        pipeline = PrePostProcessor(inputs, config.target_opset)

        if pre_steps:
            pipeline.add_pre_processing(pre_steps)
        if post_steps:
            pipeline.add_post_processing(post_steps)

        return pipeline.run(model_proto)

    def create_step_from_config(self, step_name, step_params):
        from onnxruntime_extensions.tools.pre_post_processing import IoMapEntry

        from olive.passes.onnx.pipeline.step_utils import get_customized_class

        cls = get_customized_class(step_name)
        if isinstance(step_params, tuple):
            # handle io_map
            param, io_map = step_params
            io_map_entity_list = [IoMapEntry(i[0], i[1], i[2]) for i in io_map]
            obj = cls(**param)
            return (obj, io_map_entity_list)
        else:
            return cls(**step_params)
