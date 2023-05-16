# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

import onnx
from onnxruntime import __version__ as OrtVersion

from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam


class AppendPrePostProcessingOps(Pass):
    """
    Add Pre/Post nodes to the input model
    """

    @staticmethod
    def _default_config() -> Dict[str, Dict[str, Any]]:
        config = {
            "pre": PassConfigParam(
                type_=List[str],
                default_value=None,
                description="List of pre-processing commands to add.",
            ),
            "post": PassConfigParam(
                type_=List[str],
                default_value=None,
                description="List of post-processing commands to add.",
            ),
            "tool_command": PassConfigParam(
                type_=str,
                default_value=None,
                description="Composited tool commands to invoke.",
            ),
            "tool_command_args": PassConfigParam(
                type_=Dict[str, Any], default_value=None, description="Arguments to pass to tool command."
            ),
            "target_opset": PassConfigParam(
                type_=int, default_value=16, description="The version of the default (ai.onnx) opset to target."
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(self, model: ONNXModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        output_model_path = ONNXModel.resolve_path(output_model_path)

        # temporary directory to store the model to
        # we will save the model to the final destination later with the external data config
        tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp")
        tmp_dir_path = Path(tmp_dir.name)
        tmp_model_path = str(tmp_dir_path / Path(output_model_path).name)

        tool_command = config.get("tool_command")
        if tool_command:
            if tool_command == "whisper":
                from olive.passes.utils.whisper_prepost import add_pre_post_processing_to_model as add_ppp

                kwargs = config.get("tool_command_args") or {}
                add_ppp(model.load_model(), tmp_model_path, **kwargs)
            else:
                # Use the pre-defined helper to add pre/post processing to model.
                from onnxruntime_extensions.tools import add_pre_post_processing_to_model as add_ppp

                # ORT 1.14 and later support ONNX opset 18, which added antialiasing to the Resize operator.
                # Results are much better when that can be used. Minimum opset is 16.
                onnx_opset = config.get("target_opset")
                from packaging import version

                if version.parse(OrtVersion) >= version.parse("1.14.0"):
                    onnx_opset = 18

                if isinstance(tool_command, str):
                    try:
                        tool_command = getattr(add_ppp, tool_command)
                    except AttributeError:
                        raise AttributeError(f"{tool_command} is not found in onnxruntime_extensions.tools")
                elif not isinstance(tool_command, Callable):
                    raise ValueError(
                        "tool_command must be a callable or a string defined in onnxruntime_extensions.tools"
                    )

                kwargs = config.get("tool_command_args") or {}
                kwargs["onnx_opset"] = onnx_opset

                # add the processing commands to the mode.
                tool_command(Path(model.model_path), Path(tmp_model_path), **kwargs)
        else:
            # TODO: Handle args pre and post here!
            pass

        # load the model
        onnx_model = onnx.load(tmp_model_path)
        # the model is loaded into memory, so it's safe to delete previously exported files
        tmp_dir.cleanup()

        olive_model = model_proto_to_olive_model(
            onnx_model, output_model_path, config, model.name, model.model_file_format
        )
        olive_model.use_ort_extensions = True
        return olive_model
