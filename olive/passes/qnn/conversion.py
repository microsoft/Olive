# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import platform
from pathlib import Path
from typing import Dict, List, Type, Union

from olive.common.constants import OS
from olive.constants import ModelFileFormat
from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler, PyTorchModelHandler, QNNModelHandler, TensorFlowModelHandler
from olive.model.utils import normalize_path_suffix
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.platform_sdk.qualcomm.runner import QNNSDKRunner


class QNNConversion(Pass):
    """Convert ONNX, TensorFlow, or PyTorch model to QNN C++ model.

    Quantize the model if `--input_list` is provided as extra_args.
    Uses qnn-[framework]-converter tool from the QNN SDK.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            # input_network is required for qnn conversion, but we don't have it in the config.
            # The `input_network` will be set in the runtime.
            "input_dim": PassConfigParam(
                type_=List[str],
                required=False,
                description=(
                    "The names and dimensions of the network input layers specified in the format"
                    " [input_name comma-separated-dimensions], for example:"
                    '        ["data 1,224,224,3"]'
                    " Note that the quotes should always be included in order to"
                    " handle special characters, spaces, etc."
                    " For multiple inputs specify multiple --input_dim on the command line like:"
                    '        ["data 1,224,224,3", "data2 1,224,224,3"]'
                    " If --input_dim is not specified, the input dimensions will be inferred from the model."
                    " If --input_dim is specified, the input dimensions will be used as-is."
                ),
            ),
            "out_node": PassConfigParam(
                type_=List[str],
                required=False,
                description=(
                    "The name of the output node. If not specified, the output node will be inferred from the model."
                    " If specified, the output node will be used as-is."
                    ' Example: ["out_1", "out_2"]'
                ),
            ),
            "extra_args": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Extra arguments to pass to qnn-[framework]-converter tool, e.g."
                    " --show_unconsumed_nodes --custom_io CUSTOM_IO. See the documentation for more details:"
                    " https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/tools.html"
                ),
            ),
        }

    def _run_for_config(
        self,
        model: Union[TensorFlowModelHandler, PyTorchModelHandler, ONNXModelHandler],
        config: Type[BasePassConfig],
        output_model_path: str,
    ) -> QNNModelHandler:
        if isinstance(model, TensorFlowModelHandler):
            converter_platform = "tensorflow"
        elif isinstance(model, PyTorchModelHandler):
            converter_platform = "pytorch"
        elif isinstance(model, ONNXModelHandler):
            converter_platform = "onnx"
        else:
            # TODO(trajep): add tflite support
            raise NotImplementedError(f"Unsupported model handler type: {type(model)}")
        converter_program = [f"qnn-{converter_platform}-converter"]

        runner = QNNSDKRunner(use_dev_tools=True)
        if platform.system() == OS.WINDOWS:
            converter_program = [
                "python",
                str(Path(runner.sdk_env.sdk_root_path) / "bin" / runner.sdk_env.target_arch / converter_program[0]),
            ]

        # get input dim from io_config
        input_dims = None
        if config.input_dim:
            input_dims = map(str.split, config.input_dim)
        elif model.io_config:
            input_dims_tuple = zip(model.io_config["input_names"], model.io_config["input_shapes"])
            input_dims = [[name, ",".join(map(str, shape))] for name, shape in input_dims_tuple]

        out_nodes = None
        if config.out_node:
            out_nodes = config.out_node
        elif model.io_config:
            out_nodes = model.io_config["output_names"]

        output_model_path = normalize_path_suffix(output_model_path, "model.cpp")
        # TODO(anyone): unify other cmd to list to avoid shlex.split error in non-posix mode
        cmd_list = [
            *converter_program,
            "--input_network",
            model.model_path,
            "--output_path",
            output_model_path,
        ]
        if input_dims:
            for i in input_dims:
                cmd_list.extend(["--input_dim", *i])
        if out_nodes:
            for o in out_nodes:
                cmd_list.extend(["--out_node", o])
        if config.extra_args:
            cmd_list.extend(config.extra_args.split())

        runner.run(cmd_list)
        return QNNModelHandler(output_model_path, model_file_format=ModelFileFormat.QNN_CPP)
