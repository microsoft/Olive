# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import platform
from pathlib import Path
from typing import Any, Callable, Dict, Union

from olive.constants import ModelFileFormat
from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler, PyTorchModelHandler, QNNModelHandler, TensorFlowModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import PassConfigParam
from olive.passes.qnn.common import get_env_config
from olive.platform_sdk.qualcomm.runner import QNNSDKRunner

logger = logging.getLogger(__name__)


class QNNModelLibGenerator(Pass):
    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "lib_targets": PassConfigParam(
                type_=str,
                required=False,
                description=(
                    "Specifies the targets to build the models for. Default: aarch64-android x86_64-linux-clang"
                ),
            ),
            "lib_name": PassConfigParam(
                type_=str,
                required=False,
                description=(
                    "Specifies the name to use for libraries. Default: uses name in <model.bin> if provided, "
                    " else generic qnn_model.so"
                ),
            ),
        }
        config.update(get_env_config())
        return config

    @staticmethod
    def _validators() -> Dict[str, Callable[..., Any]]:
        pass

    def _run_for_config(
        self,
        model: Union[TensorFlowModelHandler, PyTorchModelHandler, ONNXModelHandler],
        data_root: str,
        config: Dict[str, Any],
        output_model_path: str,
    ) -> QNNModelHandler:
        main_cmd = "qnn-model-lib-generator"
        runner = QNNSDKRunner(use_dev_tools=True)
        if platform.system() == "Windows":
            main_cmd = "python " + str(
                Path(runner.sdk_env.sdk_root_path) / "bin" / runner.sdk_env.target_arch / main_cmd
            )

        # input model path's name without suffix
        input_model_path = Path(model.model_path).resolve()
        input_model_bin = input_model_path.parent / (input_model_path.stem + ".bin")
        if not input_model_bin.exists():
            logger.debug("No model.bin found, using generic qnn_model.so")
            input_model_bin = None

        output_model_path = Path(output_model_path).resolve()

        cmd_list = [
            main_cmd,
            f"-c {model.model_path}",
            f"-b {input_model_bin}" if input_model_bin else "",
            f"-t {config['lib_targets']}" if config.get("lib_targets") else "",
            f"-l {config['lib_name']}" if config.get("lib_name") else "",
            f"-o {output_model_path}",
        ]
        runner = QNNSDKRunner(use_dev_tools=True)
        runner.run(" ".join(cmd_list), use_olive_env=config["use_olive_env"])
        return QNNModelHandler(output_model_path, model_file_format=ModelFileFormat.QNN_LIB)
