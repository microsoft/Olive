# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import platform
from pathlib import Path
from typing import Dict, Type

from olive.common.constants import OS
from olive.constants import ModelFileFormat
from olive.hardware import AcceleratorSpec
from olive.model import QNNModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.platform_sdk.qualcomm.runner import QNNSDKRunner

logger = logging.getLogger(__name__)


class QNNModelLibGenerator(Pass):
    """Compile QNN C++ model source code into QNN model library for a specific target.

    Uses qnn-model-lib-generator tool from the QNN SDK.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
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

    def _run_for_config(
        self,
        model: QNNModelHandler,
        config: Type[BasePassConfig],
        output_model_path: str,
    ) -> QNNModelHandler:
        main_cmd = "qnn-model-lib-generator"
        runner = QNNSDKRunner(use_dev_tools=True)
        if platform.system() == OS.WINDOWS:
            main_cmd = "python " + str(
                Path(runner.sdk_env.sdk_root_path) / "bin" / runner.sdk_env.target_arch / main_cmd
            )

        # input model path's name without suffix
        input_model_path = Path(model.model_path).resolve()
        input_model_bin = input_model_path.parent / (input_model_path.stem + ".bin")
        if not input_model_bin.exists():
            logger.debug("No model.bin found, using generic qnn_model.so")
            input_model_bin = None

        # lib generator requires the output path to be a directory
        output_model_path = Path(output_model_path).resolve()
        # for multi-components models, the output path might not exist
        output_model_path.mkdir(parents=True, exist_ok=True)
        assert output_model_path.is_dir(), f"Output path {output_model_path} is not a directory"

        cmd_list = [
            main_cmd,
            f"-c {model.model_path}",
            f"-b {input_model_bin}" if input_model_bin else "",
            f"-t {config['lib_targets']}" if config.get("lib_targets") else "",
            f"-l {config['lib_name']}" if config.get("lib_name") else "",
            f"-o {output_model_path}",
        ]
        runner = QNNSDKRunner(use_dev_tools=True)
        runner.run(" ".join(cmd_list))
        return QNNModelHandler(output_model_path, model_file_format=ModelFileFormat.QNN_LIB)
