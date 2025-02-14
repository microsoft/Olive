# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import platform
from pathlib import Path
from typing import Dict, Type, Union

from olive.common.constants import OS
from olive.constants import ModelFileFormat
from olive.hardware import AcceleratorSpec
from olive.model import QNNModelHandler, SNPEModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.platform_sdk.qualcomm.runner import QNNSDKRunner

logger = logging.getLogger(__name__)


class QNNContextBinaryGenerator(Pass):
    """Create QNN context binary from a QNN model library using a particular backend.

    Uses qnn-context-binary-generator tool from the QNN SDK.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return {
            "backend": PassConfigParam(
                type_=str,
                required=True,
                description="Path to a QNN backend .so library to create the context binary.",
            ),
            "binary_file": PassConfigParam(
                type_=str,
                required=False,
                description=(
                    "Name of the binary file to save the context binary to."
                    " Saved in the same path as --output_dir option with .bin"
                    " as the binary file extension. If not provided, no backend binary"
                    " is created."
                ),
            ),
            "extra_args": PassConfigParam(
                type_=str, default_value=None, description="Extra arguments to qnn-context-binary-generator"
            ),
        }

    def _run_for_config(
        self,
        model: Union[QNNModelHandler, SNPEModelHandler],
        config: Type[BasePassConfig],
        output_model_path: str,
    ) -> QNNModelHandler:
        if platform.system() == OS.WINDOWS:
            raise NotImplementedError("QNNContextBinaryGenerator is not supported on Windows.")

        main_cmd = "qnn-context-binary-generator"
        runner = QNNSDKRunner(use_dev_tools=True)

        extra_args = config.extra_args or ""
        model_arg = f"--model {model.model_path}"

        if model.model_file_format == ModelFileFormat.SNPE_DLC and "--dlc_path" not in extra_args:
            extra_args += f" --dlc_path {model.model_path}"

        # if dlc_path is set, use {qnn_root_path}/lib/{qnn_target_arch_name}/libQnnModelDlc.so
        # as the model argument
        if "--dlc_path" in extra_args:
            qnn_root_path = runner.sdk_env.sdk_root_path
            qnn_target_arch_name = runner.sdk_env.target_arch
            model_arg = f"--model {qnn_root_path}/lib/{qnn_target_arch_name}/libQnnModelDlc.so"

        # input model path's name without suffix
        # TODO(trajep): find .so file in the same directory as the model
        output_model_path = Path(output_model_path).resolve()

        binary_file = config.binary_file
        if not binary_file:
            binary_file = output_model_path.with_suffix(".serialized").name

        output_model_full_path = output_model_path / f"{binary_file}.bin"

        cmd_list = [
            main_cmd,
            model_arg,
            f"--backend {config.backend}",
            f"--output_dir {output_model_path}",
            f"--binary_file {binary_file}" if binary_file else "",
            extra_args,
        ]

        runner.run(" ".join(cmd_list))
        return QNNModelHandler(
            output_model_full_path,
            model_file_format=ModelFileFormat.QNN_SERIALIZED_BIN,
            model_attributes={"backend": config.backend},
        )
