# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import platform
from pathlib import Path
from typing import Any, Dict

from olive.constants import ModelFileFormat
from olive.hardware import AcceleratorSpec
from olive.model import QNNModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.pass_config import PassConfigParam
from olive.platform_sdk.qualcomm.runner import QNNSDKRunner

logger = logging.getLogger(__name__)


class QNNContextBinaryGenerator(Pass):
    """Create QNN context binary from a QNN model library using a particular backend.

    Uses qnn-context-binary-generator tool from the QNN SDK.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        if platform.system() == "Windows":
            raise NotImplementedError("QNNContextBinaryGenerator is not supported on Windows.")

        return {
            "backend": PassConfigParam(
                type_=str,
                required=True,
                description=("Path to a QNN backend .so library to create the context binary."),
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
        model: QNNModelHandler,
        data_root: str,
        config: Dict[str, Any],
        output_model_path: str,
    ) -> QNNModelHandler:
        main_cmd = "qnn-context-binary-generator"
        runner = QNNSDKRunner(use_dev_tools=True)

        # input model path's name without suffix
        # TODO(trajep): find .so file in the same directory as the model
        output_model_path = Path(output_model_path).resolve()

        binary_file = config["binary_file"]
        if not binary_file:
            binary_file = output_model_path.with_suffix(".serialized").name

        output_model_full_path = output_model_path / f"{binary_file}.bin"

        cmd_list = [
            main_cmd,
            f"--model {model.model_path}",
            f"--backend {config['backend']}",
            f"--output_dir {output_model_path}",
            f"--binary_file {binary_file}" if binary_file else "",
            config["extra_args"] or "",
        ]

        runner.run(" ".join(cmd_list))
        return QNNModelHandler(
            output_model_full_path,
            model_file_format=ModelFileFormat.QNN_SERIALIZED_BIN,
            model_attributes={"backend": config["backend"]},
        )
