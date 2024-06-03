# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import platform
from pathlib import Path

from olive.common.constants import OS
from olive.platform_sdk.qualcomm.env import SDKEnv


class QNNSDKEnv(SDKEnv):
    def __init__(self, target_arch: str = None, use_dev_tools: bool = False):
        super().__init__("QNN", "QNN_SDK_ROOT", target_arch=target_arch, use_dev_tools=use_dev_tools)

    @property
    def env(self):
        env = super().env
        sdk_root_path = self.sdk_root_path
        delimiter = os.path.pathsep
        python_env_parent_folder = "Scripts" if platform.system() == OS.WINDOWS else "bin"
        python_env_bin_path = str(Path(f"{sdk_root_path}/olive-pyenv/{python_env_parent_folder}"))

        env["PATH"] += delimiter + os.environ["PATH"]
        if self.use_dev_tools:
            if not Path(python_env_bin_path).exists():
                raise FileNotFoundError(
                    f"Path {python_env_bin_path} does not exist. Please run"
                    " 'olive configure-qualcomm-sdk --py_version 3.8 --sdk qnn'"
                    " to add the missing file."
                )
            env["PATH"] = python_env_bin_path + delimiter + env["PATH"]
        if platform.system() == OS.WINDOWS:
            for k, v in os.environ.items():
                if k not in env:
                    env[k] = v
        env["QNN_SDK_ROOT"] = sdk_root_path
        return env

    def get_qnn_backend(self, backend_name):
        backend_path = Path(self.sdk_root_path) / "lib" / self.target_arch / backend_name
        backend_path = (
            backend_path.with_suffix(".dll") if platform.system() == OS.WINDOWS else backend_path.with_suffix(".so")
        )

        if not backend_path.exists():
            raise FileNotFoundError(f"QNN backend {backend_path} does not exist.")
        return backend_path
