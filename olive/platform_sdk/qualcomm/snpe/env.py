# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import platform
from pathlib import Path

from olive.platform_sdk.qualcomm.constants import SDKTargetDevice
from olive.platform_sdk.qualcomm.env import SDKEnv


class SNPESDKEnv(SDKEnv):
    def __init__(self, target_arch: str = None, use_dev_tools: bool = False):
        super().__init__("SNPE", "SNPE_ROOT", target_arch=target_arch, use_dev_tools=use_dev_tools)

    @property
    def env(self):
        env = super().env
        target_arch = self.target_arch
        sdk_root_path = self.sdk_root_path
        delimiter = os.path.pathsep
        python_env_bin_path = str(Path(f"{sdk_root_path}/olive-pyenv/bin"))
        python_env_lib_path = str(Path(f"{sdk_root_path}/olive-pyenv/lib"))
        if platform.system() == "Linux":
            if self.use_dev_tools:
                if not Path(python_env_bin_path).exists():
                    raise FileNotFoundError(
                        f"Path {python_env_bin_path} does not exist. Please run"
                        " 'python -m olive.platform_sdk.qualcomm.configure --py_version 3.8 --sdk snpe'"
                        " to add the missing file."
                    )
                env["LD_LIBRARY_PATH"] += delimiter + python_env_lib_path
                env["PATH"] = python_env_bin_path + delimiter + env["PATH"]
        elif platform.system() == "Windows" and target_arch == SDKTargetDevice.arm64x_windows:
            bin_path = str(Path(f"{sdk_root_path}/olive-arm-win"))
            if not Path(bin_path).exists():
                raise FileNotFoundError(
                    f"Path {bin_path} does not exist. Please run"
                    " 'python -m olive.platform_sdk.qualcomm.configure --py_version 3.8 --sdk snpe' to add the"
                    " missing folder"
                )
        return env
