# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import platform
from pathlib import Path

from olive.common.constants import OS
from olive.platform_sdk.qualcomm.constants import SDKTargetDevice


class SDKEnv:
    def __init__(self, sdk: str, root_env_name: str, target_arch: str = None, use_dev_tools: bool = False):
        self.sdk = sdk

        self.root_env_name = root_env_name
        if not self.root_env_name:
            raise ValueError("root_env_name is required.")

        self.sdk_root_path = os.environ.get(root_env_name)
        if not self.sdk_root_path:
            raise ValueError(f"{root_env_name} is not set.")
        elif not Path(self.sdk_root_path).exists():
            raise FileNotFoundError(f"The path {self.sdk_root_path} for {self.root_env_name} does not exist.")

        self.use_dev_tools = use_dev_tools
        self.target_arch = self._verify_target_arch(target_arch) if target_arch else self._infer_target_arch()

    def _verify_target_arch(self, target_device: SDKTargetDevice) -> str:
        archs = list(Path(self.sdk_root_path).glob(f"lib/{target_device}*"))
        if not archs:
            raise FileNotFoundError(f"{self.sdk_root_path} missing {target_device}")
        return archs[0].name

    def _infer_target_arch(self, fail_on_unsupported: bool = True) -> str:
        """Infer the target architecture from the SDK root path based on platform and processor."""
        system = platform.system()
        target_arch = None
        if system == OS.LINUX:
            machine = platform.machine()
            if machine == "x86_64":
                target_arch = SDKTargetDevice.x86_64_linux
            else:
                if fail_on_unsupported:
                    raise ValueError(f"Unsupported machine {machine} on system {system}")
        elif system == OS.WINDOWS:
            processor_identifier = os.environ.get("PROCESSOR_IDENTIFIER", "")
            if "ARM" in processor_identifier:
                target_arch = SDKTargetDevice.arm64x_windows
            elif "AARCH" in processor_identifier:
                target_arch = SDKTargetDevice.aarch64_windows
            else:
                target_arch = SDKTargetDevice.x86_64_windows
        else:
            if fail_on_unsupported:
                raise ValueError(f"Unsupported system {system}")
        try:
            target_arch = self._verify_target_arch(target_arch)
        except FileNotFoundError:
            if fail_on_unsupported:
                raise
            else:
                target_arch = None

        return target_arch

    @property
    def env(self):
        sdk_root_path = self.sdk_root_path

        if self.use_dev_tools and self.target_arch not in (
            SDKTargetDevice.x86_64_linux,
            SDKTargetDevice.x86_64_windows,
        ):
            raise ValueError(f"Unsupported target device {self.target_arch} for development SDK")

        bin_path = str(Path(f"{sdk_root_path}/bin/{self.target_arch}"))
        lib_path = str(Path(f"{sdk_root_path}/lib/{self.target_arch}"))
        python_path = str(Path(f"{sdk_root_path}/lib/python"))

        env = {}
        delimiter = os.path.pathsep
        if platform.system() == "Linux":
            bin_path += delimiter + "/usr/bin"
            env["LD_LIBRARY_PATH"] = lib_path
        elif platform.system() == OS.WINDOWS:
            bin_path += delimiter + lib_path

        env["PATH"] = bin_path
        env["SDK_ROOT"] = sdk_root_path
        env["PYTHONPATH"] = python_path

        unfound_paths = []
        for paths in env.values():
            for path in paths.split(delimiter):
                _path = Path(path).resolve()
                if not _path.exists():
                    unfound_paths.append(str(_path))
        if unfound_paths:
            raise FileNotFoundError(f"{unfound_paths} do not exist")

        return env
