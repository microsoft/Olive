# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import platform
from pathlib import Path

from olive.platform_sdk.qualcomm.constants import SDKTargetDevice


class SDKEnv:
    def __init__(self, sdk: str, root_env_name: str, target_arch: str = None, dev: bool = False):
        self.sdk = sdk
        self.root_env_name = root_env_name
        # dev: Whether to use the development version of the SDK,
        # only applicable to SNPE/QNN x86_64 linux/windows
        self.dev = dev
        self.target_arch = self._verify_target_arch(target_arch) if target_arch else self._infer_target_arch()
        self.delimiter = os.path.pathsep

    @property
    def sdk_root_path(self) -> str:
        sdk_root = os.environ.get(self.root_env_name)
        if not sdk_root:
            raise ValueError(f"{self.root_env_name} is not set")
        return sdk_root

    def _verify_target_arch(self, target_device: SDKTargetDevice) -> str:
        if not self.root_env_name:
            raise FileNotFoundError(f"Path {self.sdk_root_path} does not exist")
        archs = list(Path(self.sdk_root_path).glob(f"lib/{target_device}*"))
        if len(archs) == 0:
            raise FileNotFoundError(f"{self.sdk_root_path} missing {target_device}")
        return archs[0].name

    def _infer_target_arch(self, fail_on_unsupported: bool = True) -> str:
        """Infer the target architecture from the SDK root path based on platform and processor."""
        system = platform.system()
        target_arch = None
        if system == "Linux":
            machine = platform.machine()
            if machine == "x86_64":
                target_arch = SDKTargetDevice.x86_64_linux
            else:
                if fail_on_unsupported:
                    raise ValueError(f"Unsupported machine {machine} on system {system}")
        elif system == "Windows":
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
            self._verify_target_arch(target_arch)
        except FileNotFoundError:
            if fail_on_unsupported:
                raise ValueError(f"Unsupported system {system}") from None
            else:
                target_arch = None

        return target_arch

    @property
    def env(self):
        sdk_root_path = self.sdk_root_path

        if self.dev and self.target_arch not in [SDKTargetDevice.x86_64_linux, SDKTargetDevice.x86_64_windows]:
            raise ValueError(f"Unsupported target device {self.target_arch} for development SDK")

        bin_path = str(Path(f"{sdk_root_path}/bin/{self.target_arch}"))
        lib_path = str(Path(f"{sdk_root_path}/lib/{self.target_arch}"))
        python_path = str(Path(f"{sdk_root_path}/lib/python"))

        env = {}
        if platform.system() == "Linux":
            bin_path += self.delimiter + "/usr/bin"
            env["LD_LIBRARY_PATH"] = lib_path
            env["PYTHONPATH"] = python_path
        elif platform.system() == "Windows":
            bin_path += self.delimiter + lib_path

        env["PATH"] = bin_path
        env["SDK_ROOT"] = sdk_root_path

        for paths in env.values():
            for path in paths.split(self.delimiter):
                if not Path(path).exists():
                    raise FileNotFoundError(f"Path {str(Path(path))} does not exist")

        env["TARGET_ARCH"] = self.target_arch
        return env
