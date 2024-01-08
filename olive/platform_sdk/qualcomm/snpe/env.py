# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import platform
from copy import deepcopy
from pathlib import Path

from olive.platform_sdk.qualcomm.constants import SDKTargetDevice
from olive.platform_sdk.qualcomm.env import SDKEnv
from olive.platform_sdk.qualcomm.utils.raw_adb import adb_push, run_adb_command


class SNPESDKEnv(SDKEnv):
    def __init__(self, target_arch: str = None, dev: bool = False):
        super().__init__("SNPE", "SNPE_ROOT", target_arch=target_arch, dev=dev)

    @property
    def env(self):
        env = deepcopy(super().env)
        target_arch = env.get("TARGET_ARCH")
        sdk_root_path = env.get("SDK_ROOT")
        python_env_bin_path = str(Path(f"{sdk_root_path}/olive-pyenv/bin"))
        python_env_lib_path = str(Path(f"{sdk_root_path}/olive-pyenv/lib"))
        if platform.system() == "Linux":
            if self.dev:
                if not Path(python_env_bin_path).exists():
                    raise FileNotFoundError(
                        f"Path {python_env_bin_path} does not exist. Please run"
                        " 'python -m olive.platform_sdk.qualcomm.snpe.configure --py_version 3.8'"
                        " to add the missing file."
                    )
                env["LD_LIBRARY_PATH"] += self.delimiter + python_env_lib_path
                env["PATH"] = python_env_bin_path + self.delimiter + env["PATH"]
        elif platform.system() == "Windows":
            if target_arch == SDKTargetDevice.arm64x_windows:
                bin_path = str(Path(f"{sdk_root_path}/olive-arm-win"))
                if not Path(bin_path).exists():
                    raise FileNotFoundError(
                        f"Path {bin_path} does not exist. Please run"
                        " 'python -m olive.platform_sdk.qualcomm.snpe.configure' to add the"
                        " missing folder"
                    )
        return env


class SNPEAndroidEnv(SDKEnv):
    def __init__(self, android_target: str, dev: bool = True, push_to_platform: bool = True):
        super().__init__("SNPE", "SNPE_ROOT", SDKTargetDevice.aarch64_android, dev=dev)
        self.android_target = android_target
        self.push_to_platform = push_to_platform

    @property
    def env(self):
        """Get the environment variables for running SNPE on the target Android device.

        android_target: The target Android device
        push_snpe: Whether to push the SNPE SDK to the target Android device
        """
        # get snpe android root
        snpe_android_root = self.sdk_root_path

        # get android arch name
        android_arch = self.target_arch

        # snpe dirs
        bin_dir = f"{snpe_android_root}/bin/{android_arch}"
        lib_dir = f"{snpe_android_root}/lib/{android_arch}"
        dsp_lib_dir = f"{snpe_android_root}/lib/dsp"

        # abd snpe dirs
        adb_snpe_root = f"/data/local/tmp/olive-snpe/{android_arch}"
        adb_bin_dir = f"{adb_snpe_root}/bin"
        adb_lib_dir = f"{adb_snpe_root}/lib"
        adb_dsp_lib_dir = f"{adb_snpe_root}/dsp"

        if self.push_to_platform:
            # push snpe dirs to android target
            push_pairs = [(bin_dir, adb_bin_dir), (lib_dir, adb_lib_dir), (dsp_lib_dir, adb_dsp_lib_dir)]
            for src, dst in push_pairs:
                adb_push(src, dst, self.android_target, clean=True)

            # change permissions for executables
            for file in Path(bin_dir).iterdir():
                run_adb_command(f"chmod u+x {adb_bin_dir}/{file.name}", self.android_target, shell_cmd=True)

        # environment variables
        return {
            "LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{adb_lib_dir}",
            "PATH": f"$PATH:{adb_bin_dir}",
            "ADSP_LIBRARY_PATH": f"'{adb_dsp_lib_dir};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'",
        }
