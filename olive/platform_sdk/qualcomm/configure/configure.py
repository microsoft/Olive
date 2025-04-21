# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import os
import platform
import shutil
from importlib import resources
from pathlib import Path

from olive.common.constants import OS
from olive.common.utils import run_subprocess
from olive.platform_sdk.qualcomm.constants import SDKTargetDevice
from olive.platform_sdk.qualcomm.qnn.env import QNNSDKEnv
from olive.platform_sdk.qualcomm.snpe.env import SNPESDKEnv

logger = logging.getLogger(__name__)


def configure_dev(py_version: str, sdk: str):
    """Configure Qualcomm SDK for model development."""
    os.environ["PIP_EXTRA_ARGS"] = "--no-cache-dir"

    resource_path = "olive.platform_sdk.qualcomm"
    if sdk == "snpe":
        sdk_env = SNPESDKEnv()
    else:
        sdk_env = QNNSDKEnv()
    sdk_arch = sdk_env.target_arch
    if sdk_arch not in (SDKTargetDevice.x86_64_linux, SDKTargetDevice.x86_64_windows):
        return

    script_name = "create_python_env.sh" if platform.system() == OS.LINUX else "create_python_env.ps1"

    logger.info("Configuring %s for %s with python %s...", sdk, sdk_arch, py_version)
    cmd = None
    with resources.path(resource_path, script_name) as create_python_env_path:
        if platform.system() == OS.LINUX:
            cmd = f"bash {create_python_env_path} -v {py_version} --sdk {sdk}"
        elif platform.system() == OS.WINDOWS:
            cmd = f"powershell {create_python_env_path} {py_version} {sdk}"
        run_subprocess(cmd, check=True)
    logger.info("Done")


def configure_eval(sdk: str):
    """Configure Qualcomm SDK for model evaluation."""
    resource_path = "olive.platform_sdk.qualcomm"
    if sdk == "snpe":
        sdk_env = SNPESDKEnv()
    else:
        sdk_env = QNNSDKEnv()
    target_arch_name = sdk_env.target_arch
    if target_arch_name not in [SDKTargetDevice.aarch64_windows, SDKTargetDevice.arm64x_windows]:
        return

    sdk_root = sdk_env.sdk_root_path

    logger.info("Configuring %s for %s...", sdk, target_arch_name)

    bin_path = Path(f"{sdk_root}/bin/{target_arch_name}")
    lib_path = Path(f"{sdk_root}/lib/{target_arch_name}")
    dsp_lib_path = Path(f"{sdk_root}/lib/dsp")

    # symlink all files under 'olive-arm-win'
    # If all files are not under the same path, there are problems with the dll files being spread under
    # multiple PATH directories
    olive_sdk_path = Path(sdk_root).resolve() / "olive-arm-win"
    if olive_sdk_path.exists():
        shutil.rmtree(olive_sdk_path)
    olive_sdk_path.mkdir()
    for path in [bin_path, lib_path, dsp_lib_path]:
        for member in path.iterdir():
            (olive_sdk_path / member.name).symlink_to(member)

    # copy over libcdsprpc.dll
    with resources.path(resource_path, "copy_libcdsprpc.ps1") as copy_libcdsprpc_path:
        cmd = f"powershell {copy_libcdsprpc_path} {olive_sdk_path}"
        run_subprocess(cmd, check=True)
        if not (olive_sdk_path / "libcdsprpc.dll").exists():
            raise RuntimeError(f"Failed to copy libcdsprpc.dll to {olive_sdk_path}")

    logger.info("Done")


def configure(py_version: str, sdk: str):
    """Configure Qualcomm SDK for Olive.

    :param py_version: Python version, use 3.6 for tensorflow 1.15 and 3.8 otherwise
    :param sdk: Qualcomm SDK, snpe or qnn
    """
    configure_dev(py_version, sdk)
    configure_eval(sdk)
