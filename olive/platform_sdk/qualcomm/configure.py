# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import logging
import os
import platform
import shutil
from importlib import resources
from pathlib import Path

from olive.common.utils import run_subprocess
from olive.platform_sdk.qualcomm.constants import SDKTargetDevice
from olive.platform_sdk.qualcomm.qnn.env import QNNSDKEnv
from olive.platform_sdk.qualcomm.snpe.env import SNPESDKEnv

# pylint: disable=redefined-outer-name

logger = logging.getLogger(__name__)

script_name = "create_python_env.sh" if platform.system() == "Linux" else "create_python_env.ps1"


def dev(args):
    resource_path = "olive.platform_sdk.qualcomm"
    if args.sdk == "snpe":
        sdk_env = SNPESDKEnv()
    else:
        sdk_env = QNNSDKEnv()
    sdk_arch = sdk_env.target_arch
    if sdk_arch not in (SDKTargetDevice.x86_64_linux, SDKTargetDevice.x86_64_windows):
        return

    logger.info("Configuring %s for %s with python %s...", args.sdk, sdk_arch, args.py_version)
    with resources.path(resource_path, script_name) as create_python_env_path:
        if platform.system() == "Linux":
            cmd = f"bash {create_python_env_path} -v {args.py_version} --sdk {args.sdk}"
        elif platform.system() == "Windows":
            cmd = f"powershell {create_python_env_path} {args.py_version} {args.sdk}"
        run_subprocess(cmd, check=True)
    logger.info("Done")


def eval(args):  # noqa: A001  #pylint: disable=redefined-builtin
    resource_path = "olive.platform_sdk.qualcomm"
    if args.sdk == "snpe":
        sdk_env = SNPESDKEnv()
    else:
        sdk_env = QNNSDKEnv()
    target_arch_name = sdk_env.target_arch
    if target_arch_name not in [SDKTargetDevice.aarch64_windows, SDKTargetDevice.arm64x_windows]:
        return

    sdk_root = sdk_env.sdk_root_path

    logger.info("Configuring %s for %s...", args.sdk, target_arch_name)

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


if __name__ == "__main__":
    # create args for py_version
    os.environ["PIP_EXTRA_ARGS"] = "--no-cache-dir"
    parser = argparse.ArgumentParser("Olive Qualcomm SDK: Configure")
    parser.add_argument(
        "--py_version",
        type=str,
        help="Python version, use 3.6 for tensorflow 1.15. Otherwise 3.8",
        required=True,
        choices=["3.6", "3.8"],
    )
    parser.add_argument(
        "--sdk",
        type=str,
        help="Qualcomm SDK, snpe or qnn",
        required=True,
        choices=["snpe", "qnn"],
    )
    args = parser.parse_args()
    dev(args)
    eval(args)
