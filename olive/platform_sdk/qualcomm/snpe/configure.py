# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import logging
import shutil
from importlib import resources
from pathlib import Path

from olive.common.utils import run_subprocess
from olive.platform_sdk.qualcomm.constants import SDKTargetDevice
from olive.platform_sdk.qualcomm.snpe.env import SNPESDKEnv

logger = logging.getLogger(__name__)


def dev(args):
    snpe_env = SNPESDKEnv()
    snpe_arch = snpe_env.target_arch
    if snpe_arch != SDKTargetDevice.x86_64_linux:
        return

    logger.info(f"Configuring SNPE for {snpe_arch} with python{args.py_version}...")
    with resources.path("olive.platform_sdk.qualcomm.snpe", "create_python_env.sh") as create_python_env_path:
        cmd = f"bash {create_python_env_path} -v {args.py_version}"
        return_code, stdout, stderr = run_subprocess(cmd)
        if return_code != 0:
            raise RuntimeError(f"Failed to create python36 environment. stdout: {stdout}, stderr: {stderr}")
    logger.info("Done")


def eval():  # noqa: A001  #pylint: disable=redefined-builtin
    snpe_env = SNPESDKEnv()
    target_arch_name = snpe_env.target_arch
    if target_arch_name not in [SDKTargetDevice.aarch64_windows, SDKTargetDevice.arm64x_windows]:
        return

    snpe_root = snpe_env.sdk_root_path

    logger.info(f"Configuring SNPE for {target_arch_name}...")

    # paths for snpe files
    bin_path = Path(f"{snpe_root}/bin/{target_arch_name}")
    lib_path = Path(f"{snpe_root}/lib/{target_arch_name}")
    dsp_lib_path = Path(f"{snpe_root}/lib/dsp")

    # symlink all files under 'olive-arm-win'
    # If all files are not under the same path, there are problems with the dll files being spread under
    # multiple PATH directories
    olive_snpe_path = Path(snpe_root).resolve() / "olive-arm-win"
    if olive_snpe_path.exists():
        shutil.rmtree(olive_snpe_path)
    olive_snpe_path.mkdir()
    for path in [bin_path, lib_path, dsp_lib_path]:
        for member in path.iterdir():
            (olive_snpe_path / member.name).symlink_to(member)

    # copy over libcdsprpc.dll
    with resources.path("olive.platform_sdk.qualcomm.snpe", "copy_libcdsprpc.ps1") as copy_libcdsprpc_path:
        cmd = f"powershell {copy_libcdsprpc_path} {olive_snpe_path}"
        return_code, stdout, stderr = run_subprocess(cmd)
        if return_code != 0 or not (olive_snpe_path / "libcdsprpc.dll").exists():
            raise RuntimeError(f"Failed to copy libcdsprpc. stdout: {stdout}, stderr: {stderr}")

    logger.info("Done")


if __name__ == "__main__":
    # create args for py_version
    parser = argparse.ArgumentParser("Olive SNPE: Configure")
    parser.add_argument(
        "--py_version",
        type=str,
        help="Python version, use 3.6 for tensorflow 1.15. Otherwise 3.8",
        required=True,
        choices=["3.6", "3.8"],
    )
    args = parser.parse_args()
    dev(args)
    eval()
