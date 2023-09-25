# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import shutil
from importlib import resources
from pathlib import Path

from olive.common.utils import run_subprocess
from olive.snpe.utils.local import get_snpe_root, get_snpe_target_arch, get_snpe_win_arch_name

logger = logging.getLogger("olive.snpe.configure")


def dev():
    snpe_arch = get_snpe_target_arch(False)
    if snpe_arch != "x64-Linux":
        return

    get_snpe_root()

    logger.info(f"Configuring SNPE for {snpe_arch}...")
    with resources.path("olive.snpe", "create_python36_env.sh") as create_python36_env_path:
        cmd = f"bash {create_python36_env_path}"
        return_code, stdout, stderr = run_subprocess(cmd)
        if return_code != 0:
            raise RuntimeError(f"Failed to create python36 environment. stdout: {stdout}, stderr: {stderr}")
    logger.info("Done")


def eval():  # noqa: A001
    snpe_arch = get_snpe_target_arch(False)
    if snpe_arch != "ARM64-Windows":
        return

    snpe_root = get_snpe_root()
    target_arch_name = get_snpe_win_arch_name(snpe_root, snpe_arch)

    logger.info(f"Configuring SNPE for {snpe_arch}...")

    # paths for snpe files
    bin_path = Path(f"{snpe_root}/bin/{target_arch_name}")
    lib_path = Path(f"{snpe_root}/lib/{target_arch_name}")
    dsp_lib_path = Path(f"{snpe_root}/lib/dsp")

    # symlink all files under 'olive-arm-win'
    # If all files are not under the same path, there are problems with the dll files being spread under
    # multiple PATH directories
    olive_snpe_path = Path(get_snpe_root()).resolve() / "olive-arm-win"
    if olive_snpe_path.exists():
        shutil.rmtree(olive_snpe_path)
    olive_snpe_path.mkdir()
    for path in [bin_path, lib_path, dsp_lib_path]:
        for member in path.iterdir():
            (olive_snpe_path / member.name).symlink_to(member)

    # copy over libcdsprpc.dll
    with resources.path("olive.snpe", "copy_libcdsprpc.ps1") as copy_libcdsprpc_path:
        cmd = f"powershell {copy_libcdsprpc_path} {olive_snpe_path}"
        return_code, stdout, stderr = run_subprocess(cmd)
        if return_code != 0 or not (olive_snpe_path / "libcdsprpc.dll").exists():
            raise RuntimeError(f"Failed to copy libcdsprpc. stdout: {stdout}, stderr: {stderr}")

    logger.info("Done")


if __name__ == "__main__":
    dev()
    eval()
