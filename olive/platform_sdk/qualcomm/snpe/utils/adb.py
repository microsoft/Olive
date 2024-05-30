# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
import platform
import time
from pathlib import Path
from typing import Tuple

from olive.common.constants import OS
from olive.common.utils import run_subprocess

logger = logging.getLogger(__name__)


def get_snpe_android_root() -> str:
    """Get the SNPE Android root directory from the SNPE_ANDROID_ROOT environment variable.

    On Linux, this is the same as the SNPE_ROOT environment variable.
    On Windows, this is the complete (non-Windows) unzipped SNPE SDK directory.
    """
    try:
        snpe_android_root = os.environ["SNPE_ANDROID_ROOT"]
        logger.debug("SNPE_ANDROID_ROOT is set to %s", snpe_android_root)
    except KeyError:
        raise ValueError("SNPE_ANDROID_ROOT is not set") from None

    return snpe_android_root


def get_snpe_android_arch(snpe_android_root: str) -> str:
    """Get the SNPE Android architecture from the SNPE Android root directory.

    snpe_android_root: The unzipped SNPE SDK directory
    """
    if not Path(snpe_android_root).exists():
        raise FileNotFoundError(f"Path {snpe_android_root} does not exist")

    android_archs = list(Path(snpe_android_root).glob("lib/aarch64-android*"))
    if len(android_archs) == 0:
        raise FileNotFoundError(f"SNPE_ANDROID_ROOT {snpe_android_root} missing aarch64-android-*")

    android_arch = android_archs[0].name
    logger.debug("SNPE Android architecture: %s", android_arch)

    return android_arch


def run_adb_command(
    cmd: str, android_target: str, shell_cmd: bool = False, runs: int = 1, sleep: int = 0, log_error: bool = True
) -> Tuple[str, str]:
    """Run an ADB command on the target Android device.

    cmd: The command to run
    android_target: The target Android device
    shell_cmd: Whether to run the command in an ADB shell
    runs: The number of times to run the command
    sleep: The number of seconds to sleep between runs
    log_error: Whether to log an error if the command fails
    """
    # create the adb command prefix
    adb_head = f"adb -s {android_target}"

    # platform specific shell command
    if shell_cmd:
        if platform.system() == OS.WINDOWS:
            cmd = f"shell {cmd}"
        elif platform.system() == OS.LINUX:
            cmd = f'shell "{cmd}"'

    # run the command
    full_cmd = f"{adb_head} {cmd}"
    for run in range(runs):
        run_log_msg = "" if runs == 1 else f" (run {run + 1}/{runs})"
        logger.debug("Running ADB command %s: %s", run_log_msg, full_cmd)
        returncode, stdout, stderr = run_subprocess(full_cmd)
        logger.debug("Return code: %d \n Stdout: %s \n Stderr: %s", returncode, stdout, stderr)
        if returncode != 0:
            break
        if sleep > 0 and run < runs - 1:
            time.sleep(sleep)

    # check the return code
    if returncode != 0:
        error_msg = (
            f"Error running ADB command. \n Command: {full_cmd} \n Return code: {returncode} \n Stdout: {stdout} \n"
            f" Stderr: {stderr}"
        )
        if log_error:
            logger.error(error_msg)
        raise RuntimeError(error_msg)

    return stdout, stderr


def adb_push(src: str, dst: str, android_target: str, clean: bool = False):
    """Push a file or directory to the target Android device.

    src: The source file or directory
    dst: The destination directory
    android_target: The target Android device
    clean: Whether to clean the destination directory before pushing
    """
    # check if src exists
    if not Path(src).exists():
        raise FileNotFoundError(f"Path {src} does not exist")

    # clean dst before push
    if clean:
        run_adb_command(f"rm -rf {Path(dst).as_posix()}", android_target, shell_cmd=True)

    # create dst
    run_adb_command(f"mkdir -p {Path(dst).as_posix()}", android_target, shell_cmd=True)

    # push src to dst
    src_path = Path(src).resolve()
    if src_path.is_dir():
        run_adb_command(f"push {src_path.as_posix()}/. {dst}", android_target)
    elif src_path.is_file():
        run_adb_command(f"push {src_path.as_posix()} {dst}", android_target)


def get_snpe_adb_env(android_target: str, push_snpe: bool = True) -> dict:
    """Get the environment variables for running SNPE on the target Android device.

    android_target: The target Android device
    push_snpe: Whether to push the SNPE SDK to the target Android device
    """
    # get snpe android root
    snpe_android_root = get_snpe_android_root()

    # get android arch name
    android_arch = get_snpe_android_arch(snpe_android_root)

    # snpe dirs
    bin_dir = f"{snpe_android_root}/bin/{android_arch}"
    lib_dir = f"{snpe_android_root}/lib/{android_arch}"
    dsp_lib_dir = f"{snpe_android_root}/lib/dsp"

    # abd snpe dirs
    adb_snpe_root = f"/data/local/tmp/olive-snpe/{android_arch}"
    adb_bin_dir = f"{adb_snpe_root}/bin"
    adb_lib_dir = f"{adb_snpe_root}/lib"
    adb_dsp_lib_dir = f"{adb_snpe_root}/dsp"

    if push_snpe:
        # push snpe dirs to android target
        push_pairs = [(bin_dir, adb_bin_dir), (lib_dir, adb_lib_dir), (dsp_lib_dir, adb_dsp_lib_dir)]
        for src, dst in push_pairs:
            adb_push(src, dst, android_target, clean=True)

        # change permissions for executables
        for file in Path(bin_dir).iterdir():
            run_adb_command(f"chmod u+x {adb_bin_dir}/{file.name}", android_target, shell_cmd=True)

    # environment variables
    return {
        "LD_LIBRARY_PATH": f"$LD_LIBRARY_PATH:{adb_lib_dir}",
        "PATH": f"$PATH:{adb_bin_dir}",
        "ADSP_LIBRARY_PATH": f"'{adb_dsp_lib_dir};/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp'",
    }


def prepare_snpe_adb(android_target: str):
    """Prepare the target Android device for running SNPE.

    android_target: The target Android device
    """
    get_snpe_adb_env(android_target, push_snpe=True)
    logger.info("SNPE prepared for Android target %s", android_target)


def run_snpe_adb_command(
    cmd: str, android_target: str, push_snpe: bool = True, runs: int = 1, sleep: int = 0, log_error: bool = True
):
    """Run a SNPE command on the target Android device.

    cmd: The command to run
    android_target: The target Android device
    push_snpe: Whether to push the SNPE SDK to the target Android device
    runs: The number of times to run the command
    sleep: The number of seconds to sleep between runs
    log_error: Whether to log an error if the command fails
    """
    # get snpe env
    env = get_snpe_adb_env(android_target, push_snpe)

    # run snpe command
    full_cmd = ""
    for key, value in env.items():
        full_cmd += f"export {key}={value} && "
    full_cmd += cmd

    return run_adb_command(full_cmd, android_target, shell_cmd=True, runs=runs, sleep=sleep, log_error=log_error)
