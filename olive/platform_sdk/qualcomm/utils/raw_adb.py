# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import platform
import time
from pathlib import Path
from typing import Tuple

from olive.common.utils import run_subprocess

logger = logging.getLogger(__name__)


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
        if platform.system() == "Windows":
            cmd = f"shell {cmd}"
        elif platform.system() == "Linux":
            cmd = f'shell "{cmd}"'

    # run the command
    full_cmd = f"{adb_head} {cmd}"
    for run in range(runs):
        run_log_msg = "" if runs == 1 else f" (run {run + 1}/{runs})"
        logger.debug(f"Running ADB command{run_log_msg}: {full_cmd}")
        returncode, stdout, stderr = run_subprocess(full_cmd)
        logger.debug(f"Return code: {returncode} \n Stdout: {stdout} \n Stderr: {stderr}")
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
