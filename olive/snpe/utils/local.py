import logging
import os
import platform
import time
from pathlib import Path
from typing import Tuple

from olive.common.utils import run_subprocess

logger = logging.getLogger(__name__)


def get_snpe_root() -> str:
    """
    Get the SNPE root directory from the SNPE_ROOT environment variable.
    """
    try:
        snpe_root = os.environ["SNPE_ROOT"]
        logger.debug(f"SNPE_ROOT is set to {snpe_root}")
    except KeyError:
        raise ValueError("SNPE_ROOT is not set")

    return snpe_root


def get_snpe_target_arch(fail_on_unsupported: bool = True) -> str:
    """
    Get the SNPE target architecture from the system and processor.

    fail_on_unsupported: Whether to raise an exception if the system or processor is not supported
    """
    system = platform.system()

    snpe_target_arch = None
    if system == "Linux":
        machine = platform.machine()
        if machine == "x86_64":
            snpe_target_arch = "x64-Linux"
        else:
            if fail_on_unsupported:
                raise ValueError(f"Unsupported machine {machine} on system {system}")
    elif system == "Windows":
        processor_identifier = os.environ.get("PROCESSOR_IDENTIFIER", "")
        snpe_target_arch = "ARM64-Windows" if "ARM" in processor_identifier else "x64-Windows"
    else:
        if fail_on_unsupported:
            raise ValueError(f"Unsupported system {system}")
    logger.debug(f"SNPE target architecture: {snpe_target_arch}")

    return snpe_target_arch


def get_snpe_env(dev: bool = False) -> dict:
    """
    Get the SNPE environment variables.

    dev: Whether to use the SNPE development environment. Only supported on x64-Linux
    """
    target_arch_mapping = {
        "x64-Linux": "x86_64-linux-clang",
        "x64-Windows": "x86_64-windows-vc19",
        "ARM64-Windows": "aarch64-windows-vc19",
    }
    target_arch = get_snpe_target_arch()
    target_arch_name = target_arch_mapping[target_arch]

    if dev and target_arch != "x64-Linux":
        raise ValueError("SNPE development environment is only supported on x64-Linux")

    snpe_root = get_snpe_root()
    bin_path = str(Path(f"{snpe_root}/bin/{target_arch_name}"))
    lib_path = str(Path(f"{snpe_root}/lib/{target_arch_name}"))

    env = {}
    delimiter = os.path.pathsep
    if platform.system() == "Linux":
        env["LD_LIBRARY_PATH"] = lib_path
        if dev:
            python36_env_path = str(Path(f"{snpe_root}/python36-env/bin"))
            if not Path(python36_env_path).exists():
                raise FileNotFoundError(
                    f"Path {python36_env_path} does not exist. Please run 'python -m olive.snpe.configure' to add the"
                    " missing file"
                )
            bin_path += delimiter + python36_env_path
            env["PYTHONPATH"] = str(Path(f"{snpe_root}/lib/python"))
        bin_path += delimiter + "/usr/bin"
    elif platform.system() == "Windows":
        if target_arch == "ARM64-Windows":
            bin_path = str(Path(f"{snpe_root}/olive-arm-win"))
            if not Path(bin_path).exists():
                raise FileNotFoundError(
                    f"Path {bin_path} does not exist. Please run 'python -m olive.snpe.configure' to add the"
                    " missing folder"
                )
        else:
            bin_path += delimiter + lib_path
    env["PATH"] = bin_path

    for paths in env.values():
        for path in paths.split(delimiter):
            if not Path(path).exists():
                raise FileNotFoundError(f"Path {str(Path(path))} does not exist")

    return env


def run_snpe_command(
    cmd: str, dev: bool = False, runs: int = 1, sleep: int = 0, log_error: bool = True
) -> Tuple[str, str]:
    """
    Run a SNPE command.

    cmd: The command to run
    dev: Whether to use the SNPE development environment. Only supported on x64-Linux
    runs: The number of times to run the command
    sleep: The number of seconds to sleep between runs
    log_error: Whether to log an error if the command fails
    """
    env = get_snpe_env(dev)
    full_cmd = cmd

    for run in range(runs):
        run_log_msg = "" if runs == 1 else f" (run {run + 1}/{runs})"
        logger.debug(f"Running SNPE command{run_log_msg}: {full_cmd}")
        returncode, stdout, stderr = run_subprocess(full_cmd, env)
        logger.debug(f"Return code: {returncode} \n Stdout: {stdout} \n Stderr: {stderr}")
        if returncode != 0:
            break
        if sleep > 0 and run < runs - 1:
            time.sleep(sleep)

    if returncode != 0:
        error_msg = (
            f"Error running SNPE command. \n Command: {full_cmd} \n Return code: {returncode} \n Stdout: {stdout} \n"
            f" Stderr: {stderr}"
        )
        if log_error:
            logger.error(error_msg)
        raise RuntimeError(error_msg)

    return stdout, stderr
