# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import time

from olive.common.utils import run_subprocess
from olive.platform_sdk.qualcomm.snpe.env import SNPEAndroidEnv, SNPESDKEnv
from olive.platform_sdk.qualcomm.utils.raw_adb import run_adb_command

logger = logging.getLogger(__name__)


class SDKRunner:
    def __init__(
        self,
        platform,
        cmd: str,
        dev: bool = False,
        runs: int = 1,
        sleep: int = 0,
        log_error: bool = True,
        android_target: str = None,
    ):
        self.platform = platform
        self.cmd = cmd
        self.dev = dev
        self.runs = runs
        self.sleep = sleep
        self.log_error = log_error
        self.android_target = android_target

    def runner_env(self):
        if self.platform == "SNPE":
            env = SNPESDKEnv(dev=self.dev).env
        else:
            raise ValueError(f"Unsupported platform {self.platform}")
        return env

    def run(self):
        env = self.runner_env()
        full_cmd = self.cmd
        for run in range(self.runs):
            run_log_msg = "" if self.runs == 1 else f" (run {run + 1}/{self.runs})"
            logger.debug(f"Running SNPE command{run_log_msg}: {full_cmd}")
            returncode, stdout, stderr = run_subprocess(full_cmd, env)
            logger.debug(f"Return code: {returncode} \n Stdout: {stdout} \n Stderr: {stderr}")
            if returncode != 0:
                break
            if self.sleep > 0 and run < self.runs - 1:
                time.sleep(self.sleep)

        if returncode != 0:
            error_msg = [
                "Error running SNPE command.",
                f"Command: {full_cmd}",
                f"Return code: {returncode}Stdout: {stdout}" if stdout else "",
                f"Stderr: {stderr}" if stderr else "",
                f"ENV: {env}",
            ]
            if self.log_error:
                logger.error("\n".join(error_msg))
            raise RuntimeError(error_msg)
        return stdout, stderr

    def adb_runer_env(self):
        assert self.android_target, "android_target must be specified for adb runner"
        if self.platform == "SNPE":
            env = SNPEAndroidEnv(android_target=self.android_target, dev=self.dev).env
        else:
            raise ValueError(f"Unsupported platform {self.platform}")
        return env

    def adb_run(self):
        env = self.adb_runer_env()
        full_cmd = ""
        for key, value in env.items():
            full_cmd += f"export {key}={value} && "
        full_cmd += self.cmd

        return run_adb_command(
            full_cmd, self.android_target, shell_cmd=True, runs=self.runs, sleep=self.sleep, log_error=self.log_error
        )


class SNPESDKRunner(SDKRunner):
    def __init__(self, cmd: str, dev: bool = False, runs: int = 1, sleep: int = 0, log_error: bool = True):
        super().__init__("SNPE", cmd, dev, runs, sleep, log_error)


class QNNSDKRunner(SDKRunner):
    def __init__(self, cmd: str, dev: bool = False, runs: int = 1, sleep: int = 0, log_error: bool = True):
        super().__init__("QNN", cmd, dev, runs, sleep, log_error)
