# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import time

from olive.common.utils import run_subprocess
from olive.platform_sdk.qualcomm.snpe.env import SNPESDKEnv

logger = logging.getLogger(__name__)


class SDKRunner:
    def __init__(
        self,
        platform,
        dev: bool = False,
        runs: int = 1,
        sleep: int = 0,
        log_error: bool = True,
    ):
        self.platform = platform
        self.dev = dev
        self.runs = runs
        self.sleep = sleep
        self.log_error = log_error

        if self.platform not in ("SNPE", "QNN"):
            raise ValueError(f"Unsupported platform {platform}")
        elif self.platform == "SNPE":
            self.env = SNPESDKEnv(dev=self.dev).env
        elif self.platform == "QNN":
            raise NotImplementedError("QNN not supported yet, coming soon!")

    def run(self, cmd: str):
        for run in range(self.runs):
            run_log_msg = "" if self.runs == 1 else f" (run {run + 1}/{self.runs})"
            logger.debug(f"Running {self.platform} command{run_log_msg}: {cmd}")
            returncode, stdout, stderr = run_subprocess(cmd, self.env)
            logger.debug(f"Return code: {returncode} \n Stdout: {stdout} \n Stderr: {stderr}")
            if returncode != 0:
                break
            if self.sleep > 0 and run < self.runs - 1:
                time.sleep(self.sleep)

        if returncode != 0:
            error_msg = [
                "Error running {self.platform} command.",
                f"Command: {cmd}",
                f"Return code: {returncode}Stdout: {stdout}" if stdout else "",
                f"Stderr: {stderr}" if stderr else "",
                f"ENV: {self.env}",
            ]
            if self.log_error:
                logger.error("\n".join(error_msg))
            raise RuntimeError(error_msg)
        return stdout, stderr


class SNPESDKRunner(SDKRunner):
    def __init__(self, dev: bool = False, runs: int = 1, sleep: int = 0, log_error: bool = True):
        super().__init__("SNPE", dev, runs, sleep, log_error)


class QNNSDKRunner(SDKRunner):
    def __init__(self, dev: bool = False, runs: int = 1, sleep: int = 0, log_error: bool = True):
        super().__init__("QNN", dev, runs, sleep, log_error)
