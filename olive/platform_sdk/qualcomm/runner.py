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
    ):
        self.platform = platform
        self.dev = dev
        self.runs = runs
        self.sleep = sleep

        if self.platform not in ("SNPE", "QNN"):
            raise ValueError(f"Unsupported platform {platform}")
        elif self.platform == "SNPE":
            self.env = SNPESDKEnv(dev=self.dev).env
        elif self.platform == "QNN":
            raise NotImplementedError("QNN not supported yet, coming soon!")

    def run(self, cmd: str):
        for run in range(self.runs):
            run_log_msg = "" if self.runs == 1 else f" (run {run + 1}/{self.runs})"
            logger.debug(f"Running {self.platform} command{run_log_msg}: {cmd}, with env: {self.env}")
            _, stdout, stderr = run_subprocess(cmd, self.env, check=True)
            if self.sleep > 0 and run < self.runs - 1:
                time.sleep(self.sleep)

        return stdout, stderr


class SNPESDKRunner(SDKRunner):
    def __init__(self, dev: bool = False, runs: int = 1, sleep: int = 0):
        super().__init__("SNPE", dev, runs, sleep)


class QNNSDKRunner(SDKRunner):
    def __init__(self, dev: bool = False, runs: int = 1, sleep: int = 0):
        super().__init__("QNN", dev, runs, sleep)
