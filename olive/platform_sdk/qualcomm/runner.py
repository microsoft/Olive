# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import time

from olive.common.utils import run_subprocess
from olive.platform_sdk.qualcomm.qnn.env import QNNSDKEnv
from olive.platform_sdk.qualcomm.snpe.env import SNPESDKEnv

logger = logging.getLogger(__name__)


class SDKRunner:
    def __init__(
        self,
        platform,
        optional_local_run: bool = False,
        runs: int = 1,
        sleep: int = 0,
    ):
        self.platform = platform
        # optional_local_run: if the runner can be run locally
        self.optional_local_run = optional_local_run
        self.runs = runs
        self.sleep = sleep

        if self.platform not in ("SNPE", "QNN"):
            raise ValueError(f"Unsupported platform {platform}")
        elif self.platform == "SNPE":
            self.sdk_env = SNPESDKEnv(optional_local_run=self.optional_local_run)
        elif self.platform == "QNN":
            self.sdk_env = QNNSDKEnv(optional_local_run=self.optional_local_run)

    def run(self, cmd: str, use_olive_env: bool = True):
        env = self.sdk_env.env if use_olive_env else None
        for run in range(self.runs):
            run_log_msg = "" if self.runs == 1 else f" (run {run + 1}/{self.runs})"
            logger.debug(f"Running {self.platform} command{run_log_msg}: {cmd}, with env: {env}")
            _, stdout, stderr = run_subprocess(cmd, env, check=True)
            if self.sleep > 0 and run < self.runs - 1:
                time.sleep(self.sleep)

        return stdout, stderr


class SNPESDKRunner(SDKRunner):
    def __init__(self, optional_local_run: bool = False, runs: int = 1, sleep: int = 0):
        super().__init__("SNPE", optional_local_run, runs, sleep)


class QNNSDKRunner(SDKRunner):
    def __init__(self, optional_local_run: bool = False, runs: int = 1, sleep: int = 0):
        super().__init__("QNN", optional_local_run, runs, sleep)
