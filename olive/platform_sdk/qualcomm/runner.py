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
        use_dev_tools: bool = False,
        runs: int = 1,
        sleep: int = 0,
    ):
        self.platform = platform
        # use_dev_tools: whether use dev tools under the sdk
        self.use_dev_tools = use_dev_tools
        self.runs = runs
        self.sleep = sleep

        if self.platform not in ("SNPE", "QNN"):
            raise ValueError(f"Unsupported platform {platform}")
        elif self.platform == "SNPE":
            self.sdk_env = SNPESDKEnv(use_dev_tools=self.use_dev_tools)
        elif self.platform == "QNN":
            self.sdk_env = QNNSDKEnv(use_dev_tools=self.use_dev_tools)

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
    def __init__(self, use_dev_tools: bool = False, runs: int = 1, sleep: int = 0):
        super().__init__("SNPE", use_dev_tools, runs, sleep)


class QNNSDKRunner(SDKRunner):
    def __init__(self, use_dev_tools: bool = False, runs: int = 1, sleep: int = 0):
        super().__init__("QNN", use_dev_tools, runs, sleep)
