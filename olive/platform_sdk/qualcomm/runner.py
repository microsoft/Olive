# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
import shlex
import shutil
import time
from pathlib import Path
from typing import List, Union

from olive.common.constants import OS
from olive.common.utils import run_subprocess
from olive.platform_sdk.qualcomm.qnn.env import QNNSDKEnv
from olive.platform_sdk.qualcomm.snpe.env import SNPESDKEnv

logger = logging.getLogger(__name__)

USE_OLIVE_ENV = "USE_OLIVE_ENV"
USE_OLIVE_ENV_DEFAULT_VALUE = "1"


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

    def _use_olive_env(self):
        return os.environ.get(USE_OLIVE_ENV, USE_OLIVE_ENV_DEFAULT_VALUE) == USE_OLIVE_ENV_DEFAULT_VALUE

    def _resolve_cmd(self, cmd: Union[str, List[str]]) -> List[str]:
        # TODO(trajep): use list instead of string to avoid shlex.split error in non-posix mode
        import platform

        if isinstance(cmd, str):
            cmd_list = shlex.split(cmd, posix=(platform.system() != OS.WINDOWS))
        else:
            cmd_list = cmd

        if platform.system() == OS.WINDOWS and cmd_list[0].startswith(("snpe-", "qnn-")):
            logger.debug("Resolving command %s on Windows.", cmd_list)
            cmd_dir = Path(self.sdk_env.sdk_root_path) / "bin" / self.sdk_env.target_arch
            cmd_name = cmd_list[0]
            cmd_full_path = cmd_dir / cmd_name
            if cmd_full_path.with_suffix(".exe").exists():
                cmd_full_path = cmd_full_path.with_suffix(".exe")
            if cmd_full_path.exists():
                cmd_list[0] = str(cmd_full_path)
                try:
                    with cmd_full_path.open() as f:
                        first_line = f.readline()
                        if "python" in first_line:
                            cmd_list.insert(0, "python")
                except UnicodeDecodeError as e:
                    logger.warning(
                        "Failed to read the first line of %s: %s. Will ignore to wrap it with python.", cmd_name, e
                    )

        path_env = self.sdk_env.env.get("PATH") if self._use_olive_env() else None
        cmd_list[0] = shutil.which(cmd_list[0], path=path_env) or cmd_list[0]
        return cmd_list

    def run(self, cmd: str):
        env = self.sdk_env.env if self._use_olive_env() else None
        cmd = self._resolve_cmd(cmd)

        for run in range(self.runs):
            run_log_msg = "" if self.runs == 1 else f" (run {run + 1}/{self.runs})"
            logger.debug("Running %s command%s: ", self.platform, run_log_msg)
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
