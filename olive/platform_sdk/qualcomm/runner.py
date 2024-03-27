# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import shlex
import shutil
import time
from pathlib import Path

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
        use_olive_env: bool = True,
    ):
        self.platform = platform
        # use_dev_tools: whether use dev tools under the sdk
        self.use_dev_tools = use_dev_tools
        self.runs = runs
        self.sleep = sleep
        self.use_olive_env = use_olive_env

        if self.platform not in ("SNPE", "QNN"):
            raise ValueError(f"Unsupported platform {platform}")
        elif self.platform == "SNPE":
            self.sdk_env = SNPESDKEnv(use_dev_tools=self.use_dev_tools)
        elif self.platform == "QNN":
            self.sdk_env = QNNSDKEnv(use_dev_tools=self.use_dev_tools)

    def _resolve_cmd(self, cmd: str, env: dict):
        import platform

        if platform.system() == "Windows" and cmd.startswith(("snpe-", "qnn-")):
            logger.debug("Resolving command %s on Windows.", cmd)
            cmd_dir = Path(self.sdk_env.sdk_root_path) / "bin" / self.sdk_env.target_arch
            cmd_name = cmd.split(" ")[0]
            cmd_full_path = cmd_dir / cmd_name
            if cmd_full_path.with_suffix(".exe").exists():
                cmd_full_path = cmd_full_path.with_suffix(".exe")
            if cmd_full_path.exists():
                cmd = str(cmd_full_path) + cmd[len(cmd_name) :]
                try:
                    with cmd_full_path.open() as f:
                        first_line = f.readline()
                        if "python" in first_line:
                            cmd = f"python {cmd}"
                except UnicodeDecodeError as e:
                    logger.warning(
                        "Failed to read the first line of %s: %s. Will ignore to wrap it with python.", cmd_name, e
                    )
        if isinstance(cmd, str):
            cmd = shlex.split(cmd, posix=(platform.system() != "Windows"))
            cmd[0] = shutil.which(cmd[0], path=env.get("PATH")) or cmd[0]
        return cmd

    def run(self, cmd: str):
        env = self.sdk_env.env if self.use_olive_env else {}
        cmd = self._resolve_cmd(cmd, env)

        for run in range(self.runs):
            run_log_msg = "" if self.runs == 1 else f" (run {run + 1}/{self.runs})"
            logger.debug("Running %s command%s: ", self.platform, run_log_msg)
            _, stdout, stderr = run_subprocess(cmd, env, check=True)
            if self.sleep > 0 and run < self.runs - 1:
                time.sleep(self.sleep)

        return stdout, stderr


class SNPESDKRunner(SDKRunner):
    def __init__(self, use_dev_tools: bool = False, runs: int = 1, sleep: int = 0, use_olive_env: bool = True):
        super().__init__("SNPE", use_dev_tools, runs, sleep, use_olive_env)


class QNNSDKRunner(SDKRunner):
    def __init__(self, use_dev_tools: bool = False, runs: int = 1, sleep: int = 0, use_olive_env: bool = True):
        super().__init__("QNN", use_dev_tools, runs, sleep, use_olive_env)
