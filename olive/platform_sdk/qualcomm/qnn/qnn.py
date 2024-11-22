# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from olive.common.config_utils import ConfigBase
from olive.common.constants import OS
from olive.constants import ModelFileFormat
from olive.platform_sdk.qualcomm.runner import QNNSDKRunner
from olive.platform_sdk.qualcomm.utils.input_list import get_input_ids

logger = logging.getLogger(__name__)


class QNNSessionOptions(ConfigBase):
    backend: str
    model_file_format: ModelFileFormat = None
    # backend: Path to a QNN backend to execute the model.
    # retrieve_context: ath to cached binary from which to load a saved
    # context from and execute graphs. --retrieve_context and
    # --model are mutually exclusive. Only one of the options
    # can be specified at a time.
    retrieve_context: str = None
    # accumulate_outputs: Accumulate the outputs from all the runs to a single folder.
    accumulate_outputs: bool = False
    # output_dir: Path to a directory where the output files will be
    # written. If not specified, the output files will be written
    # to the current working directory: ./output.
    output_dir: str = None
    # Enable profiling. Valid Values:
    #     1. basic:    captures execution and init time.
    #     2. detailed: in addition to basic, captures per Op timing
    #                 for execution, if a backend supports it.
    profiling_level: str = "basic"
    # extra_args: Extra arguments to be passed to the QNN runtime.
    # provide as a string. e.g. "--model_prefix QnnModel --debug".
    # More details on the arguments can be found in the QNN documentation.
    # https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/tools.html
    extra_args: str = None


class QNNInferenceSession:
    def __init__(
        self, model_path: str = None, io_config: Optional[Dict] = None, session_options: QNNSessionOptions = None
    ):
        self.model_path = model_path
        self.io_config = io_config
        self.session_options = session_options or QNNSessionOptions()
        self.model_file_format = self.session_options.model_file_format
        assert self.model_file_format in (
            ModelFileFormat.QNN_SERIALIZED_BIN,
            ModelFileFormat.QNN_LIB,
            ModelFileFormat.QNN_CPP,
        ), "None model format is not supported."

    def __call__(self, input_list: str, data_dir: str = None, runs: int = 1, sleep: int = 0) -> dict:
        # TODO(anyone): android case for inference
        return self.net_run(input_list, data_dir, runs, sleep)

    def net_run(self, input_list: str, data_dir: str = None, runs: int = 1, sleep: int = 0) -> dict:
        main_cmd = "qnn-net-run"
        # pylint: disable=consider-using-with
        tmp_dir = tempfile.TemporaryDirectory(dir=Path.cwd(), prefix="olive_tmp_qnn_")
        tmp_dir_path = Path(tmp_dir.name)
        runner = QNNSDKRunner(runs=runs, sleep=sleep)
        backend = runner.sdk_env.get_qnn_backend(self.session_options.backend)

        if self.model_file_format == ModelFileFormat.QNN_SERIALIZED_BIN:
            # only support serialized bin format
            input_model_arg = f"--retrieve_context {self.model_path}"
        else:
            input_model_arg = f"--model {self.model_path}"

        output_dir = Path(self.session_options.output_dir or "./output").resolve()
        cmd = [
            main_cmd,
            input_model_arg,
            f"--backend {backend}",
            f"--input_list {input_list}",
            f"--output_dir {tmp_dir_path}",
            f"--profiling_level {self.session_options.profiling_level}",
            self.session_options.extra_args or "",
        ]

        runner.run(" ".join(cmd))

        result_files = self._parse_model_output(output_dir, input_list, tmp_dir_path)
        result = [np.fromfile(rf[1], dtype=np.float32) for rf in result_files]
        latencies = {"init": [], "net_run": [], "net_run_throughput": []}
        for i in range(runs):
            latencies_item = self._parse_latency(i, tmp_dir_path)
            for key in latencies:  # noqa: PLC0206 # pylint: disable=consider-using-dict-items
                latencies[key].append(latencies_item[key])

        tmp_dir.cleanup()
        return {
            "output_dir": output_dir,
            "latencies": latencies,
            "result_files": result_files,
            "result": np.array(result),
        }

    def _parse_model_output(self, output_dir, input_list, tmp_dir_path):
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_result_dir = output_dir / "qnn-model-output"
            if not self.session_options.accumulate_outputs:
                shutil.rmtree(output_dir, ignore_errors=True)
            output_result_dir.mkdir(parents=True, exist_ok=True)

        result_files = []
        input_ids = get_input_ids(input_list)
        for member in tmp_dir_path.iterdir():
            if "Result_" not in member.name:
                continue
            result_idx = int(member.name.split("_")[1])
            input_id = input_ids[result_idx]
            # TODO(jiapli): what is the case when there are multiple outputs?
            raw_file = member / "output.raw"
            # copy the raw file to the workspace and rename it
            if output_dir is not None:
                output_file = output_dir / f"{input_id}.raw"
                if platform.system() == OS.WINDOWS:
                    output_file = output_dir / f"{input_id}.raw".replace(":", "_")
                raw_file.rename(output_file)
                result_files.append((input_id, output_file))
        return result_files

    def _parse_latency(self, run_id: int = None, tmp_dir_path: Path = None):
        # parse latency from qnn-profiling-data.log.csv
        # return a dict of latencies
        # {
        #     "init": 0, // qnn-net-run init time(ms)
        #     "net_run": 0, // per inference time(ms)
        #     "net_run_throughput": 0, // IPS, inference per second
        # }
        log_file_name = "qnn-profiling-data.log" if not run_id else f"qnn-profiling-data_{run_id}.log"
        cmd = [
            "qnn-profile-viewer",
            f"--input_log {tmp_dir_path / log_file_name}",
            f"--output {tmp_dir_path / 'qnn-profiling-data.log.csv'}",
        ]
        runner = QNNSDKRunner()
        runner.run(" ".join(cmd))

        latencies = {"init": 0, "net_run": 0, "net_run_throughput": 0}
        with (tmp_dir_path / "qnn-profiling-data.log.csv").open() as f:
            for raw_line in f:
                line = raw_line.lower()
                if "init" in line:
                    # us to s by / 1e6
                    latencies[",init,"] = float(line.split(",")[2]) / 1e6
                elif ",execute," in line:
                    latencies["net_run"] = float(line.split(",")[2]) / 1e6
                elif ",execute ips," in line:
                    latencies["net_run_throughput"] = float(line.split(",")[2])
        return latencies
