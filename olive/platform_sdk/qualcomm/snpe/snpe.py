# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging

from olive.common.config_utils import ConfigBase
from olive.platform_sdk.qualcomm.constants import PerfProfile, ProfilingLevel, SNPEDevice
from olive.platform_sdk.qualcomm.snpe.tools.dev import get_dlc_metrics
from olive.platform_sdk.qualcomm.snpe.tools.inference import init_snpe_net_adb, snpe_net_run, snpe_throughput_net_run

logger = logging.getLogger(__name__)


class SNPESessionOptions(ConfigBase):
    device: SNPEDevice = "cpu"
    android_target: str = None
    set_output_tensors: bool = False
    perf_profile: PerfProfile = None
    profiling_level: ProfilingLevel = None
    inferences_per_duration: int = None
    duration: int = None
    enable_cpu_fallback: bool = False
    extra_args: str = None
    workspace: str = None
    accumulate_outputs: str = False
    return_numpy_results: bool = False
    snpe_adb_prepared: bool = False

    def get_run_options(self):
        return {
            "android_target": self.android_target,
            "device": self.device,
            "set_output_tensors": self.set_output_tensors,
            "perf_profile": self.perf_profile,
            "profiling_level": self.profiling_level,
            "inferences_per_duration": self.inferences_per_duration,
            "duration": self.duration,
            "enable_cpu_fallback": self.enable_cpu_fallback,
            "extra_args": self.extra_args,
            "workspace": self.workspace,
            "accumulate_outputs": self.accumulate_outputs,
            "return_numpy_results": self.return_numpy_results,
        }

    def get_throughput_options(self):
        return {
            "android_target": self.android_target,
            "device": self.device,
            "perf_profile": self.perf_profile,
            "enable_cpu_fallback": self.enable_cpu_fallback,
        }


class SNPEInferenceSession:
    def __init__(self, model_path: str, io_config: dict, session_options: SNPESessionOptions = None):
        self.model_path = model_path
        self.io_config = io_config
        self.session_options = session_options or SNPESessionOptions()
        if self.session_options.android_target is not None:
            init_snpe_net_adb(
                self.model_path, self.session_options.android_target, self.session_options.snpe_adb_prepared
            )

    def __call__(self, input_list: str, data_dir: str = None, runs: int = 1, sleep: int = 0) -> dict:
        return self.net_run(input_list, data_dir, runs, sleep)

    def net_run(self, input_list: str, data_dir: str = None, runs: int = 1, sleep: int = 0) -> dict:
        if self.session_options.android_target is not None and data_dir is None:
            raise ValueError("Data directory must be specified when using Android target")

        return snpe_net_run(
            self.model_path,
            input_list,
            data_dir,
            self.io_config["output_names"],
            self.io_config["output_shapes"],
            runs,
            sleep,
            android_persist_ws=True,
            android_initialized=True,
            **self.session_options.get_run_options(),
        )

    def throughput(self, duration: int, input_list: str, data_dir: str = None):
        if self.session_options.android_target is not None and data_dir is None:
            raise ValueError("Data directory must be specified when using Android target")
        return snpe_throughput_net_run(
            self.model_path,
            duration,
            input_list,
            data_dir,
            android_persist_ws=True,
            android_initialized=True,
            **self.session_options.get_throughput_options(),
        )

    def get_dlc_metrics(self):
        return get_dlc_metrics(self.model_path)
