# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import platform
import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np

import olive.platform_sdk.qualcomm.snpe.utils.adb as adb_utils
from olive.common.config_utils import validate_enum
from olive.common.constants import OS
from olive.platform_sdk.qualcomm.constants import PerfProfile, ProfilingLevel, SNPEDevice
from olive.platform_sdk.qualcomm.runner import SNPESDKRunner as SNPERunner
from olive.platform_sdk.qualcomm.utils.input_list import get_input_ids, resolve_input_list

logger = logging.getLogger(__name__)


def init_snpe_net_adb(dlc_path: str, android_target: str, snpe_adb_prepared: bool = False):
    """Initialize a snpe inference session by pushing the DLC to the target Android device.

    dlc_path: The path to the DLC
    android_target: The target Android device
    snpe_adb_prepared: Whether the SNPE SDK has been pushed to the target Android device
    """
    adb_utils.run_adb_command("root", android_target)

    if not snpe_adb_prepared:
        adb_utils.prepare_snpe_adb(android_target)

    # Push the DLC to target
    target_ws = "/data/local/tmp/olive-snpe/ws"
    adb_utils.adb_push(dlc_path, target_ws, android_target, True)


def _snpe_net_run_adb(
    cmd: str,
    android_target: str,
    data_dir: str,
    runs: int = 1,
    sleep: int = 0,
    persist_ws: bool = False,
    initialized: bool = False,
):
    """Run snpe-net-run on the target Android device.

    cmd: snpe-net-run command with local paths to DLC, input list and output directory
    android_target: The target Android device.
    data_dir: The path to the data directory.
    persist_ws: Whether to persist the workspace on the target Android device.
    runs: The number of times to run snpe-net-run.
    sleep: The number of seconds to sleep between runs.
    initialized: Whether the snpe inference session has been initialized using init_snpe_net_adb.
    """
    # parse main command to get paths to DLC, input list and output directory
    dlc_path = cmd.split("--container")[1].split()[0]
    input_list = cmd.split("--input_list")[1].split()[0]
    output_dir = cmd.split("--output_dir")[1].split()[0]

    # target workspace and directories
    target_ws = "/data/local/tmp/olive-snpe/ws"
    target_output_dir = f"{target_ws}/output"
    target_data_dir = f"{target_ws}/data"

    try:
        adb_utils.run_adb_command("root", android_target)

        # create clean target workspace
        if not initialized:
            adb_utils.run_adb_command(f"rm -rf {target_ws}", android_target, True)
        else:
            adb_utils.run_adb_command(f"rm -rf {target_data_dir} {target_output_dir}", android_target, True)
        adb_utils.run_adb_command(f"mkdir -p {target_ws} {target_data_dir} {target_output_dir}", android_target, True)

        # create new input list with target paths
        target_input_list = resolve_input_list(target_data_dir, input_list, str(Path(output_dir).parent), data_dir)

        # push dlc, input list and data to target
        push_pairs = [(target_input_list, target_ws), (data_dir, target_data_dir)]
        if not initialized:
            push_pairs.append((dlc_path, target_ws))
        for src, dst in push_pairs:
            adb_utils.adb_push(src, dst, android_target)

        # create new command with target paths
        cmd = cmd.replace(dlc_path, f"{target_ws}/{Path(dlc_path).name}")
        cmd = cmd.replace(input_list, f"{target_ws}/{Path(target_input_list).name}")
        cmd = cmd.replace(output_dir, target_output_dir)

        # run snpe-net-run on target
        push_snpe = not initialized
        adb_utils.run_snpe_adb_command(cmd, android_target, push_snpe, runs=runs, sleep=sleep)

        # replace ":" in output filenames with "_" if on windows before pulling
        if platform.system() == OS.WINDOWS:
            rename_cmd = (
                f"cd {target_output_dir} &&"
                " find -name '*:*' -exec sh -c 'for x; do mv $x $(echo $x | sed \"s/:/_/g\"); done' _ {} +"
            )
            adb_utils.run_adb_command(rename_cmd, android_target, True)

        # pull output from target
        adb_utils.run_adb_command(f"pull {target_output_dir}/. {Path(output_dir).as_posix()}", android_target)
    finally:
        if not persist_ws:
            # clean target workspace
            adb_utils.run_adb_command(f"rm -rf {target_ws}", android_target, True)


def snpe_net_run(
    dlc_path: str,
    input_list: str,
    data_dir: str,
    output_names: List[str],
    output_shapes: List[List[int]],
    runs: int = 1,
    sleep: int = 0,
    device: SNPEDevice = "cpu",
    android_target: str = None,
    set_output_tensors: bool = False,
    perf_profile: PerfProfile = None,
    profiling_level: ProfilingLevel = None,
    inferences_per_duration: int = None,
    duration: int = None,
    enable_cpu_fallback: bool = False,
    extra_args: str = None,
    workspace: str = None,
    accumulate_outputs: str = False,
    return_numpy_results: bool = False,
    android_persist_ws: bool = False,
    android_initialized: bool = False,
) -> dict:
    """Run snpe-net-run on the given DLC and input list.

    dlc_path: The path to the DLC.
    input_list: The path to the input list.
    data_dir: The path to the data directory.
    output_names: The names of the output tensors.
    output_shapes: The shapes of the output tensors.
    runs: The number of times to run snpe-net-run.
    sleep: The number of seconds to sleep between runs.
    device: The device to run on.
    android_target: The android target to run on. If None, run on host.
    set_output_tensors: Whether to set the output tensors.
    perf_profile: The performance profile to use.
    profiling_level: The profiling level to use.
    inferences_per_duration: The number of inferences per duration.
    duration: The duration.
    enable_cpu_fallback: Whether to enable CPU fallback.
    extra_args: Extra arguments to pass to snpe-net-run.
    workspace: The workspace to copy results and diagnostics to.
    accumulate_outputs: Whether to accumulate outputs. If False, workspace/snpe-output will be cleared before each run.
    return_numpy_results: Whether to return the results as numpy arrays.
    android_persist_ws: Whether to persist the workspace on android.
    android_initialized: Whether the inference session has already been initialized on android using init_snpe_net_adb.
    """
    tmp_dir = tempfile.TemporaryDirectory(prefix="olive_tmp_")  # pylint: disable=consider-using-with
    tmp_dir_path = Path(tmp_dir.name).resolve()

    # Create the snpe-net-run command
    cmd = f"snpe-net-run --container {dlc_path} --input_list {input_list} --output_dir {str(tmp_dir_path)}"
    if device != SNPEDevice.CPU:
        device = validate_enum(SNPEDevice, device)
        cmd += f" --use_{device}"
    if perf_profile is not None:
        perf_profile = validate_enum(PerfProfile, perf_profile)
        cmd += f" --perf_profile {perf_profile}"
    if profiling_level is not None:
        profiling_level = validate_enum(ProfilingLevel, profiling_level)
        cmd += f" --profiling_level {profiling_level}"
    if inferences_per_duration is not None:
        cmd += f" --inferences_per_duration {inferences_per_duration}"
    if duration is not None:
        cmd += f" --duration {duration}"
    if enable_cpu_fallback:
        cmd += " --enable_cpu_fallback"
    if extra_args is not None:
        cmd += " " + extra_args
    if set_output_tensors:
        cmd += f" --set_output_tensors={','.join(output_names)}"

    if android_target is not None:
        _snpe_net_run_adb(
            cmd,
            android_target,
            data_dir,
            runs=runs,
            sleep=sleep,
            persist_ws=android_persist_ws,
            initialized=android_initialized,
        )
    else:
        SNPERunner(runs=runs, sleep=sleep).run(cmd)

    output_dir = None
    if workspace is not None:
        # Create the workspace directory if it doesn't exist
        Path(workspace).mkdir(parents=True, exist_ok=True)
        # snpe output directory
        output_dir = Path(workspace) / "snpe-output"
        if not accumulate_outputs:
            shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)
    elif accumulate_outputs:
        logger.warning("accumulate_outputs is set to True, but no workspace is provided. Ignoring accumulate_outputs.")

    # input ids for each sample in the input list
    input_ids = get_input_ids(input_list)

    # get the delimiter for the output files
    delimiter = None
    if platform.system() == OS.LINUX:
        delimiter = ":"
    elif platform.system() == OS.WINDOWS:
        delimiter = "_"

    # dictionary to store the results as numpy arrays
    results = {}
    result_files = {}
    for output_name in output_names:
        results[output_name] = []
        result_files[output_name] = []

    for member in tmp_dir_path.iterdir():
        if "Result_" not in member.name:
            continue
        result_idx = int(member.name.split("_")[1])
        # Note if we are doing the perf test with inferences_per_duration then the Result_N
        # folders do not match the inputs, instead we get one Result_N folder per duration.
        inferences_per_duration = 0 if inferences_per_duration is None else inferences_per_duration
        input_id = f"result_{result_idx}" if inferences_per_duration > 0 else input_ids[result_idx]
        for output_name, output_shape in zip(output_names, output_shapes):
            # output names for dlcs converted from tensorflow models contain ":"
            # try adding `:0` or `_0` to output file name in case original model was tensorflow and
            # user provided original output names
            output_file_name = f"{output_name}{delimiter}0.raw"
            if not (member / output_file_name).exists():
                # `:0` is already in the output name or source model was not tensorflow
                output_file_name = f"{output_name}.raw"
            if platform.system() == OS.WINDOWS:
                # replace ":" with "_" in the file name.
                output_file_name = output_file_name.replace(":", "_")
            raw_file = member / output_file_name

            # copy the raw file to the workspace and rename it
            if output_dir is not None:
                output_file = output_dir / f"{input_id}.{output_name}.raw"
                if platform.system() == OS.WINDOWS:
                    # replace ":" with "_" in the file name.
                    output_file = output_dir / f"{input_id}.{output_name}.raw".replace(":", "_")
                if len(output_names) == 1:
                    # no need to encode the output name in the file name
                    output_file = output_dir / f"{input_id}.raw"
                raw_file.rename(output_file)
                result_files[output_name].append((result_idx, output_file))

            # read the raw file and convert it to a numpy array
            if return_numpy_results:
                float_array = np.fromfile(raw_file, dtype=np.float32)
                float_array = float_array.reshape(output_shape)
                results[output_name].append((result_idx, float_array))

    if return_numpy_results:
        # sort the results by the input id and stack them into a single numpy array
        for output_name in output_names:
            results[output_name].sort(key=lambda x: x[0])
            results[output_name] = [x[1] for x in results[output_name]]

    if workspace is not None:
        # sort the result files by the input id
        for output_name in output_names:
            result_files[output_name].sort(key=lambda x: x[0])
            result_files[output_name] = [x[1] for x in result_files[output_name]]

    latencies = {"init": [], "total_inference_time": []}
    diag_log_files = []
    for snpe_diag_log in tmp_dir_path.iterdir():
        filename_match_group = re.match(r"SNPEDiag_(\d+)\.log", snpe_diag_log.name)
        if not filename_match_group:
            continue
        diag_log_files.append(snpe_diag_log)
        run_id = filename_match_group.group(1)

        snpe_diag_csv = tmp_dir_path / f"{snpe_diag_log.stem}.csv"
        cmd = f"snpe-diagview --input_log {snpe_diag_log} --output {snpe_diag_csv}"
        SNPERunner().run(cmd)

        diag_log = {"init": None, "avg_total_inference_time": None}
        with snpe_diag_csv.open() as f:
            for line in f:
                message_name = line.split(",")[1].lower()
                message_value = line.split(",")[3]
                if message_name in ["init", "avg_total_inference_time"] and diag_log[message_name] is None:
                    diag_log[message_name] = float(message_value) / 1000000
        latencies["init"].append(diag_log["init"])
        latencies["total_inference_time"].append(diag_log["avg_total_inference_time"])

        if output_dir is not None:
            snpe_diag_csv.rename(output_dir / f"perf_results_{run_id}.csv")

    if len(diag_log_files) != runs:
        # sometimes the SNPEDiag log files are not created by snpe-net-run successfully
        # also because the SNPEDiag logs are only used for latency measurement, to unblock the
        # main processing, we only throw a error log here for awareness.
        logger.error(
            "Number of SNPEDiag log files does not match the number of runs (%d). The diag files are: %s",
            runs,
            diag_log_files,
        )

    # explicitly delete the tmp directory just to be safe
    tmp_dir.cleanup()

    output_dict = {"latencies": latencies}
    if return_numpy_results:
        output_dict["results"] = [{k: v[i] for k, v in results.items()} for i in range(len(results[output_names[0]]))]
    if output_dir is not None:
        output_dict["output_dir"] = str(output_dir)
        output_dict["result_files"] = result_files

    return output_dict


def _snpe_throughput_net_run_adb(
    cmd: str, android_target: str, data_dir: str, persist_ws: bool = False, initialized: bool = False
) -> Tuple[str, str]:
    """Run snpe-throughput-net-run on the target Android device.

    cmd: snpe-throughput-net-run command with local paths to the dlc and input raw files.
    android_target: target Android device.
    data_dir: the path to the data directory.
    persist_ws: Whether to persist the workspace on the target device.
    Whether the snpe inference session has been initialized using init_snpe_net_adb.
    """
    dlc_path = cmd.split("--container")[1].split()[0]
    input_raw = cmd.split("--input_raw")[1].split()[0]

    target_ws = "/data/local/tmp/olive-snpe/ws"
    target_data_dir = f"{target_ws}/data"

    try:
        adb_utils.run_adb_command("root", android_target)

        # create clean target workspace
        if not initialized:
            adb_utils.run_adb_command(f"rm -rf {target_ws}", android_target, True)
        else:
            adb_utils.run_adb_command(f"rm -rf {target_data_dir}", android_target, True)
        adb_utils.run_adb_command(f"mkdir -p {target_ws} {target_data_dir}", android_target, True)

        # create input_raw with target paths
        inputs = input_raw.split(",")
        target_inputs = []
        for input_item in inputs:
            target_input = Path(input_item).resolve().relative_to(Path(data_dir).resolve())
            target_input = (Path(target_data_dir) / target_input).as_posix()
            target_inputs.append(target_input)
        target_input_raw = ",".join(target_inputs)

        # push dlc and input raw files to target
        push_pairs = []
        if not initialized:
            push_pairs = [(dlc_path, target_ws)]
        for input_item, target_input in zip(inputs, target_inputs):
            push_pairs.append((input_item, Path(target_input).parent.as_posix()))
        for src, dst in push_pairs:
            adb_utils.adb_push(src, dst, android_target)

        # create new command with target paths
        cmd = cmd.replace(dlc_path, f"{target_ws}/{Path(dlc_path).name}")
        cmd = cmd.replace(input_raw, target_input_raw)

        push_snpe = not initialized
        stdout, stderr = adb_utils.run_snpe_adb_command(cmd, android_target, push_snpe=push_snpe)
    finally:
        if not persist_ws:
            adb_utils.run_adb_command(f"rm -rf {target_ws}", android_target, True)

    return stdout, stderr


def snpe_throughput_net_run(
    dlc_path: str,
    duration: int,
    input_list: str,
    data_dir: str,
    device: SNPEDevice = "cpu",
    android_target: str = None,
    perf_profile: PerfProfile = None,
    enable_cpu_fallback: bool = False,
    android_persist_ws: bool = False,
    android_initialized: bool = False,
) -> float:
    """Run snpe-throughput-net-run on the given DLC and input list for the given duration.

    Returns the throughput value.

    dlc_path: The path to the DLC.
    duration: The duration.
    input_list: The path to the input list.
    data_dir: The path to the data directory.
    device: The device to run on.
    android_target: The android target to run on. If None, run on host.
    perf_profile: The performance profile to use.
    enable_cpu_fallback: Whether to enable CPU fallback.
    android_persist_ws: Whether to persist the workspace on android.
    android_initialized: Whether the inference session has already been initialized on android using
        init_snpe_net_adb.
    """
    cmd = f"snpe-throughput-net-run --container {dlc_path} --duration {duration} --use_{device}"

    input_raw = ""
    first = ""
    with Path(input_list).open() as f:
        for line in f:
            if line.startswith(("#", "%")):
                continue
            else:
                first = line.strip()
                break

        if ":=" in first:
            inputs = [(x.split(":=")[0], x.split(":=")[1]) for x in first.split()]
            inputs = sorted(inputs, key=lambda x: x[0])
            input_raw = ",".join([x[1] for x in inputs])
        else:
            input_raw = first
    cmd += f" --input_raw {input_raw}"

    if perf_profile is not None:
        try:
            perf_profile = PerfProfile(perf_profile)
        except ValueError:
            raise ValueError(
                f"Invalid perf profile '{perf_profile}'. Valid perf profiles are {[p.value for p in PerfProfile]}"
            ) from None
        cmd += f" --perf_profile {perf_profile}"
    if enable_cpu_fallback:
        cmd += " --enable_cpu_fallback"

    if android_target is not None:
        stdout, _ = _snpe_throughput_net_run_adb(
            cmd, android_target, data_dir, persist_ws=android_persist_ws, initialized=android_initialized
        )
    else:
        stdout, _ = SNPERunner().run(cmd)
    return float(stdout.split("Total throughput: ")[1].split(" ")[0])
