# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import platform
from collections import OrderedDict
from pathlib import Path

from olive.common.constants import OS


def raw_qnn_config(mode: str):
    # pylint: disable=redefined-outer-name

    with Path("raw_qnn_sdk_template.json").open("r") as f:
        raw_qnn_config = json.load(f)

    sys_platform = platform.system()

    if mode == "convert":
        used_passes = {"converter", "build_model_lib"}
    elif mode == "quantize":
        used_passes = {"quantization", "build_model_lib"}
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if sys_platform == OS.LINUX:
        used_passes.add("qnn_context_binary")
        raw_qnn_config["passes"]["build_model_lib"]["lib_targets"] = "x86_64-linux-clang"
    elif sys_platform == OS.WINDOWS:
        raw_qnn_config["passes"]["build_model_lib"]["lib_targets"] = "x86_64-windows-msvc"

    for metric_config in raw_qnn_config["evaluators"]["common_evaluator"]["metrics"]:
        if sys_platform == OS.WINDOWS:
            metric_config["user_config"]["inference_settings"]["qnn"]["backend"] = "QnnCpu"
        elif sys_platform == OS.LINUX:
            metric_config["user_config"]["inference_settings"]["qnn"]["backend"] = "libQnnCpu"

    raw_qnn_config["passes"] = OrderedDict([(k, v) for k, v in raw_qnn_config["passes"].items() if k in used_passes])

    with Path("raw_qnn_sdk_config.json").open("w") as f:
        json_str = json.dumps(raw_qnn_config, indent=4)
        input_file_path = Path("data/eval/input_order.txt").resolve().as_posix()
        f.write(json_str.replace("<input_list.txt>", str(input_file_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["convert", "quantize"],
        help="Mode selection",
    )
    args = parser.parse_args()
    raw_qnn_config(args.mode)
