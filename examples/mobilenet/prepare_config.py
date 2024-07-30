# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import platform
from pathlib import Path

from olive.common.constants import OS


def raw_qnn_config():
    # pylint: disable=redefined-outer-name

    with Path("./raw_qnn_sdk_template.json").open("r") as f:
        raw_qnn_config = json.load(f)

    sys_platform = platform.system()

    if sys_platform == OS.LINUX:
        raw_qnn_config["passes"]["qnn_context_binary"] = {
            "type": "QNNContextBinaryGenerator",
            "backend": "libQnnHtp.so",
        }
        raw_qnn_config["pass_flows"].append(["converter", "build_model_lib", "qnn_context_binary"])
        raw_qnn_config["passes"]["build_model_lib"]["lib_targets"] = "x86_64-linux-clang"
    elif sys_platform == OS.WINDOWS:
        raw_qnn_config["passes"]["build_model_lib"]["lib_targets"] = "x86_64-windows-msvc"

    for metric_config in raw_qnn_config["evaluators"]["common_evaluator"]["metrics"]:
        if sys_platform == OS.WINDOWS:
            metric_config["user_config"]["inference_settings"]["qnn"]["backend"] = "QnnCpu"
        elif sys_platform == OS.LINUX:
            metric_config["user_config"]["inference_settings"]["qnn"]["backend"] = "libQnnCpu"

    with Path("raw_qnn_sdk_config.json").open("w") as f:
        json_str = json.dumps(raw_qnn_config, indent=4)
        input_file_path = Path("./data/eval/input_order.txt").resolve().as_posix()
        f.write(json_str.replace("<input_list.txt>", str(input_file_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_raw_qnn_sdk",
        action="store_true",
        help="If set, use the raw qnn sdk instead of the qnn EP",
    )
    args = parser.parse_args()
    if args.use_raw_qnn_sdk:
        raw_qnn_config()
