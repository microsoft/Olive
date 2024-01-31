# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import os
import platform
from pathlib import Path


def onnx_qnn_config():
    try:
        qnn_env_path = Path(os.environ["QNN_ENV_PATH"]).resolve().as_posix()
    except KeyError:
        raise ValueError("QNN_ENV_PATH environment variable is not set") from None
    try:
        qnn_lib_path = Path(os.environ["QNN_LIB_PATH"]).resolve().as_posix()
    except KeyError as e:
        raise ValueError("QNN_LIB_PATH environment variable is not set") from e

    template_config_path = Path(__file__).parent / "mobilenet_config_template.json"

    config = None
    with template_config_path.open() as f:
        config = f.read()
        config = config.replace("<python-environment-path>", qnn_env_path)
        config = config.replace("<qnn-lib-path>", qnn_lib_path)

    with open("mobilenet_config.json", "w") as f:  # noqa: PTH123
        f.write(config)


def raw_qnn_config():
    # pylint: disable=redefined-outer-name

    with Path("./raw_qnn_sdk_template.json").open("r") as f:
        raw_qnn_config = json.load(f)

    sys_platform = platform.system()

    if sys_platform == "Linux":
        raw_qnn_config["passes"]["qnn_context_binary"] = {
            "type": "QNNContextBinaryGenerator",
            "config": {"backend": "libQnnHtp.so"},
        }
        raw_qnn_config["pass_flows"].append(["converter", "build_model_lib", "qnn_context_binary"])
        raw_qnn_config["passes"]["build_model_lib"]["config"]["lib_targets"] = "x86_64-linux-clang"
    elif sys_platform == "Windows":
        raw_qnn_config["passes"]["build_model_lib"]["config"]["lib_targets"] = "x86_64-windows-msvc"

    for metric_config in raw_qnn_config["evaluators"]["common_evaluator"]["metrics"]:
        if sys_platform == "Windows":
            metric_config["user_config"]["inference_settings"]["qnn"]["backend"] = "QnnCpu"
        elif sys_platform == "Linux":
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
    else:
        onnx_qnn_config()
