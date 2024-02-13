# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import json

from onnxruntime import __version__ as OrtVersion
from packaging import version

import olive.workflows.run as olive_run

SUPPORTED_WORKFLOWS = {
    "cpu_fp32": [["convert", "optimize_cpu", "perf_tuning"]],
    "cpu_int4": [["convert", "optimize_cpu", "blockwise_quant_int4", "perf_tuning"]],
    "cuda_fp16": [["convert", "optimize_cuda", "perf_tuning"]],
    "cuda_int4": [["convert", "optimize_cuda", "blockwise_quant_int4", "perf_tuning"]],
}

DEVICE_TO_EP = {
    "cpu": "CPUExecutionProvider",
    "gpu": "CUDAExecutionProvider",
}

def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Phi2 optimization")

    parser.add_argument(
        "--model_type", type=str, default="cpu_fp32", help="Choose from cpu_fp32, cpu_int4, cuda_fp16, cuda_int4"
    )

    return parser.parse_args(raw_args)


def main(raw_args=None):
    if version.parse(OrtVersion) < version.parse("1.17.0"):
        raise ValueError("Please use onnxruntime>=1.17.0 for phi2 optimization")

    args = get_args(raw_args)

    json_file_template = "phi2_optimize_template.json"
    with open(json_file_template) as f:  # noqa: PTH123
        template_json = json.load(f)

    # add pass flows
    model_type = str(args.model_type)
    template_json["pass_flows"] = SUPPORTED_WORKFLOWS[model_type]
    # remove unused passes
    used_passes = {pass_name for pass_flow in SUPPORTED_WORKFLOWS[model_type] for pass_name in pass_flow}
    for pass_name in list(template_json["passes"].keys()):
        if pass_name not in used_passes:
            del template_json["passes"][pass_name]
            continue

    if "cuda" in model_type:
        template_json["engine"]["execution_providers"] = ["CUDAExecutionProvider"]
    if "cpu" in model_type:
        template_json["engine"]["execution_providers"] = ["CPUExecutionProvider"]
    

    olive_run(template_json)  # pylint: disable=not-callable


if __name__ == "__main__":
    main()
