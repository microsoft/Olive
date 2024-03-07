# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import json
import platform
from pathlib import Path

from onnxruntime import __version__ as OrtVersion
from packaging import version

import olive.workflows.run as olive_run

SUPPORTED_WORKFLOWS = {
    "cpu_fp32": [["convert", "optimize_cpu", "perf_tuning"]],
    "cpu_int4": [["convert", "optimize_cpu", "blockwise_quant_int4", "perf_tuning"]],
    "cuda_fp16": [["convert", "optimize_cuda", "perf_tuning"]],
    "cuda_int4": [["convert", "optimize_cuda", "blockwise_quant_int4", "perf_tuning"]],
}
SUPPORTED_INFERENCE_CONFIG = {
    "cpu_fp32": {
        "use_buffer_share": False,
        # -1 to use CPU
        "device_id": -1,
        "use_fp16": False,
        "use_step": platform.system() == "Linux",
    },
    "cpu_int4": {
        "use_buffer_share": False,
        "device_id": -1,
        "use_fp16": False,
        "use_step": platform.system() == "Linux",
    },
    "cuda_fp16": {
        "use_buffer_share": False,
        "device_id": 0,
        "use_fp16": True,
        "use_step": platform.system() == "Linux",
    },
    "cuda_int4": {
        "use_buffer_share": False,
        "device_id": 0,
        "use_fp16": True,
        "use_step": platform.system() == "Linux",
    },
}

DEVICE_TO_EP = {
    "cpu": "CPUExecutionProvider",
    "gpu": "CUDAExecutionProvider",
}


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Phi2 optimization")

    parser.add_argument(
        "--model_type",
        type=str,
        default="cpu_fp32",
        help="Choose from cpu_fp32, cpu_int4, cuda_fp16, cuda_int4",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run inference with optimized model",
    )
    parser.add_argument(
        "--optimum_optimization",
        action="store_true",
        help="Run inference with optimized model",
    )
    parser.add_argument(
        "--prompt",
        nargs="*",
        type=str,
        default=["Write a function to print 1 to n", "Write a extremely long story starting with once upon a time"],
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Max length for generation",
    )

    return parser.parse_args(raw_args)


def get_output_model_path(footprints):
    # only one model output in phi2 optimization
    for footprint in footprints.values():
        for model_id in footprint.nodes:
            model_path = Path(footprint.get_model_path(model_id)) / "model.onnx"
            break
    return model_path


def main(raw_args=None):
    args = get_args(raw_args)

    if not args.optimum_optimization and version.parse(OrtVersion) < version.parse("1.18.0"):
        # Check if onnxruntime version is supported
        # in linux, it requires the
        # 1. model_type as `phi`
        # 2. "optimization_options": {"attention_op_type": "MultiHeadAttention"}
        # in windows, it requires the
        # 1. model_type as `gpt2`
        # 2. "optimization_options": {"attention_op_type": "MultiHeadAttention"}
        # and `phi` and `MultiHeadAttention` requires ort-nightly version >= 1.18.0
        raise ValueError(
            "Please use onnxruntime>=1.18.0 for phi2 optimization in Linux, you can refer to "
            "https://onnxruntime.ai/docs/install/#inference-install-table-for-all-languages "
            "for ort-nightly installation. If you are optimizing phi2 model in GPU, only cuda11 "
            "is supported in onnxruntime>=1.18.0"
        )

    json_file_template = "phi2_optimize_template.json"
    with open(json_file_template) as f:
        template_json = json.load(f)

    if platform.system() == "Windows":
        legacy_optimization_setting(template_json)

    # add pass flows
    model_type = str(args.model_type)
    template_json["pass_flows"] = SUPPORTED_WORKFLOWS[model_type]
    if args.optimum_optimization:
        # if args.model_type in ("cpu_int4", "cuda_int4"):
        #     raise ValueError("Int4 optimization is not supported in phi2 optimum optimization")
        # set evaluator as None:
        template_json["engine"]["evaluate_input_model"] = False
        template_json["engine"]["evaluator"] = None
        legacy_optimization_setting(template_json)
        for pass_flow in template_json["pass_flows"]:
            pass_flow[0] = "optimum_convert"
            if "perf_tuning" in pass_flow:
                pass_flow.remove("perf_tuning")

    # remove unused passes
    used_passes = {pass_name for pass_flow in SUPPORTED_WORKFLOWS[model_type] for pass_name in pass_flow}
    for pass_name in list(template_json["passes"].keys()):
        if pass_name not in used_passes:
            del template_json["passes"][pass_name]
            continue

    if "cuda" in model_type:
        template_json["system"]["local_system"]["config"]["accelerators"][0]["execution_providers"] = [
            "CUDAExecutionProvider"
        ]
    if "cpu" in model_type:
        template_json["system"]["local_system"]["config"]["accelerators"][0]["execution_providers"] = [
            "CPUExecutionProvider"
        ]

    with open("phi2_optimize.json", "w") as f:
        json.dump(template_json, f, indent=4)

    footprints = olive_run(template_json)  # pylint: disable=not-callable
    output_model_path = get_output_model_path(footprints)
    if args.inference and model_type in SUPPORTED_INFERENCE_CONFIG:
        from generate import run as generate_run

        for text in generate_run(
            args.prompt,
            output_model_path,
            **SUPPORTED_INFERENCE_CONFIG[model_type],
            use_optimum=args.optimum_optimization,
            max_length=args.max_length,
        ):
            print(f"Generation output: {text}")  # noqa: T201
            print("*" * 50)  # noqa: T201


def legacy_optimization_setting(config):
    config["passes"]["convert"]["config"]["use_dynamo_exporter"] = False
    config["passes"]["convert"]["config"]["target_opset"] = 17
    config["passes"]["optimize_cpu"]["config"]["model_type"] = "gpt2"
    config["passes"]["optimize_cuda"]["config"]["model_type"] = "gpt2"


if __name__ == "__main__":
    main()
