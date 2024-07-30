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

from olive.common.constants import OS
from olive.workflows import run as olive_run

# flake8: noqa: T201


SUPPORTED_WORKFLOWS = {
    "cpu_fp32": [["convert", "optimize_cpu", "perf_tuning"]],
    "cpu_int4": [["convert", "optimize_cpu", "blockwise_quant_int4", "perf_tuning"]],
    "cuda_fp16": [["convert", "optimize_cuda", "perf_tuning"]],
    "cuda_int4": [["convert", "optimize_cuda", "blockwise_quant_int4", "perf_tuning"]],
    "slicegpt": [["slice"]],
    "web": [["builder", "io_float16_to_float32"]],
}
SUPPORTED_INFERENCE_CONFIG = {
    "cpu_fp32": {
        "use_buffer_share": False,
        # -1 to use CPU
        "device_id": -1,
        "use_fp16": False,
        "use_step": platform.system() == OS.LINUX,
    },
    "cpu_int4": {
        "use_buffer_share": False,
        "device_id": -1,
        "use_fp16": False,
        "use_step": platform.system() == OS.LINUX,
    },
    "cuda_fp16": {
        "use_buffer_share": False,
        "device_id": 0,
        "use_fp16": True,
        "use_step": platform.system() == OS.LINUX,
    },
    "cuda_int4": {
        "use_buffer_share": False,
        "device_id": 0,
        "use_fp16": True,
        "use_step": platform.system() == OS.LINUX,
    },
}

DEVICE_TO_EP = {
    "cpu": "CPUExecutionProvider",
    "gpu": "CUDAExecutionProvider",
    "web": "JsExecutionProvider",
}


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Phi2 optimization")

    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=["cpu_fp32", "cpu_int4", "cuda_fp16", "cuda_int4", "web"],
        help="Choose from cpu_fp32, cpu_int4, cuda_fp16, cuda_int4 or web",
    )
    parser.add_argument(
        "--finetune_method",
        type=str,
        default=None,
        help="Finetune method before onnxruntime optimization, use 'qlora' as of now "
        "it should be same with the pass name in phi2_optimize_template.json",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run inference with optimized model",
    )
    parser.add_argument(
        "--optimum_optimization",
        action="store_true",
        help="Use optimum optimization",
    )
    parser.add_argument(
        "--genai_optimization",
        action="store_true",
        help="Use GenAI optimization",
    )
    parser.add_argument(
        "--slicegpt",
        action="store_true",
        help="Use slicegpt compression",
    )
    parser.add_argument(
        "--export_mlflow_format",
        action="store_true",
        help="Export the model in mlflow format.",
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
    if not args.model_type and not args.finetune_method and not args.slicegpt:
        raise ValueError("Please specify either model_type or finetune_method or args.slicegpt")

    model_type = str(args.model_type) if args.model_type else ""

    if args.genai_optimization:
        json_file_template = "phi2_genai.json"
        with open(json_file_template) as f:
            template_json = json.load(f)
            ep_str, precision = model_type.split("_")
            device = "GPU" if ep_str == "cuda" else "CPU"
            template_json["passes"]["builder"]["precision"] = precision
            template_json["systems"]["local_system"]["accelerators"] = [
                {"device": device, "execution_providers": [DEVICE_TO_EP[device.lower()]]}
            ]
        new_json_file = f"phi2_genai_{device.lower()}.json"
        with open(new_json_file, "w") as f:
            json.dump(template_json, f, indent=4)
    elif model_type == "web":
        json_file_template = "phi2_genai.json"
        with open(json_file_template) as f:
            template_json = json.load(f)
            template_json["passes"]["builder"]["precision"] = "int4"
            template_json["systems"]["local_system"]["accelerators"] = [
                {"device": "GPU", "execution_providers": ["JsExecutionProvider"]}
            ]
            fl_type = {"type": "OnnxIOFloat16ToFloat32"}
            template_json["passes"]["fp32_logits"] = fl_type
        new_json_file = "phi2_web.json"
        with open(new_json_file, "w") as f:
            json.dump(template_json, f, indent=4)
    else:
        if not args.optimum_optimization and not args.slicegpt and version.parse(OrtVersion) < version.parse("1.18.0"):
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

        if platform.system() == OS.WINDOWS:
            legacy_optimization_setting(template_json)

        # add pass flows
        pass_flows = [[]]
        if args.finetune_method:
            pass_flows[0].append(args.finetune_method)
            # torch fine tuning does not require execution provider, just set it to CUDAExecutionProvider
            update_accelerator(template_json, "gpu")
        if args.slicegpt:
            pass_flows[0].extend(SUPPORTED_WORKFLOWS["slicegpt"][0])
            update_accelerator(template_json, "gpu")
            del template_json["input_model"]["io_config"]

        if model_type:
            pass_flows[0].extend(SUPPORTED_WORKFLOWS[model_type][0])
            template_json["pass_flows"] = pass_flows
            if args.optimum_optimization:
                legacy_optimization_setting(template_json)
                for pass_flow in template_json["pass_flows"]:
                    pass_flow[0] = "optimum_convert"
                    if "perf_tuning" in pass_flow:
                        pass_flow.remove("perf_tuning")

            if "cuda" in model_type:
                update_accelerator(template_json, "gpu")

            if "cpu" in model_type:
                update_accelerator(template_json, "cpu")

        if args.optimum_optimization or (args.finetune_method and not args.model_type) or args.slicegpt:
            # set evaluator as None:
            template_json["evaluate_input_model"] = False
            del template_json["evaluator"]

        used_passes = {pass_name for pass_flow in pass_flows for pass_name in pass_flow}
        for pass_name in list(template_json["passes"].keys()):
            if pass_name not in used_passes:
                del template_json["passes"][pass_name]
                continue

        if args.export_mlflow_format:
            template_json["packaging_config"] = [
                {"type": "Zipfile", "name": "mlflow_model", "export_in_mlflow_format": True}
            ]

        new_json_file = "phi2_slicegpt.json" if args.slicegpt else f"phi2_{model_type}.json"
        with open(new_json_file, "w") as f:
            json.dump(template_json, f, indent=4)

    # only evaluate onnx generate model
    footprints = olive_run(new_json_file)  # pylint: disable=not-callable
    output_model_path = get_output_model_path(footprints)
    if args.genai_optimization and args.inference:
        # TODO(anyone): add genai generation script to examples/utils/generator.py
        from generate import genai_run

        prompts = args.prompt if isinstance(args.prompt, list) else [args.prompt]
        genai_run(prompts, str(output_model_path.parent))
    elif model_type and not args.slicegpt:
        if args.inference and model_type in SUPPORTED_INFERENCE_CONFIG:
            from generate import run as generate_run

            for text in generate_run(
                args.prompt,
                output_model_path,
                **SUPPORTED_INFERENCE_CONFIG[model_type],
                use_optimum=args.optimum_optimization,
                max_length=args.max_length,
            ):
                print(f"Generation output: {text}")
                print("*" * 50)


def update_accelerator(config, device):
    config["systems"]["local_system"]["accelerators"][0]["device"] = device
    config["systems"]["local_system"]["accelerators"][0]["execution_providers"] = [DEVICE_TO_EP[device]]


def legacy_optimization_setting(config):
    config["passes"]["convert"]["use_dynamo_exporter"] = False
    config["passes"]["convert"]["target_opset"] = 17
    config["passes"]["optimize_cpu"]["model_type"] = "gpt2"
    config["passes"]["optimize_cuda"]["model_type"] = "gpt2"


if __name__ == "__main__":
    main()
