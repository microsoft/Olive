# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import json
import logging
import sys

from onnxruntime import __version__ as OrtVersion
from packaging import version

from olive.workflows import run as olive_run

SUPPORTED_WORKFLOWS = {
    "cpu": [
        ["conversion_merged", "transformers_optimization_fp32"],
        ["conversion_merged", "transformers_optimization_fp32", "onnx_dynamic_quant_int8"],
        ["conversion_merged", "transformers_optimization_fp32", "blockwise_quant_int4"],
    ],
    "gpu": [
        ["conversion_merged", "transformers_optimization_fp16"],
        ["conversion_merged", "transformers_optimization_fp16", "blockwise_quant_int4"],
        ["gptq_quant_int4", "conversion_merged", "transformers_optimization_fp32"],
        ["gptq_quant_int4", "conversion_merged", "transformers_optimization_fp16"],
    ],
}
DEVICE_TO_EP = {
    "cpu": "CPUExecutionProvider",
    "gpu": "CUDAExecutionProvider",
}


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Llama2 optimization")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model name, currently only supports llama2 7B/13B",
    )
    parser.add_argument("--gpu", action="store_true", required=False, help="Whether to use gpu for optimization.")
    parser.add_argument(
        "--use_gqa",
        action="store_true",
        required=False,
        help="Whether to use GQA(grouped query attention) instead of MHA(multi-head attention). Only supported on gpu.",
    )
    parser.add_argument(
        "--use_gptq",
        action="store_true",
        required=False,
        help="Whether to use GPTQ quantization instead of RTN quantization. Only supported on gpu.",
    )
    parser.add_argument(
        "--only_config",
        action="store_true",
        required=False,
        help="Whether to only dump the config file without running the optimization.",
    )
    parser.add_argument("--tempdir", type=str, help="Root directory for tempfile directories and files", required=False)

    return parser.parse_args(raw_args)


def main(raw_args=None):
    if version.parse(OrtVersion) < version.parse("1.16.2"):
        raise ValueError("Please use onnxruntime>=1.16.2 for llama2 optimization")

    args = get_args(raw_args)

    if args.use_gqa and not args.gpu:
        raise ValueError("GQA is only supported on gpu.")

    json_file_template = "llama2_template.json"
    with open(json_file_template) as f:
        template_json = json.load(f)

    model_name = args.model_name
    # update model name
    template_json_str = json.dumps(template_json)
    template_json_str = template_json_str.replace("<model_name_placeholder>", model_name)
    template_json = json.loads(template_json_str)

    # update configs
    device = "gpu" if args.gpu else "cpu"
    gqa = "gqa" if args.use_gqa else "mha"
    config_name = f"llama2_{device}_{gqa}"

    # add pass flows
    if not args.use_gptq:
        template_json["pass_flows"] = [flow for flow in SUPPORTED_WORKFLOWS[device] if "gptq" not in flow[0]]
    else:
        template_json["pass_flows"] = [flow for flow in SUPPORTED_WORKFLOWS[device] if "gptq" in flow[0]]
        auto_gptq_logger = logging.getLogger("auto_gptq")
        auto_gptq_logger.addHandler(logging.StreamHandler(sys.stdout))
        auto_gptq_logger.setLevel(logging.INFO)

    # remove unused passes and set gqa related configs
    used_passes = {pass_name for pass_flow in SUPPORTED_WORKFLOWS[device] for pass_name in pass_flow}
    for pass_name in list(template_json["passes"].keys()):
        if pass_name not in used_passes:
            del template_json["passes"][pass_name]
            continue
        if not args.use_gqa and template_json["passes"][pass_name].get("evaluator", None) == "gqa_evaluator":
            # remove gqa evaluator if not using gqa
            del template_json["passes"][pass_name]["evaluator"]
        if not args.use_gqa and template_json["passes"][pass_name].get("use_gqa", False):
            # set use_gqa to False if not using gqa
            template_json["passes"][pass_name]["use_gqa"] = False
    if not args.use_gqa:
        del template_json["evaluators"]["gqa_evaluator"]

    template_json["systems"]["local_system"]["accelerators"][0]["device"] = device
    template_json["systems"]["local_system"]["accelerators"][0]["execution_providers"] = [DEVICE_TO_EP[device]]
    template_json["output_dir"] = f"models/{config_name}/{model_name}"

    # dump config
    with open(f"{config_name}.json", "w") as f:
        json.dump(template_json, f, indent=4)

    if not args.only_config:
        olive_run(template_json, tempdir=args.tempdir)  # pylint: disable=not-callable


if __name__ == "__main__":
    main()
