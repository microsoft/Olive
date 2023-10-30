# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import json
import re
from pathlib import Path

from onnxruntime import __version__ as OrtVersion
from packaging import version

import olive.workflows.run as olive_run

SUPPORTED_WORKFLOWS = {
    "cpu": [
        ["conversion_merged", "transformers_optimization_fp32"],
        ["conversion_merged", "transformers_optimization_fp32", "onnx_dynamic_quant_int8"],
        ["conversion_merged", "transformers_optimization_fp32", "blockwise_quant_int4"],
    ],
    "gpu": [
        ["conversion_merged", "transformers_optimization_fp16"],
        ["conversion_merged", "transformers_optimization_fp16", "blockwise_quant_int4"],
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
        help="Whether to use GQA(grouped query attention) instead of MHA(multi-head attention).",
    )
    parser.add_argument(
        "--only_config",
        action="store_true",
        required=False,
        help="Whether to use GQA(grouped query attention) instead of MHA(multi-head attention).",
    )

    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)

    # version check
    version_1_17 = version.parse(OrtVersion) >= version.parse("1.17.0")

    if not version_1_17:
        raise ValueError("Please use onnxruntime>=1.17.0 for llama2 optimization")

    json_file_template = "llama2_template.json"
    with open(json_file_template) as f:  # noqa: PTH123
        template_json = json.load(f)

    model_name = args.model_name
    # update model name
    template_json["input_model"]["config"]["hf_config"]["model_name"] = model_name

    # update ep
    device = "cpu" if not args.gpu else "gpu"
    template_json["pass_flows"] = SUPPORTED_WORKFLOWS[device]

    template_json["engine"]["execution_providers"] = [DEVICE_TO_EP[device]]
    template_json["engine"]["output_dir"] = f"llama2_{device}/{model_name}"

    if not args.use_gqa and args.gpu:
        template_json["passes"]["transformers_optimization_fp16"]["config"]["use_gqa"] = False
        # after applying GQA, the model's input will be changed, we need to remove the special dataloader implementation

        del template_json["passes"]["transformers_optimization_fp16"]["evaluator"]
        del template_json["passes"]["blockwise_quant_int4"]["evaluator"]
        del template_json["evaluators"]["gqa_evaluator"]

    # update user script
    user_script_path = Path(__file__).parent / "user_script.py"
    update_user_script(user_script_path, model_name)

    device = "gpu" if args.gpu else "cpu"
    gqa = "gqa" if args.use_gqa else "mha"
    # dump config
    with open(f"llama2_{device}_{gqa}.json", "w") as f:  # noqa: PTH123
        json.dump(template_json, f, indent=4)

    if not args.only_config:
        olive_run(template_json)  # pylint: disable=not-callable


def update_user_script(file_path, model_name):
    with open(file_path) as file:  # noqa: PTH123
        lines = file.readlines()

    new_lines = []
    for line in lines:
        updated_line = line
        if "meta-llama/Llama-2" in line:
            updated_line = re.sub(r"meta-llama/Llama-2-(\d+)b-hf", model_name, line)
        new_lines.append(updated_line)

    with open(file_path, "w") as file:  # noqa: PTH123
        file.writelines(new_lines)


if __name__ == "__main__":
    main()
