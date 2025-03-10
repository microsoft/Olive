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
        "--quantize",
        choices=["gptq", "blockwise", "dynamic"],
        required=False,
        help="Quantization method to use.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        required=False,
        help="Whether to use GPTQ quantization instead of RTN quantization. Only supported on gpu.",
    )
    parser.add_argument(
        "--only_config",
        action="store_true",
        required=False,
        help="Whether to only dump the config file without running the optimization.",
    )
    parser.add_argument(
        "--remote_config",
        type=str,
        required=False,
        help="Path to the azureml config file. If provided, the config file will be used to create the client.",
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        required=False,
        help="Whether to use qlora for optimization. Only supported on gpu.",
    )
    parser.add_argument(
        "--account_name",
        type=str,
        required=False,
        help="Account name for the shared cache.",
    )
    parser.add_argument(
        "--container_name",
        type=str,
        required=False,
        help="Container name for the shared cache.",
    )
    parser.add_argument(
        "--update_shared_cache",
        action="store_true",
        required=False,
        help="Whether to update the shared cache.",
    )
    parser.add_argument("--tempdir", type=str, help="Root directory for tempfile directories and files", required=False)

    return parser.parse_args(raw_args)


def main(raw_args=None):
    if version.parse(OrtVersion) < version.parse("1.16.2"):
        raise ValueError("Please use onnxruntime>=1.16.2 for llama2 optimization")

    args = get_args(raw_args)

    if args.use_gqa and not args.gpu:
        raise ValueError("GQA is only supported on gpu.")

    if args.gpu and args.quantize == "dynamic":
        raise ValueError("Dynamic quantization is only supported on CPU.")

    if args.quantize == "gptq" and not args.gpu:
        raise ValueError("GPTQ is only supported on gpu.")

    if args.qlora:
        template_json, config_name = get_qlora_config()
    else:
        template_json, config_name = get_general_config(args)

    if args.remote_config:
        with open(args.remote_config) as f:
            remote_config = json.load(f)
        template_json["azureml_client"] = {
            "subscription_id": get_valid_config(remote_config, "subscription_id"),
            "resource_group": get_valid_config(remote_config, "resource_group"),
            "workspace_name": get_valid_config(remote_config, "workspace_name"),
            "keyvault_name": get_valid_config(remote_config, "keyvault_name"),
        }
        template_json["systems"]["aml_system"] = {
            "type": "AzureML",
            "accelerators": [{"device": "GPU", "execution_providers": ["CUDAExecutionProvider"]}],
            "aml_compute": get_valid_config(remote_config, "compute"),
            "aml_docker_config": {
                "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
                "conda_file_path": "conda_gpu.yaml",
            },
            "hf_token": True,
        }
        template_json["workflow_host"] = "aml_system"

    if args.account_name and args.container_name:
        template_json["cache_config"] = {
            "account_name": args.account_name,
            "container_name": args.container_name,
            "update_shared_cache": args.update_shared_cache,
        }

    # dump config
    with open(f"{config_name}.json", "w") as f:
        json.dump(template_json, f, indent=4)

    if not args.only_config:
        olive_run(template_json, tempdir=args.tempdir)  # pylint: disable=not-callable


def get_valid_config(config, key, default=None):
    if key in config:
        return config[key]
    if default is not None:
        return default
    raise ValueError(f"Key {key} is required in the config file.")


def get_qlora_config():
    with open("llama2_qlora.json") as f:
        template_json = json.load(f)
    return template_json, "llama2_gpu_qlora"


def get_general_config(args):
    with open("llama2_template.json") as f:
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

    precision = args.precision or ("fp16" if args.gpu else "fp32")

    # add pass names
    used_passes = {"conversion_merged"}
    used_passes.add("transformers_optimization_fp16" if precision == "fp16" else "transformers_optimization_fp32")

    if args.quantize == "gptq":
        used_passes.add("gptq_quant_int4")

        auto_gptq_logger = logging.getLogger("auto_gptq")
        auto_gptq_logger.addHandler(logging.StreamHandler(sys.stdout))
        auto_gptq_logger.setLevel(logging.INFO)
    elif args.quantize == "blockwise":
        used_passes.add("blockwise_quant_int4")
    elif args.quantize == "dynamic":
        used_passes.add("onnx_dynamic_quant_int8")

    # remove unused passes and set gqa related configs
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

    return template_json, config_name


if __name__ == "__main__":
    main()
