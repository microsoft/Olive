# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import json
import time
from pathlib import Path

import onnxruntime_genai as og

from olive.common.utils import unescaped_str
from olive.workflows import run as olive_run

# flake8: noqa: T201


TARGETS = ["cpu", "cuda", "mobile", "web"]

TARGET_TO_EP = {
    "cpu": "CPUExecutionProvider",
    "mobile": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "web": "JsExecutionProvider",
}

AML_MODEL_Path = {
    "type": "azureml_registry_model",
    "registry_name": "azureml",
    "name": "Phi-3-mini-4k-instruct",
    "version": "7",
}


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="phi3 optimization")

    parser.add_argument(
        "--model_path",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help="Path to the model to optimize. Can be a hf model id or local path",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="HF",
        choices=["HF", "AzureML"],
        help=(
            "Choose from HF(default), AzureML. If AzureML, model_path is overridden with the Phi-3-mini-4k-instruct"
            " from azureml model registry"
        ),
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        required=True,
        choices=TARGETS,
        help="Choose from cpu, cuda, mobile or web",
    )
    parser.add_argument(
        "--finetune_method",
        type=str,
        default=None,
        choices=["qlora", "lora"],
        help="Finetune method before onnxruntime optimization",
    )

    quant_group = parser.add_mutually_exclusive_group()
    quant_group.add_argument(
        "--awq",
        action="store_true",
        help="Run AWQ on the base model or the finetuned model",
    )
    quant_group.add_argument(
        "--tune-session-params",
        action="store_true",
        help="Tune onnx session params",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default="int4",
        choices=["fp32", "fp16", "int4"],
        help=(
            "Choose from fp32 or int4(default) for cpu target; "
            "fp32 or fp16 or int4(default) for gpu target; int4(default) for mobile or web"
        ),
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Run inference with optimized model",
    )
    parser.add_argument(
        "--prompt",
        nargs="*",
        type=str,
        default=["Write a joke"],
        help="The prompt text fed into the model. Only used with --inference",
    )
    parser.add_argument(
        "--chat_template",
        type=unescaped_str,
        default=None,
        help=(
            "The chat template for the prompt. If not provided, will use default templates for base and finetuned"
            " models. Only used with --inference"
        ),
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Max length for generation. Only used with --inference",
    )

    parser.add_argument("--output_dir", type=str, default="models/phi3", help="Output path for optimized model")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Path to cache directory",
    )

    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)
    if args.target in ("mobile", "web") and args.precision != "int4":
        raise ValueError("mobile or web only supports int4(default)")
    elif args.target == "cpu" and args.precision == "fp16":
        raise ValueError("Choose from fp32 or int4(default) for cpu target")

    if args.inference and args.target == "web":
        raise ValueError("Web model inference is not supported in this script")

    # Generate Olive configuration file for specific target
    print("\nGenerating Olive configuration file...")
    config_file = generate_config(args)
    print("Olive configuration file is generated...\n")

    # Generate optimized model for specific target
    print("Generating optimized model for", args.target, "...\n")
    output_path = Path(args.output_dir)
    with open(config_file) as f:
        run_config = json.load(f)
        run_config["output_dir"] = args.output_dir

    olive_run(run_config)

    if args.inference:
        if not args.chat_template:
            args.chat_template = (
                "### Question: {input} \n### Answer: "
                if args.finetune_method
                else "<|user|>\n{input}<|end|>\n<|assistant|>"
            )

        prompts = "Write a joke" if not args.prompt else "".join(args.prompt)

        prompts = f"{args.chat_template.format(input=prompts)}"

        max_length = 200 if not args.max_length else args.max_length

        genai_run(prompts, str(output_path / "model"), max_length)


def use_passes(template_json, *passes):
    use_data_configs = set()

    # remove unused passes
    for key in list(template_json["passes"].keys()):
        if key not in passes:
            del template_json["passes"][key]
            continue
        for param, value in template_json["passes"][key].items():
            if param.endswith("data_config"):
                use_data_configs.add(value)

    # remove unused data_configs
    if use_data_configs:
        template_json["data_configs"] = [
            data_config for data_config in template_json["data_configs"] if data_config["name"] in use_data_configs
        ]
    else:
        del template_json["data_configs"]

    for pass_name in set(template_json["passes"].keys()):
        if pass_name not in passes:
            template_json["passes"].pop(pass_name, None)

    return template_json


def generate_config(args):

    json_file_template = "phi3_template.json"
    with open(json_file_template) as f:
        template_json = json.load(f)

    config_prefix = "phi3_run_"

    # use aml instance of model
    if args.source == "AzureML":
        template_json["input_model"]["model_path"] = AML_MODEL_Path
    else:
        template_json["input_model"]["model_path"] = args.model_path

    # finetune
    passes_to_use = []
    if args.finetune_method:
        # adapters will be fine-tuned and merged into the model
        passes_to_use.extend([args.finetune_method, "merge_adapter_weights"])
    if args.awq:
        passes_to_use.append("awq")
        if args.precision != "int4":
            print("AWQ only supports int4 precision. Changing precision to int4")
            args.precision = "int4"
    passes_to_use.append("builder")

    if args.tune_session_params:
        passes_to_use.append("tune_session_params")
        template_json["search_strategy"] = {"execution_order": "joint", "sampler": "sequential"}
        template_json["evaluator"] = "common_evaluator"
    else:
        del template_json["evaluators"]

    target = str(args.target)
    if target == "web":
        # web doesn't have fp16 io
        passes_to_use.append("fp32_logits")

    # use the relevant passes
    template_json = use_passes(template_json, *passes_to_use)

    # set the accelerator
    device = "GPU" if target in ("cuda", "web") else "CPU"
    template_json["systems"]["local_system"]["accelerators"] = [
        {"device": device, "execution_providers": [TARGET_TO_EP[target.lower()]]}
    ]

    # set the precision
    template_json["passes"]["builder"]["precision"] = args.precision
    if target == "mobile":
        template_json["passes"]["builder"]["int4_accuracy_level"] = 4

    # set cache dir
    template_json["cache_dir"] = args.cache_dir

    new_json_file = f"{config_prefix}{target.lower()}_{args.precision}.json"
    with open(new_json_file, "w") as f:
        json.dump(template_json, f, indent=4)

    return new_json_file


def genai_run(prompt, model_path, max_length):

    print("\nModel inference starts...")

    print("Loading model...")
    app_started_timestamp = time.time()
    model = og.Model(model_path)
    model_loaded_timestamp = time.time()
    print("Model loaded in {:.2f} seconds".format(model_loaded_timestamp - app_started_timestamp))

    print("Creating tokenizer...")
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    input_tokens = tokenizer.encode(prompt)
    started_timestamp = time.time()

    print("Creating generator ...")
    params = og.GeneratorParams(model)
    # optimal search options for Phi3
    search_options = {
        "max_length": max_length,
        "top_k": 40,
        "top_p": 0.95,
        "temperature": 0.8,
        "repetition_penalty": 1.0,
    }
    params.set_search_options(**search_options)
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)
    print("Generator created")

    first = True
    first_token_timestamp = None
    new_tokens = []

    print("\n", prompt)

    try:
        while not generator.is_done():
            generator.generate_next_token()
            if first:
                first_token_timestamp = time.time()
                first = False

            new_token = generator.get_next_tokens()[0]
            print(tokenizer_stream.decode(new_token), end="", flush=True)
            new_tokens.append(new_token)
    except KeyboardInterrupt:
        print("  --control+c pressed, aborting generation--")

    del generator

    run_time = time.time() - started_timestamp
    if first_token_timestamp is None:
        print("\n\nNo tokens generated")
    else:
        print(
            "\n\n"
            f"Prompt tokens: {len(input_tokens)}, New tokens: {len(new_tokens)},"
            f" Time to first: {(first_token_timestamp - started_timestamp):.2f}s,"
            f" New tokens per second: {len(new_tokens)/run_time:.2f} tps"
        )


if __name__ == "__main__":
    main()
