# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import json
import time
from pathlib import Path

import onnxruntime_genai as og

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
        help=(
            "Finetune method before onnxruntime optimization. "
            "qlora finetuned model cannot be converted to onnx by model builder."
        ),
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
        help="The prompt text fed into the model. Not supported with Web target.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="Max length for generation. Not supported with Web target.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="HF",
        choices=["HF", "AzureML"],
        help="Choose from HF(default), AzureML",
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
    footprints = olive_run(config_file)
    if footprints:
        print("\nOptimized model is generated...")

    if args.inference:
        if args.finetune_method == "qlora":
            raise ValueError(
                "qlora finetuned model cannot be converted to onnx "
                "by model builder as of now. Please remove --inference flag"
            )
        prompts = "Write a joke" if not args.prompt else "".join(args.prompt)

        chat_template = "<|user|>\n{input}<|end|>\n<|assistant|>"
        prompts = f"{chat_template.format(input=prompts)}"

        max_length = 200 if not args.max_length else args.max_length

        output_model_path = get_output_model_path(footprints)
        genai_run(prompts, str(output_model_path), max_length)


def get_finetune_passes():
    with open("pass_configs/finetune.json") as f:
        return json.load(f)


def get_data_configs():
    with open("pass_configs/data_configs.json") as f:
        return json.load(f)


def generate_config(args):

    json_file_template = "phi3_template.json"
    with open(json_file_template) as f:
        template_json = json.load(f)

    # finetune
    if args.finetune_method:
        assert args.target == "cuda", "Finetune only supports cuda target"
        finetune_passes = get_finetune_passes()
        data_configs = get_data_configs()
        template_json["data_configs"] = data_configs
        template_json["passes"][args.finetune_method] = finetune_passes[args.finetune_method]
        template_json["passes"]["merge_adapter_weights"] = {"type": "MergeAdapterWeights"}
    if args.source == "AzureML":
        template_json["input_model"]["model_path"] = AML_MODEL_Path

    target = str(args.target)
    device = "GPU" if target in ("cuda", "web") else "CPU"
    execution_providers = [TARGET_TO_EP[target.lower()]]
    template_json["systems"]["local_system"]["accelerators"] = [
        {"device": device, "execution_providers": execution_providers}
    ]

    model_builder = {"type": "ModelBuilder", "precision": args.precision}
    if args.finetune_method is None or args.finetune_method == "lora":
        template_json["passes"]["builder"] = model_builder

    if target == "mobile":
        template_json["passes"]["builder"]["int4_accuracy_level"] = 4

    elif target == "web":
        fl_type = {"type": "OnnxIOFloat16ToFloat32"}
        template_json["passes"]["fp32_logits"] = fl_type

    new_json_file = f"phi3_{target.lower()}_{args.precision}.json"
    with open(new_json_file, "w") as f:
        json.dump(template_json, f, indent=4)

    return new_json_file


def get_output_model_path(footprints):
    # only one model output in phi2 optimization
    for footprint in footprints.values():
        for model_id in footprint.nodes:
            model_path = Path(footprint.get_model_path(model_id))
            break
    return model_path


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
    params.input_ids = input_tokens
    generator = og.Generator(model, params)
    print("Generator created")

    first = True
    first_token_timestamp = None
    new_tokens = []

    print("\n", prompt)

    try:
        while not generator.is_done():
            generator.compute_logits()
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
