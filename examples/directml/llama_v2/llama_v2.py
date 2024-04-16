# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import json
import os
import shutil
import urllib.request
import warnings
from pathlib import Path

import config
from chat_app.app import launch_chat_app
from run_llama_v2_io_binding import run_llama_v2_io_binding

from olive.model import ONNXModelHandler
from olive.workflows import run as olive_run

# pylint: disable=redefined-outer-name
# ruff: noqa: T201


def optimize(optimized_model_dir: Path, model_type: str):
    script_dir = Path(__file__).resolve().parent
    model_info = {}
    submodel_names = ["argmax_sampling", "llama_v2"]

    for submodel_name in submodel_names:
        print(f"\nOptimizing {submodel_name}")

        olive_config = None
        with Path.open(script_dir / f"config_{submodel_name}.json") as fin:
            olive_config = json.load(fin)

            # ORT-DML doesn't support SimplifiedLayerNorm or SkipSimplifiedLayerNorm yet, so only enable the fusions if
            # LayerNorm is selected
            if submodel_name == "llama_v2":
                if config.normalization_type == "layer_norm":
                    olive_config["passes"]["optimize"]["config"]["optimization_options"]["enable_layer_norm"] = True
                    del olive_config["passes"]["optimize"]["config"]["force_fp32_nodes"]

                # Fewer than 32 layers can be provided for debugging purposes so we have to remove them from the config
                if config.num_layers < 32:
                    model_components = olive_config["input_model"]["config"]["model_components"]
                    for model_component in model_components:
                        layer_range = range(config.num_layers, 32)

                        # Remove the extra inputs
                        key_inputs_to_remove = {f"past_key_values.{idx}.key" for idx in layer_range}
                        value_inputs_to_remove = {f"past_key_values.{idx}.value" for idx in layer_range}
                        input_names = model_component["config"]["io_config"]["input_names"]
                        input_names = [x for x in input_names if x not in key_inputs_to_remove]
                        input_names = [x for x in input_names if x not in value_inputs_to_remove]
                        model_component["config"]["io_config"]["input_names"] = input_names

                        # Remove the extra outputs
                        key_output_to_remove = {f"present.{idx}.key" for idx in layer_range}
                        value_output_to_remove = {f"present.{idx}.value" for idx in layer_range}
                        output_names = model_component["config"]["io_config"]["output_names"]
                        output_names = [x for x in output_names if x not in key_output_to_remove]
                        output_names = [x for x in output_names if x not in value_output_to_remove]
                        model_component["config"]["io_config"]["output_names"] = output_names

                        # Remove the dynamic axes
                        for idx in layer_range:
                            del model_component["config"]["io_config"]["dynamic_axes"][f"past_key_values.{idx}.key"]
                            del model_component["config"]["io_config"]["dynamic_axes"][f"past_key_values.{idx}.value"]

        olive_run(olive_config)

        footprints_file_path = (
            Path(__file__).resolve().parent / "footprints" / f"{submodel_name}_gpu-dml_footprints.json"
        )
        with footprints_file_path.open("r") as footprint_file:
            footprints = json.load(footprint_file)

            conversion_footprint = None
            optimizer_footprint = None
            merging_footprint = None
            for footprint in footprints.values():
                if footprint["from_pass"] == "OnnxConversion":
                    conversion_footprint = footprint
                elif footprint["from_pass"] == "OrtTransformersOptimization":
                    optimizer_footprint = footprint
                elif footprint["from_pass"] == "OptimumMerging":
                    merging_footprint = footprint

            assert conversion_footprint is not None

            if submodel_name == "llama_v2":
                assert optimizer_footprint is not None
                assert merging_footprint is not None
                optimized_olive_model = ONNXModelHandler(**merging_footprint["model_config"]["config"])
            else:
                optimized_olive_model = ONNXModelHandler(**conversion_footprint["model_config"]["config"])

            model_info[submodel_name] = {
                "optimized": {
                    "path": Path(optimized_olive_model.model_path),
                },
            }

            print(f"Optimized Model   : {model_info[submodel_name]['optimized']['path']}")

    print("Copying optimized models...")
    for submodel_name in submodel_names:
        src_path = model_info[submodel_name]["optimized"]["path"]
        dst_path = optimized_model_dir / submodel_name / src_path.name
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copyfile(src_path, dst_path)

        src_weights_path = src_path.with_suffix(".onnx.data")
        if src_weights_path.is_file():
            dst_weights_path = dst_path.with_suffix(".onnx.data")
            shutil.copyfile(src_weights_path, dst_weights_path)

    raw_data_folder = Path(__file__).resolve().parent / "raw_model_data" / model_type
    raw_data_folder.mkdir(exist_ok=True, parents=True)
    src_tokenizer_path = raw_data_folder / "tokenizer.model"
    dst_tokenizer_path = optimized_model_dir / "tokenizer.model"
    shutil.copyfile(src_tokenizer_path, dst_tokenizer_path)

    print(f"The optimized pipeline is located here: {optimized_model_dir}")


def download_checkpoint(model_type: str):
    model_name = f"llama-2-{model_type}"

    raw_data_folder = Path(__file__).resolve().parent / "raw_model_data" / model_type
    raw_data_folder.mkdir(exist_ok=True, parents=True)

    license_path = raw_data_folder / "LICENSE"
    use_policy_path = raw_data_folder / "USE_POLICY.md"
    tokenizer_path = raw_data_folder / "tokenizer.model"
    weights_path = raw_data_folder / f"{model_name}.pth"

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "wget")]
    urllib.request.install_opener(opener)

    email_url = None
    if not (
        license_path.is_file() and use_policy_path.is_file() and tokenizer_path.is_file() and weights_path.is_file()
    ):
        email_url = input(
            "URL from the e-mail that was received after requesting access from "
            "https://ai.meta.com/resources/models-and-libraries/llama-downloads/ (only valid for 24h): "
        )

    if not license_path.is_file():
        print("Downloading LICENSE")
        urllib.request.urlretrieve(email_url.replace("*", "LICENSE"), license_path)

    if not use_policy_path.is_file():
        print("Downloading Acceptable Usage Policy")
        urllib.request.urlretrieve(email_url.replace("*", "USE_POLICY.md"), use_policy_path)

    if not tokenizer_path.is_file():
        print("Downloading tokenizer")
        urllib.request.urlretrieve(email_url.replace("*", "tokenizer.model"), tokenizer_path)

    if not weights_path.is_file():
        print(f"Downloading {model_name}")
        urllib.request.urlretrieve(email_url.replace("*", f"{model_name}/consolidated.00.pth"), weights_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--interactive", action="store_true", help="Run with a GUI")
    parser.add_argument(
        "--expose_locally",
        action="store_true",
        help="Expose the web UI on the local network (does nothing if --interactive is not supplied)",
    )
    parser.add_argument(
        "--normalization_type",
        default="rms",
        choices=["layer_norm", "rms"],
        help="Whether to use LayerNorm for the normalization layers or RMS.",
        type=str,
    )
    parser.add_argument("--prompt", default="What is the lightest element?", type=str)
    parser.add_argument("--max_seq_len", default=2048, type=int, help="The size of the cache")
    parser.add_argument("--device_id", default=0, type=int, help="GPU device to use during inference")
    parser.add_argument(
        "--max_gen_len", default=256, type=int, help="The maximum number of tokens that can be included in an answer"
    )
    parser.add_argument(
        "--model_type",
        default="7b-chat",
        choices=["7b", "7b-chat"],
        help="Which model to convert. The 7b model is the original one without any finetuning, and the 7b-chat "
        "version is the finetuned model optimized for chat.",
        type=str,
    )
    parser.add_argument(
        "--num_layers",
        default=32,
        help="This is a debugging option to be able to quickly generate and optimize an ONNX model with fewer layers "
        "than 32 that barely takes any memory and is easy to load in Netron. This value should ALWAYS be 32 for "
        "production purposes.",
        type=int,
    )
    args = parser.parse_args()

    config.model_type = args.model_type
    config.normalization_type = args.normalization_type
    config.num_layers = args.num_layers

    script_dir = Path(__file__).resolve().parent
    optimized_model_dir = script_dir / "models" / "optimized" / "llama_v2"

    if args.optimize or not optimized_model_dir.exists():
        download_checkpoint(args.model_type)
        optimize(optimized_model_dir, args.model_type)

    if not args.optimize:
        if args.interactive:
            launch_chat_app(args.expose_locally)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run_llama_v2_io_binding(args.prompt, args.max_seq_len, args.max_gen_len, args.device_id)
