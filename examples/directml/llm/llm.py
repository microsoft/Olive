# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Optional
from model_type_mapping import get_model_repo_id, get_all_supported_models, get_model_dir, get_model_name, get_supported_lvlm_models

import config
import torch
import transformers
from chat_app.app import launch_chat_app
from huggingface_hub import hf_hub_download
from run_llm_io_binding import run_llm_io_binding
from run_vision_llm_io_binding import run_vision_llm_io_binding

from olive.model import ONNXModelHandler
from olive.workflows import run as olive_run


def set_config_parameters(tokenizer: transformers.AutoTokenizer, repo_id: str, num_layers: Optional[int]):
    if repo_id == "llava-hf/llava-1.5-7b-hf":
        hugggingface_model = transformers.LlavaForConditionalGeneration.from_pretrained(repo_id)
        llm_model = hugggingface_model.language_model
        main_model = hugggingface_model
    else:
        pipeline = transformers.pipeline(
            "text-generation", model=repo_id, tokenizer=tokenizer, torch_dtype=torch.float32, device="cpu"
        )
        llm_model = pipeline.model
        main_model = pipeline.model
        config.state_dict = pipeline.model.state_dict()

    config.hidden_size = llm_model.config.hidden_size
    config.num_heads = llm_model.config.num_attention_heads
    config.num_layers = num_layers or llm_model.config.num_hidden_layers
    config.vocab_size = llm_model.config.vocab_size
    config.model_type = main_model.config.model_type
    config.apply_residual_connection_post_layernorm = getattr(
        llm_model.config, "apply_residual_connection_post_layernorm", True
    )

    if hasattr(llm_model.config, "intermediate_size"):
        config.intermediate_size = llm_model.config.intermediate_size
    else:
        config.intermediate_size = llm_model.config.hidden_size * 4

    if hasattr(llm_model.config, "multi_query") and llm_model.config.multi_query:
        config.num_key_value_heads = 1
    else:
        config.num_key_value_heads = llm_model.config.num_key_value_heads

    if hasattr(llm_model.config, "rms_norm_eps"):
        config.normalization_type = "rms"
        config.epsilon = llm_model.config.rms_norm_eps
    elif hasattr(llm_model.config, "layer_norm_epsilon"):
        config.normalization_type = "layer_norm"
        config.epsilon = llm_model.config.layer_norm_epsilon
    else:
        raise ValueError("Normalization epsilon value was not found")

    config.normalization_type = "rms" if hasattr(llm_model.config, "rms_norm_eps") else "layer_norm"
    config.strict_weights_loading = config.num_layers == llm_model.config.num_hidden_layers
    config.state_dict = main_model.state_dict()


def optimize(model_dir: Path, repo_id: str, model_name: str, num_layers: Optional[int]):
    print(f"\nOptimizing {repo_id}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id)
    set_config_parameters(tokenizer, repo_id, num_layers)

    script_dir = Path(__file__).resolve().parent
    model_info = {}

    with Path.open(script_dir / "config_llm.json") as fin:
        olive_config = json.load(fin)
        olive_config["engine"]["output_name"] = model_name
        olive_config["passes"]["optimize"]["config"]["hidden_size"] = config.hidden_size
        olive_config["passes"]["optimize"]["config"]["num_heads"] = config.num_heads
        olive_config["passes"]["optimize"]["config"]["num_key_value_heads"] = config.num_key_value_heads

        # Some models are too fragile and need layer norm to be performed in fp32 to keep their accuracy.
        # bfloat16 could fix this, but since DML doesn't support it we need to fall back to fp32.
        models_that_need_fp32_layer_norm = ["llava-hf_llava-1.5-7b-hf", "tiiuae_falcon-7b-instruct"]
        models_that_need_fp32_mha = ["llava-hf_llava-1.5-7b-hf"]
        if model_name in models_that_need_fp32_layer_norm:
            olive_config["passes"]["optimize"]["config"]["force_fp32_ops"] = [
                "SimplifiedLayerNormalization",
                "LayerNormalization",
            ]

        if model_name in models_that_need_fp32_mha:
            olive_config["passes"]["optimize"]["config"]["force_fp32_ops"] = ["MultiHeadAttention"]

        if repo_id == "llava-hf/llava-1.5-7b-hf":
            olive_config["passes"]["optimize"]["config"]["replace_attn_mask_input_with_seq_len"] = False

        # Set the input names and dynamic axes
        model_components = olive_config["input_model"]["config"]["model_components"]

        if repo_id != "llava-hf/llava-1.5-7b-hf":
            model_components[0]["config"]["io_config"]["input_names"].append("position_ids")
            model_components[1]["config"]["io_config"]["input_names"].append("position_ids_increment")

        model_components[0]["config"]["io_config"]["input_names"].append("attention_mask")
        model_components[1]["config"]["io_config"]["input_names"].append("attention_mask")

        if repo_id == "llava-hf/llava-1.5-7b-hf":
            model_components[0]["config"]["io_config"]["input_names"].append("pixel_values")
            model_components[0]["config"]["io_config"]["dynamic_axes"]["pixel_values"] = {
                "0": "batch_size",
                "1": "channel_count",
                "2": "image_size",
                "3": "image_size",
            }

        for model_component in model_components:
            for layer_idx in range(config.num_layers):
                model_component["config"]["io_config"]["input_names"].append(f"cache.{layer_idx}.key")
                model_component["config"]["io_config"]["input_names"].append(f"cache.{layer_idx}.value")
                model_component["config"]["io_config"]["output_names"].append(f"cache_out.{layer_idx}.key")
                model_component["config"]["io_config"]["output_names"].append(f"cache_out.{layer_idx}.value")
                model_component["config"]["io_config"]["dynamic_axes"][f"cache.{layer_idx}.key"] = {
                    "0": "batch_size",
                    "2": "max_seq_len",
                }
                model_component["config"]["io_config"]["dynamic_axes"][f"cache.{layer_idx}.value"] = {
                    "0": "batch_size",
                    "2": "max_seq_len",
                }

        olive_run(olive_config)

        footprints_file_path = Path(__file__).resolve().parent / "footprints" / f"{model_name}_gpu-dml_footprints.json"
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
            assert optimizer_footprint is not None
            assert merging_footprint is not None
            optimized_olive_model = ONNXModelHandler(**merging_footprint["model_config"]["config"])

            model_info[model_name] = {
                "optimized": {
                    "path": Path(optimized_olive_model.model_path),
                },
            }

            print(f"Optimized Model   : {model_info[model_name]['optimized']['path']}")

    print("Copying optimized model...")

    # Copy the ONNX models
    src_path = model_info[model_name]["optimized"]["path"]
    dst_path = model_dir / src_path.name
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copyfile(src_path, dst_path)

    # Copy the weights
    src_weights_path = src_path.with_suffix(".onnx.data")
    if src_weights_path.is_file():
        dst_weights_path = dst_path.with_suffix(".onnx.data")
        shutil.copyfile(src_weights_path, dst_weights_path)

    # Copy the tokenizer files
    tokenizer.save_pretrained(dst_path.parents[0])

    # Copy the preprocessor config file
    if config.model_type == "llava":
        src_preprocessor_config_path = hf_hub_download(repo_id=repo_id, filename="preprocessor_config.json")
        dst_preprocessor_config_path = dst_path.parents[0] / "preprocessor_config.json"
        shutil.copyfile(src_preprocessor_config_path, dst_preprocessor_config_path)

    print(f"The optimized pipeline is located here: {model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--interactive", action="store_true", help="Run with a GUI")
    parser.add_argument(
        "--expose_locally",
        action="store_true",
        help="Expose the web UI on the local network (does nothing if --interactive is not supplied)",
    )
    parser.add_argument("--prompt", default="What is the lightest element?", type=str)
    parser.add_argument("--max_seq_len", default=2048, type=int, help="The size of the cache")
    parser.add_argument("--device_id", default=0, type=int, help="GPU device to use during inference")
    parser.add_argument(
        "--max_gen_len", default=256, type=int, help="The maximum number of tokens that can be included in an answer"
    )
    parser.add_argument("--device", type=str, choices=["dml", "cuda"], default="dml")
    parser.add_argument(
        "--model_type",
        choices=get_all_supported_models(),
        help="Which model to convert.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--num_layers",
        help="This is a debugging option to be able to quickly generate and optimize an ONNX model with fewer layers "
        "that barely takes any memory and is easy to load in Netron. This value should NOT be provided for production "
        "purposes.",
        type=int,
    )
    args = parser.parse_args()

    model_name = get_model_name(args.model_type)
    model_dir = get_model_dir(args.model_type)
    repo_id = get_model_repo_id(args.model_type)

    if args.optimize or not (model_dir).exists():
        optimize(model_dir, repo_id, model_name, args.num_layers)

    if not args.optimize:
        if args.interactive:
            launch_chat_app(args.expose_locally)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if args.model_type in get_supported_lvlm_models():
                    run_vision_llm_io_binding(
                        args.model_type,
                        "What is in this image?",
                        "placeholder.png",
                        args.max_seq_len,
                        args.max_gen_len,
                        args.device,
                        args.device_id,
                    )
                else:
                    run_llm_io_binding(
                        args.model_type,
                        args.prompt,
                        args.max_seq_len,
                        args.max_gen_len,
                        args.device,
                        args.device_id,
                    )
