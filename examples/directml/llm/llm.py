# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# ruff: noqa: T201

import argparse
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Optional

import config
import torch
import transformers
from huggingface_hub import hf_hub_download
from model_type_mapping import (
    get_all_supported_models,
    get_model_dir,
    get_model_name,
    get_model_repo_id,
    get_supported_lvlm_models,
)
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
            "text-generation",
            model=repo_id,
            tokenizer=tokenizer,
            torch_dtype=torch.float32,
            device="cpu",
            trust_remote_code=True,
        )
        llm_model = pipeline.model
        main_model = pipeline.model
        config.state_dict = pipeline.model.state_dict()

    config.hidden_size = llm_model.config.hidden_size
    config.num_heads = llm_model.config.num_attention_heads
    config.head_dim = getattr(llm_model.config, "head_dim", config.hidden_size // config.num_heads)
    config.num_layers = num_layers or llm_model.config.num_hidden_layers
    config.vocab_size = llm_model.config.vocab_size
    config.model_type = main_model.config.model_type

    if hasattr(llm_model.config, "apply_residual_connection_post_layernorm"):
        config.apply_residual_connection_post_layernorm = llm_model.config.apply_residual_connection_post_layernorm
    elif llm_model.config.model_type == "phi":
        config.apply_residual_connection_post_layernorm = False
    else:
        config.apply_residual_connection_post_layernorm = True

    config.use_bias = llm_model.config.model_type == "phi"

    if hasattr(llm_model.config, "architecture"):
        config.hidden_act = llm_model.config.architecture["mlp"]["act_fn"]
    elif hasattr(llm_model.config, "hidden_act"):
        config.hidden_act = llm_model.config.hidden_act
    elif hasattr(llm_model.config, "activation_function"):
        config.hidden_act = llm_model.config.activation_function
    elif llm_model.config.model_type == "falcon":
        config.hidden_act = "gelu"
    else:
        raise ValueError("Activation function was not found")

    if hasattr(llm_model.config, "intermediate_size"):
        config.intermediate_size = llm_model.config.intermediate_size
    elif hasattr(llm_model.config, "architecture"):
        config.intermediate_size = llm_model.config.architecture["mlp"]["n_inner"]
    else:
        config.intermediate_size = llm_model.config.hidden_size * 4

    if hasattr(llm_model.config, "multi_query") and llm_model.config.multi_query:
        config.num_key_value_heads = 1
    elif hasattr(llm_model.config, "num_key_value_heads"):
        config.num_key_value_heads = llm_model.config.num_key_value_heads
    else:
        config.num_key_value_heads = config.num_heads

    if hasattr(llm_model.config, "rms_norm_eps"):
        config.normalization_type = "rms"
        config.epsilon = llm_model.config.rms_norm_eps
    elif hasattr(llm_model.config, "layer_norm_epsilon"):
        config.normalization_type = "layer_norm"
        config.epsilon = llm_model.config.layer_norm_epsilon
    elif hasattr(llm_model.config, "layer_norm_eps"):
        config.normalization_type = "layer_norm"
        config.epsilon = llm_model.config.layer_norm_eps
    else:
        raise ValueError("Normalization epsilon value was not found")

    config.model_id = repo_id

    config.use_split_sigmoid = config.model_type == "phi3"
    config.normalization_type = "rms" if hasattr(llm_model.config, "rms_norm_eps") else "layer_norm"
    config.partial_rotary_factor = getattr(llm_model.config, "partial_rotary_factor", 1.0)
    config.max_position_embeddings = (
        llm_model.config.max_position_embeddings if hasattr(llm_model.config, "max_position_embeddings") else 4096
    )
    config.strict_weights_loading = config.num_layers == llm_model.config.num_hidden_layers
    config.state_dict = main_model.state_dict()


def optimize(
    model_dir: Path,
    repo_id: str,
    model_name: str,
    device: str,
    num_layers: Optional[int],
    quant_strategy: Optional[str],
    block_size: int,
    bit_size: int,
):
    print(f"\nOptimizing {repo_id}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id)
    set_config_parameters(tokenizer, repo_id, num_layers)

    script_dir = Path(__file__).resolve().parent
    shutil.rmtree(script_dir / "footprints", ignore_errors=True)
    shutil.rmtree(script_dir / "cache", ignore_errors=True)

    with Path.open(script_dir / "config_llm.json") as fin:
        olive_config = json.load(fin)

        if quant_strategy is not None:
            olive_config["passes"]["quantize"] = {
                "type": "IncStaticQuantization",
                "disable_search": True,
                "backend": f"onnxrt_{device}_ep",
                "approach": "weight_only",
                "device": "gpu",
                "weight_only_config": {
                    "bits": bit_size,
                    "algorithm": quant_strategy.upper(),
                    "group_size": block_size,
                },
                "dataloader_func": "calib_dataloader",
                "calibration_sampling_size": [8],
                "save_as_external_data": True,
                "all_tensors_to_one_file": True,
                "user_script": "user_script.py",
            }

        olive_config["systems"]["local_system"]["accelerators"][0]["execution_providers"] = {
            "dml": ["DmlExecutionProvider"],
            "cuda": ["CUDAExecutionProvider"],
        }[device]

        olive_config["output_name"] = model_name
        olive_config["passes"]["optimize"]["hidden_size"] = config.hidden_size
        olive_config["passes"]["optimize"]["num_heads"] = config.num_heads
        olive_config["passes"]["optimize"]["num_key_value_heads"] = config.num_key_value_heads

        # Some models are too fragile and need layer norm to be performed in fp32 to keep their accuracy.
        # bfloat16 could fix this, but since DML doesn't support it we need to fall back to fp32.
        models_that_need_fp32_layer_norm = ["llava-hf_llava-1.5-7b-hf", "tiiuae_falcon-7b-instruct"]
        vision_models = ["llava-hf_llava-1.5-7b-hf"]

        force_fp32_ops = olive_config["passes"]["optimize"].get("force_fp32_ops", [])

        if model_name in models_that_need_fp32_layer_norm:
            force_fp32_ops.extend(["SimplifiedLayerNormalization", "LayerNormalization"])

        is_vision_model = model_name in vision_models

        olive_config["passes"]["optimize"]["force_fp32_ops"] = force_fp32_ops

        # Set the input names and dynamic axes
        io_config = olive_config["input_model"]["io_config"]

        if not is_vision_model:
            io_config["input_names"].append("position_ids")

        io_config["input_names"].append("attention_mask")

        if is_vision_model:
            io_config["input_names"].append("pixel_values")
            io_config["dynamic_axes"]["pixel_values"] = {
                "0": "batch_size",
                "1": "channel_count",
                "2": "image_size",
                "3": "image_size",
            }

        for layer_idx in range(config.num_layers):
            io_config["input_names"].append(f"past_key_values.{layer_idx}.key")
            io_config["input_names"].append(f"past_key_values.{layer_idx}.value")
            io_config["output_names"].append(f"present.{layer_idx}.key")
            io_config["output_names"].append(f"present.{layer_idx}.value")

            # Name the input cache dynamic axes
            io_config["dynamic_axes"][f"past_key_values.{layer_idx}.key"] = {
                "0": "batch_size",
                "2": "past_sequence_length",
            }
            io_config["dynamic_axes"][f"past_key_values.{layer_idx}.value"] = {
                "0": "batch_size",
                "2": "past_sequence_length",
            }

            # Name the output cache dynamic axes
            io_config["dynamic_axes"][f"present.{layer_idx}.key"] = {
                "0": "batch_size",
                "2": "total_sequence_length",
            }
            io_config["dynamic_axes"][f"present.{layer_idx}.value"] = {
                "0": "batch_size",
                "2": "total_sequence_length",
            }

        olive_run(olive_config)

        footprints_file_path = (
            Path(__file__).resolve().parent / "footprints" / f"{model_name}_gpu-{device}_footprints.json"
        )
        with footprints_file_path.open("r") as footprint_file:
            footprints = json.load(footprint_file)

            conversion_footprint = None
            optimizer_footprint = None
            quantizer_footprint = None
            for footprint in footprints.values():
                if footprint["from_pass"] == "OnnxConversion":
                    conversion_footprint = footprint
                elif footprint["from_pass"] == "OrtTransformersOptimization":
                    optimizer_footprint = footprint
                elif footprint["from_pass"] == "IncStaticQuantization":
                    quantizer_footprint = footprint

            assert conversion_footprint is not None
            assert optimizer_footprint is not None

            if quant_strategy is not None:
                assert quantizer_footprint is not None
                optimized_olive_model = ONNXModelHandler(**quantizer_footprint["model_config"]["config"])
            else:
                optimized_olive_model = ONNXModelHandler(**optimizer_footprint["model_config"]["config"])

            model_path = Path(optimized_olive_model.model_path)
            print(f"Optimized Model   : {model_path}")

    print("Copying optimized model...")

    # Copy the ONNX models
    src_path = model_path
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
        # src_preprocessor_config_path = hf_hub_download(repo_id=repo_id, filename="preprocessor_config.json")
        src_preprocessor_config_path = hf_hub_download(
            repo_id="llava-hf/llava-1.5-7b-hf", filename="preprocessor_config.json"
        )
        dst_preprocessor_config_path = dst_path.parents[0] / "preprocessor_config.json"
        shutil.copyfile(src_preprocessor_config_path, dst_preprocessor_config_path)

    print(f"The optimized pipeline is located here: {model_dir}")


def main():
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
        help=(
            "This is a debugging option to be able to quickly generate and optimize an ONNX model with fewer layers"
            " that barely takes any memory and is easy to load in Netron. This value should NOT be provided for"
            " production purposes."
        ),
        type=int,
    )
    parser.add_argument(
        "--quant_strategy",
        choices=["awq", "rtn"],
        help="Which quantization strategy to use. Defaults to None (no quantization).",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--block_size",
        choices=[32, 64, 128],
        help="Block size to use during quantization if --quant_strategy is provided. Defaults to 32.",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--bit_size",
        choices=[4],
        help="Number of bits to quantize to if --quant_strategy is provided. Defaults to 4.",
        default=4,
        type=int,
    )
    args = parser.parse_args()

    model_dir = get_model_dir(args.model_type)

    if args.optimize or not (model_dir).exists():
        repo_id = get_model_repo_id(args.model_type)
        model_name = get_model_name(args.model_type)
        optimize(
            model_dir,
            repo_id,
            model_name,
            args.device,
            args.num_layers,
            args.quant_strategy,
            args.block_size,
            args.bit_size,
        )

    if not args.optimize:
        if args.interactive:
            from chat_app.app import launch_chat_app

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


if __name__ == "__main__":
    main()
