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

import config
import torch
import transformers
from chat_app.app import launch_chat_app
from huggingface_hub import hf_hub_download
from run_llm_io_binding import run_llm_io_binding

from olive.model import ONNXModelHandler
from olive.workflows import run as olive_run

import safetensors

_MODELS = {
    "microsoft/phi-2": [
        "https://huggingface.co/microsoft/phi-2/blob/main/model-00001-of-00002.safetensors",
        "https://huggingface.co/microsoft/phi-2/blob/main/model-00002-of-00002.safetensors",
    ],
}

PHI_MAPPING = {
    "transformer.embd.wte.weight": "model.embed_tokens.weight",
    "lm_head.linear": "lm_head",
    "lm_head.ln": "model.norm",
    "layers": "model.layers",
    "transformer": "model",
    ".h.": ".layers.",
    "ln": "input_layernorm",
    "mixer": "self_attn",
    "Wqkv": "query_key_value",
    "out_proj": "o_proj",
}

def map_key(origin_key):
    for k, v in PHI_MAPPING.items():
        if k in origin_key:
            origin_key = origin_key.replace(k, v)
    return origin_key
 
def convert_weights(original_weights, mapping, config):
    converted_weights = {}
    original_weights_keys = sorted(original_weights.keys())

    for original_weights_key in original_weights_keys:
        new_key = original_weights_key

        if "rotary_emb" in new_key:
            continue

        if "Wqkv" in new_key:
            if "weight" in new_key:
                weight = original_weights[new_key]
                weights_shape = weight.shape
                weight = (
                    weight.view(3, config.num_heads, -1, config.hidden_size)
                    .transpose(0, 1)
                    .reshape(*weights_shape)
                ).view(config.num_heads, 3, -1)
                q_proj_key = map_key(new_key.replace("Wqkv", "q_proj"))
                k_proj_key = map_key(new_key.replace("Wqkv", "k_proj"))
                v_proj_key = map_key(new_key.replace("Wqkv", "v_proj"))
                converted_weights[q_proj_key] = weight[:, 0, ...].reshape(2560, 2560)
                converted_weights[k_proj_key] = weight[:, 1, ...].reshape(2560, 2560)
                converted_weights[v_proj_key] = weight[:, 2, ...].reshape(2560, 2560)

                original_weights.pop(new_key)

            elif "bias" in new_key:
                bias = original_weights[new_key]
                bias_shape = bias.shape
                bias = bias.view(3, config.num_heads, -1).transpose(0, 1).reshape(*bias_shape).view(config.num_heads, 3, -1)
                q_proj_key = map_key(new_key.replace("Wqkv", "q_proj"))
                k_proj_key = map_key(new_key.replace("Wqkv", "k_proj"))
                v_proj_key = map_key(new_key.replace("Wqkv", "v_proj"))
                converted_weights[q_proj_key] = bias[..., 0, :].reshape(2560)
                converted_weights[k_proj_key] = bias[..., 1, :].reshape(2560)
                converted_weights[v_proj_key] = bias[..., 2, :].reshape(2560)

                original_weights.pop(new_key)

            continue

        new_key = map_key(new_key)

        converted_weights[new_key] = original_weights.pop(original_weights_key)

    return converted_weights

def _download(url: str, root: str):
    repo_id = f"{url.split('/')[3]}/{url.split('/')[4]}"
    filename = f"{url.split('/')[-1]}"
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        force_filename=root,
        local_dir_use_symlinks=False,
    )

def load_phi2_checkpoint(checkpoint_path):
    model_checkpoint = {}
    for model_name, model_url in _MODELS.items():
        device = "cpu"
        # for phi-2 the weights are stored in 2 different safetensors file so we need to iterate over that list and download one at a time
        for model_each_url in model_url:
            model_path = os.path.join(checkpoint_path, model_name + "_" + model_each_url.split("/")[-1])
            if not os.path.exists(model_path):
                print(f"\n{model_name} was not found! Downloading it to {model_path}")
                _download(url=model_each_url, root=model_path)

            if model_path.endswith("safetensors"):
                loaded_weights = safetensors.torch.load_file(model_path, device=device)
            else:
                loaded_weights = torch.load(model_path, map_location=device)
            model_checkpoint.update(**loaded_weights)
    return model_checkpoint

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

    # checkpoint_path = "C:\\Users\\xianz\\work\\Olive\\examples\\directml\\phi\\checkpoints"
    # model_checkpoint = load_phi2_checkpoint(checkpoint_path)
    # converted_checkpoint = {}
    # converted_checkpoint.update(**convert_weights(model_checkpoint, PHI_MAPPING, config))
    # config.state_dict = converted_checkpoint


def optimize(optimized_model_dir: Path, repo_id: str, model_name: str, num_layers: Optional[int]):
    print(f"\nOptimizing {repo_id}")

    set_config_parameters(repo_id, num_layers)

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
        olive_config["passes"]["optimize"]["config"]["force_fp32_ops"] = ["LayerNormalization", "MultiHeadAttention"]

        # Fewer than 32 layers can be provided for debugging purposes so we have to remove them from the config
        if config.num_layers < 32:
            model_components = olive_config["input_model"]["config"]["model_components"]
            for model_component in model_components:
                layer_range = range(config.num_layers, 32)

                # Remove the extra inputs
                key_inputs_to_remove = {f"cache.{idx}.key" for idx in layer_range}
                value_inputs_to_remove = {f"cache.{idx}.value" for idx in layer_range}
                input_names = model_component["config"]["io_config"]["input_names"]
                input_names = [x for x in input_names if x not in key_inputs_to_remove]
                input_names = [x for x in input_names if x not in value_inputs_to_remove]
                model_component["config"]["io_config"]["input_names"] = input_names

                # Remove the extra outputs
                key_output_to_remove = {f"cache_out.{idx}.key" for idx in layer_range}
                value_output_to_remove = {f"cache_out.{idx}.value" for idx in layer_range}
                output_names = model_component["config"]["io_config"]["output_names"]
                output_names = [x for x in output_names if x not in key_output_to_remove]
                output_names = [x for x in output_names if x not in value_output_to_remove]
                model_component["config"]["io_config"]["output_names"] = output_names

                # Remove the dynamic axes
                for idx in layer_range:
                    del model_component["config"]["io_config"]["dynamic_axes"][f"cache.{idx}.key"]
                    del model_component["config"]["io_config"]["dynamic_axes"][f"cache.{idx}.value"]

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
    dst_path = optimized_model_dir / model_name / src_path.name
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copyfile(src_path, dst_path)

    # Copy the weights
    src_weights_path = src_path.with_suffix(".onnx.data")
    if src_weights_path.is_file():
        dst_weights_path = dst_path.with_suffix(".onnx.data")
        shutil.copyfile(src_weights_path, dst_weights_path)

    # Copy the tokenizer file
    src_tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
    dst_tokenizer_path = dst_path.parents[0] / "tokenizer.json"
    shutil.copyfile(src_tokenizer_path, dst_tokenizer_path)

    # Copy the tokenizer config file
    src_tokenizer_config_path = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json")
    dst_tokenizer_config_path = dst_path.parents[0] / "tokenizer_config.json"
    shutil.copyfile(src_tokenizer_config_path, dst_tokenizer_config_path)

    print(f"The optimized pipeline is located here: {optimized_model_dir}")


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
        "--max_gen_len", default=200, type=int, help="The maximum number of tokens that can be included in an answer"
    )
    parser.add_argument("--device", type=str, choices=["dml", "cuda"], default="dml")
    parser.add_argument(
        "--model_type",
        default="phi-2",
        choices=["llama-2-7b-chat", "mistral-7b-chat", "phi-2"],
        help="Which model to convert.",
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

    script_dir = Path(__file__).resolve().parent
    optimized_model_dir = script_dir / "models" / "optimized"

    repo_id = {
        "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
        "mistral-7b-chat": "mistralai/Mistral-7B-Instruct-v0.1",
        "phi-2": "microsoft/phi-2",
    }[args.model_type]

    model_name = repo_id.replace("/", "_")
    if args.optimize or not (optimized_model_dir / model_name).exists():
        optimize(optimized_model_dir, repo_id, model_name, args.num_layers)

    if not args.optimize:
        if args.interactive:
            launch_chat_app(args.expose_locally)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run_llm_io_binding(
                    optimized_model_dir / model_name,
                    args.prompt,
                    args.max_seq_len,
                    args.max_gen_len,
                    args.device,
                    args.device_id,
                )
