# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
import shutil
import torch
import argparse
import json
from diffusers import OnnxStableDiffusionPipeline, OnnxRuntimeModel, StableDiffusionPipeline
import onnxruntime as ort

from olive.workflows import run as olive_run


def run_inference(optimized_model_dir):
    ort.set_default_logger_severity(3)

    sess_options = ort.SessionOptions()
    sess_options.enable_mem_pattern = False
    pipe = OnnxStableDiffusionPipeline.from_pretrained(
        optimized_model_dir, provider="DmlExecutionProvider", sess_options=sess_options
    )

    # Keep running until a result passes the safety checker
    succeeded = False
    attempt = 1
    while not succeeded and attempt < 10:
        result = pipe(args.prompt)
        print(f"NSFW Content: {result.nsfw_content_detected[0]}")
        succeeded = not result.nsfw_content_detected[0]
        attempt += 1

    result.images[0].save("./result.png")


def optimize(original_model: str, optimized_model_dir: Path):
    script_dir = Path(__file__).resolve().parent
    olive_model_cache_dir = script_dir / "cache" / "models"

    original_model_paths = {}
    optimized_model_paths = {}

    for model_name in ("text_encoder", "vae_encoder", "vae_decoder", "safety_checker", "unet"):
        # TODO: load and update JSON configs to match the original model

        # Optimize the model with Olive
        best_execution = olive_run(str(script_dir / f"config_{model_name}.json"))

        print("")
        print(f"Model: {model_name}")

        # Save path to the original ONNX model
        conversion_pass_id = 0
        model_info_json_path = olive_model_cache_dir / (best_execution["model_ids"][conversion_pass_id] + ".json")
        with model_info_json_path.open("r") as model_info_json_file:
            model_info = json.load(model_info_json_file)
            original_model_paths[model_name] = Path(model_info["config"]["model_path"])
            print(f"Original Model: {original_model_paths[model_name]}")

        # Save path to the optimized model
        optimization_pass_id = 1
        model_info_json_path = olive_model_cache_dir / (best_execution["model_ids"][optimization_pass_id] + ".json")
        with model_info_json_path.open("r") as model_info_json_file:
            model_info = json.load(model_info_json_file)
            optimized_model_paths[model_name] = Path(model_info["config"]["model_path"])
            print(f"Optimized Model: {optimized_model_paths[model_name]}")

    # Save the original models in a directory structure that the diffusers library can load and run.
    # This is optional, and the optimized models can be used directly in a custom pipeline if desired.
    pipeline = StableDiffusionPipeline.from_pretrained(original_model, torch_dtype=torch.float32)
    onnx_pipeline = OnnxStableDiffusionPipeline(
        vae_encoder=OnnxRuntimeModel.from_pretrained(original_model_paths["vae_encoder"].parent),
        vae_decoder=OnnxRuntimeModel.from_pretrained(original_model_paths["vae_decoder"].parent),
        text_encoder=OnnxRuntimeModel.from_pretrained(original_model_paths["text_encoder"].parent),
        tokenizer=pipeline.tokenizer,
        unet=OnnxRuntimeModel.from_pretrained(original_model_paths["unet"].parent),
        scheduler=pipeline.scheduler,
        safety_checker=OnnxRuntimeModel.from_pretrained(original_model_paths["safety_checker"].parent),
        feature_extractor=pipeline.feature_extractor,
        requires_safety_checker=True,
    )
    onnx_pipeline.save_pretrained(optimized_model_dir)

    # Copy the optimized models.
    for model_name in ("text_encoder", "vae_encoder", "vae_decoder", "safety_checker", "unet"):
        src_path = optimized_model_paths[model_name]
        dst_path = optimized_model_dir / model_name / "model.onnx"
        shutil.copyfile(src_path, dst_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Optimize the models with Olive (no inference)")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", type=str)
    parser.add_argument("--prompt", default="a photo of an astronaut riding a horse on mars.", type=str)
    args = parser.parse_args()

    optimized_model_dir = Path(__file__).resolve().parent / "models" / args.model

    if args.optimize or not Path(optimized_model_dir).exists():
        optimize(args.model, optimized_model_dir)

    if not args.optimize:
        run_inference(optimized_model_dir)
