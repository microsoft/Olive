# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
import shutil
import torch
import argparse
import json
import warnings
from diffusers import OnnxStableDiffusionPipeline, OnnxRuntimeModel, StableDiffusionPipeline
import onnxruntime as ort

from olive.workflows import run as olive_run


def run_inference(optimized_model_dir, prompt, num_images, batch_size):
    ort.set_default_logger_severity(3)

    print("Loading models into ORT session...")
    sess_options = ort.SessionOptions()
    sess_options.enable_mem_pattern = False
    pipeline = OnnxStableDiffusionPipeline.from_pretrained(
        optimized_model_dir, provider="DmlExecutionProvider", sess_options=sess_options
    )

    images_saved = 0
    while images_saved < num_images:
        print(f"\nInference Batch Start (batch size = {batch_size}).")
        result = pipeline([prompt] * batch_size)
        passed_safety_checker = 0

        for image_index in range(batch_size):
            if not result.nsfw_content_detected[image_index]:
                passed_safety_checker += 1
                if images_saved < num_images:
                    output_path = f"result_{images_saved}.png"
                    result.images[image_index].save(output_path)
                    images_saved += 1
                    print(f"Generated {output_path}")

        print(f"Inference Batch End ({passed_safety_checker}/{batch_size} images passed the safety checker).")


def optimize(model_name: str, unoptimized_model_dir: Path, optimized_model_dir: Path):
    ort.set_default_logger_severity(4)

    model_info = dict()

    for submodel_name in ("text_encoder", "vae_encoder", "vae_decoder", "safety_checker", "unet"):
        # Optimize the model with Olive
        print(f"\nOptimizing {submodel_name}")
        olive_run(str(script_dir / f"config_{submodel_name}.json"))

        footprints_file_path = Path(__file__).resolve().parent / "footprints" / f"{submodel_name}_footprints.json"
        with footprints_file_path.open("r") as footprint_file:
            footprints = json.load(footprint_file)
            unoptimized_config = footprints[list(footprints)[0]]["model_config"]["config"]
            optimized_config = footprints[list(footprints)[1]]["model_config"]["config"]

            model_info[submodel_name] = {
                "unoptimized": {
                    "path": Path(unoptimized_config["model_path"]),
                    "is_file": unoptimized_config["is_file"],
                },
                "optimized": {
                    "path": Path(optimized_config["model_path"]),
                    "is_file": optimized_config["is_file"],
                },
            }

            print(f"Unoptimized Model : {model_info[submodel_name]['unoptimized']['path']}")
            print(f"Optimized Model   : {model_info[submodel_name]['optimized']['path']}")

    # Save the unoptimized models in a directory structure that the diffusers library can load and run.
    # This is optional, and the optimized models can be used directly in a custom pipeline if desired.
    pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    onnx_pipeline = OnnxStableDiffusionPipeline(
        vae_encoder=OnnxRuntimeModel.from_pretrained(model_info["vae_encoder"]["unoptimized"]["path"].parent),
        vae_decoder=OnnxRuntimeModel.from_pretrained(model_info["vae_decoder"]["unoptimized"]["path"].parent),
        text_encoder=OnnxRuntimeModel.from_pretrained(model_info["text_encoder"]["unoptimized"]["path"].parent),
        tokenizer=pipeline.tokenizer,
        unet=OnnxRuntimeModel.from_pretrained(model_info["unet"]["unoptimized"]["path"].parent),
        scheduler=pipeline.scheduler,
        safety_checker=OnnxRuntimeModel.from_pretrained(model_info["safety_checker"]["unoptimized"]["path"].parent),
        feature_extractor=pipeline.feature_extractor,
        requires_safety_checker=True,
    )
    onnx_pipeline.save_pretrained(unoptimized_model_dir)

    # Create a copy of the unoptimized model directory, then overwrite with optimized models from the olive cache.
    shutil.copytree(unoptimized_model_dir, optimized_model_dir, ignore=shutil.ignore_patterns('weights.pb'))
    for submodel_name in ("text_encoder", "vae_encoder", "vae_decoder", "safety_checker", "unet"):
        src_path = model_info[submodel_name]["optimized"]["path"]
        dst_path = optimized_model_dir / submodel_name / "model.onnx"
        shutil.copyfile(src_path, dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--clean_cache", action="store_true", help="Deletes the Olive cache")
    parser.add_argument("--test_unoptimized", action="store_true", help="Use unoptimized model for inference")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", type=str)
    parser.add_argument("--prompt", default="a photo of an astronaut riding a horse on mars.", type=str)
    parser.add_argument("--num_images", default=1, type=int, help="Number of images to generate")
    parser.add_argument("--batch_size", default=1, type=int, help="Number of images to generate per batch")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    unoptimized_model_dir = script_dir / "models" / "unoptimized" / args.model
    optimized_model_dir = script_dir / "models" / "optimized" / args.model

    if args.clean_cache:
        shutil.rmtree(script_dir / "cache", ignore_errors=True)

    if args.optimize:
        shutil.rmtree(script_dir / "footprints", ignore_errors=True)
        shutil.rmtree(unoptimized_model_dir, ignore_errors=True)
        shutil.rmtree(optimized_model_dir, ignore_errors=True)

    if args.optimize or not optimized_model_dir.exists():
        # TODO: clean up warning filter (mostly during conversion from torch to ONNX)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimize(args.model, unoptimized_model_dir, optimized_model_dir)

    if not args.optimize:
        model_dir = unoptimized_model_dir if args.test_unoptimized else optimized_model_dir
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_inference(model_dir, args.prompt, args.num_images, args.batch_size)
