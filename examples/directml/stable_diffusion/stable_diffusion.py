# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
import warnings
from pathlib import Path

import onnxruntime as ort
import PySimpleGUI as sg
import torch
from diffusers import OnnxRuntimeModel, OnnxStableDiffusionPipeline, StableDiffusionPipeline
from packaging import version

from olive.workflows import run as olive_run


def run_inference_loop(pipeline, prompt, num_images, batch_size, num_inference_steps, callback=None):
    images_saved = 0
    while images_saved < num_images:
        print(f"\nInference Batch Start (batch size = {batch_size}).")
        result = pipeline([prompt] * batch_size, num_inference_steps=num_inference_steps, callback=callback)
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


def run_inference(optimized_model_dir, prompt, num_images, batch_size, num_inference_steps, static_dims, interactive):
    ort.set_default_logger_severity(3)

    print("Loading models into ORT session...")
    sess_options = ort.SessionOptions()
    sess_options.enable_mem_pattern = False

    if static_dims:
        # Not necessary, but helps DML EP further optimize runtime performance.
        # batch_size is doubled for sample & hidden state because of classifier free guidance:
        # https://github.com/huggingface/diffusers/blob/46c52f9b9607e6ecb29c782c052aea313e6487b7/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L672
        sess_options.add_free_dimension_override_by_name("unet_sample_batch", batch_size * 2)
        sess_options.add_free_dimension_override_by_name("unet_sample_channels", 4)
        sess_options.add_free_dimension_override_by_name("unet_sample_height", 64)
        sess_options.add_free_dimension_override_by_name("unet_sample_width", 64)
        sess_options.add_free_dimension_override_by_name("unet_time_batch", batch_size)
        sess_options.add_free_dimension_override_by_name("unet_hidden_batch", batch_size * 2)
        sess_options.add_free_dimension_override_by_name("unet_hidden_sequence", 77)

    pipeline = OnnxStableDiffusionPipeline.from_pretrained(
        optimized_model_dir, provider="DmlExecutionProvider", sess_options=sess_options
    )

    if interactive:
        sg.theme("SystemDefault")

        layout = [
            [sg.Image(key="sd_output", size=(512, 512), background_color="black")],
            [sg.ProgressBar(num_inference_steps, key="sb_progress", expand_x=True, size=(8, 8))],
            [sg.InputText(key="sd_prompt", default_text=prompt, expand_x=True), sg.Button("Generate")],
        ]

        window = sg.Window("Stable Diffusion", layout)

        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED:
                break
            elif event == "Generate":

                def update_progress_bar(step, timestep, latents):
                    window["sb_progress"].update_bar(step)

                def generate_image():
                    run_inference_loop(pipeline, values["sd_prompt"], 1, 1, num_inference_steps, update_progress_bar)

                window["Generate"].update(disabled=True)
                window.start_thread(generate_image, "image_generation_done")
            elif event == "image_generation_done":
                window["sd_output"].update(filename="result_0.png")
                window["Generate"].update(disabled=False)

    else:
        run_inference_loop(pipeline, prompt, num_images, batch_size, num_inference_steps)


def optimize(model_name: str, unoptimized_model_dir: Path, optimized_model_dir: Path, optimize_provider: str):
    ort.set_default_logger_severity(4)
    script_dir = Path(__file__).resolve().parent

    # Clean up previously optimized models, if any.
    shutil.rmtree(script_dir / "footprints", ignore_errors=True)
    shutil.rmtree(unoptimized_model_dir, ignore_errors=True)
    shutil.rmtree(optimized_model_dir, ignore_errors=True)

    # Load the entire PyTorch pipeline to ensure all models and their configurations are downloaded and cached.
    # This avoids an issue where the non-ONNX components (tokenizer, scheduler, and feature extractor) are not
    # automatically cached correctly if individual models are fetched one at a time.
    print("Download stable diffusion PyTorch pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)

    model_info = dict()

    for submodel_name in ("text_encoder", "vae_encoder", "vae_decoder", "safety_checker", "unet"):
        # Optimize the model with Olive
        print(f"\nOptimizing {submodel_name}")

        olive_config = None
        with open(script_dir / f"config_{submodel_name}.json", "r") as fin:
            olive_config = json.load(fin)
        if optimize_provider:
            olive_config["passes"]["optimize"]["config"]["target_provider"] = optimize_provider

        olive_run(olive_config)

        # TODO: rename the 0 prefix in the path when the hardware accelerator feature is implemented.
        footprints_file_path = Path(__file__).resolve().parent / "footprints" / f"{submodel_name}_0_footprints.json"
        with footprints_file_path.open("r") as footprint_file:
            footprints = json.load(footprint_file)

            conversion_footprint = None
            optimizer_footprint = None
            for _, footprint in footprints.items():
                if footprint["from_pass"] == "OnnxConversion":
                    conversion_footprint = footprint
                elif footprint["from_pass"] == "OrtTransformersOptimization":
                    optimizer_footprint = footprint

            assert conversion_footprint and optimizer_footprint

            unoptimized_config = conversion_footprint["model_config"]["config"]
            optimized_config = optimizer_footprint["model_config"]["config"]

            model_info[submodel_name] = {
                "unoptimized": {
                    "path": Path(unoptimized_config["model_path"]),
                },
                "optimized": {
                    "path": Path(optimized_config["model_path"]),
                },
            }

            print(f"Unoptimized Model : {model_info[submodel_name]['unoptimized']['path']}")
            print(f"Optimized Model   : {model_info[submodel_name]['optimized']['path']}")

    # Save the unoptimized models in a directory structure that the diffusers library can load and run.
    # This is optional, and the optimized models can be used directly in a custom pipeline if desired.
    print("\nCreating ONNX pipeline...")
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

    print("Saving unoptimized models...")
    onnx_pipeline.save_pretrained(unoptimized_model_dir)

    # Create a copy of the unoptimized model directory, then overwrite with optimized models from the olive cache.
    print("Copying optimized models...")
    shutil.copytree(unoptimized_model_dir, optimized_model_dir, ignore=shutil.ignore_patterns("weights.pb"))
    for submodel_name in ("text_encoder", "vae_encoder", "vae_decoder", "safety_checker", "unet"):
        src_path = model_info[submodel_name]["optimized"]["path"]
        dst_path = optimized_model_dir / submodel_name / "model.onnx"
        shutil.copyfile(src_path, dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Run with a GUI")
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--optimize_provider", type=str, default="directml_future", help="EP target for inference")
    parser.add_argument("--clean_cache", action="store_true", help="Deletes the Olive cache")
    parser.add_argument("--test_unoptimized", action="store_true", help="Use unoptimized model for inference")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", type=str)
    parser.add_argument("--prompt", default="cyberpunk dog, glasses, neon, bokeh, close up", type=str)
    parser.add_argument("--num_images", default=1, type=int, help="Number of images to generate")
    parser.add_argument("--batch_size", default=1, type=int, help="Number of images to generate per batch")
    parser.add_argument("--num_inference_steps", default=50, type=int, help="Number of steps in diffusion process")
    # This should be on by default in ORT 1.15.0 when symbolic shape inference issue is addressed:
    # https://github.com/microsoft/onnxruntime/issues/15521
    parser.add_argument("--static_dims", action="store_true", help="Fixes dynamic shapes to static for better perf")
    args = parser.parse_args()

    if version.parse(ort.__version__) < version.parse("1.15.0"):
        print("This script requires onnxruntime-directml 1.15.0 or newer")
        exit(1)

    script_dir = Path(__file__).resolve().parent
    unoptimized_model_dir = script_dir / "models" / "unoptimized" / args.model
    optimized_model_dir = script_dir / "models" / "optimized" / args.model

    if args.clean_cache:
        shutil.rmtree(script_dir / "cache", ignore_errors=True)

    if args.optimize or not optimized_model_dir.exists():
        # TODO: clean up warning filter (mostly during conversion from torch to ONNX)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimize(args.model, unoptimized_model_dir, optimized_model_dir, args.optimize_provider)

    if not args.optimize:
        model_dir = unoptimized_model_dir if args.test_unoptimized else optimized_model_dir
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_inference(
                model_dir,
                args.prompt,
                args.num_images,
                args.batch_size,
                args.num_inference_steps,
                args.static_dims,
                args.interactive,
            )
