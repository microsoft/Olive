# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
import threading
import tkinter as tk
import tkinter.ttk as ttk
import warnings
from pathlib import Path

import config
import onnxruntime as ort
import torch
from diffusers import DiffusionPipeline, OnnxRuntimeModel
from diffusers.utils import load_image
from optimum.onnxruntime import ORTStableDiffusionXLImg2ImgPipeline, ORTStableDiffusionXLPipeline
from packaging import version
from PIL import Image, ImageTk

from olive.model import ONNXModel
from olive.workflows import run as olive_run


def run_inference_loop(
    pipeline,
    prompt,
    num_images,
    batch_size,
    image_size,
    num_inference_steps,
    base_images=None,
    image_callback=None,
    step_callback=None,
):
    images_saved = 0

    def update_steps(step, timestep, latents):
        if step_callback:
            step_callback((images_saved // batch_size) * num_inference_steps + step)

    print(f"\nInference Batch Start (batch size = {batch_size}).")

    if base_images is None:
        result = pipeline(
            [prompt] * batch_size,
            num_inference_steps=num_inference_steps,
            callback=update_steps if step_callback else None,
            height=image_size,
            width=image_size,
        )
    else:
        base_images_rgb = [load_image(base_image).convert("RGB") for base_image in base_images]

        result = pipeline(
            [prompt] * batch_size,
            negative_prompt=[""] * batch_size,
            image=base_images_rgb,
            num_inference_steps=num_inference_steps,
            callback=update_steps if step_callback else None,
        )

    for image_index in range(batch_size):
        if images_saved < num_images:
            image_suffix = "base" if base_images is None else "refined"
            output_path = f"result_{images_saved}_{image_suffix}.png"
            result.images[image_index].save(output_path)
            if image_callback:
                image_callback(images_saved, output_path)
            images_saved += 1
            print(f"Generated {output_path}")

    print("Inference Batch End.")


def run_refiner_inference_loop(
    pipeline, prompt, num_images, batch_size, base_images, num_inference_steps, image_callback=None, step_callback=None
):
    images_saved = 0

    def update_steps(step, timestep, latents):
        if step_callback:
            step_callback((images_saved // batch_size) * num_inference_steps + step)

    print(f"\nInference Batch Start (batch size = {batch_size}).")
    refiner_result = pipeline(
        [prompt] * batch_size,
        image=base_images,
        num_inference_steps=num_inference_steps,
        callback=update_steps if step_callback else None,
    )

    for image_index in range(batch_size):
        if images_saved < num_images:
            output_path = f"result_{images_saved}_refined.png"
            refiner_result.images[image_index].save(output_path)
            if image_callback:
                image_callback(images_saved, output_path)
            images_saved += 1
            print(f"Generated {output_path}")

    print("Inference Batch End.")


def run_inference_gui(pipeline, prompt, num_images, batch_size, image_size, num_inference_steps, base_images=None):
    def update_progress_bar(total_steps_completed):
        progress_bar["value"] = total_steps_completed

    def image_completed(index, path):
        img = Image.open(path)
        photo = ImageTk.PhotoImage(img)
        gui_images[index].config(image=photo)
        gui_images[index].image = photo
        if index == num_images - 1:
            generate_button["state"] = "normal"

    def on_generate_click():
        generate_button["state"] = "disabled"
        progress_bar["value"] = 0
        threading.Thread(
            target=run_inference_loop,
            args=(
                pipeline,
                prompt_textbox.get(),
                num_images,
                batch_size,
                image_size,
                num_inference_steps,
                base_images,
                image_completed,
                update_progress_bar,
            ),
        ).start()

    if num_images > 9:
        print("WARNING: interactive UI only supports displaying up to 9 images")
        num_images = 9

    image_rows = 1 + (num_images - 1) // 3
    image_cols = 2 if num_images == 4 else min(num_images, 3)
    min_batches_required = 1 + (num_images - 1) // batch_size

    bar_height = 10
    button_width = 80
    button_height = 30
    padding = 2
    window_width = image_cols * image_size + (image_cols + 1) * padding
    window_height = image_rows * image_size + (image_rows + 1) * padding + bar_height + button_height

    window = tk.Tk()
    window.title("Stable Diffusion")
    window.resizable(width=False, height=False)
    window.geometry(f"{window_width}x{window_height}")

    gui_images = []
    for row in range(image_rows):
        for col in range(image_cols):
            label = tk.Label(window, width=image_size, height=image_size, background="black")
            gui_images.append(label)
            label.place(x=col * image_size, y=row * image_size)

    y = image_rows * image_size + (image_rows + 1) * padding

    progress_bar = ttk.Progressbar(window, value=0, maximum=num_inference_steps * min_batches_required)
    progress_bar.place(x=0, y=y, height=bar_height, width=window_width)

    y += bar_height

    prompt_textbox = tk.Entry(window)
    prompt_textbox.insert(tk.END, prompt)
    prompt_textbox.place(x=0, y=y, width=window_width - button_width, height=button_height)

    generate_button = tk.Button(window, text="Generate", command=on_generate_click)
    generate_button.place(x=window_width - button_width, y=y, width=button_width, height=button_height)

    window.mainloop()


def run_inference(
    model_dir,
    prompt,
    num_images,
    batch_size,
    image_size,
    num_inference_steps,
    static_dims,
    interactive,
    base_images=None,
):
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
        sess_options.add_free_dimension_override_by_name("unet_sample_height", image_size // 8)
        sess_options.add_free_dimension_override_by_name("unet_sample_width", image_size // 8)
        sess_options.add_free_dimension_override_by_name("unet_time_batch", 1)
        sess_options.add_free_dimension_override_by_name("unet_hidden_batch", batch_size * 2)
        sess_options.add_free_dimension_override_by_name("unet_hidden_sequence", 77)
        sess_options.add_free_dimension_override_by_name("unet_text_embeds_batch", batch_size * 2)
        sess_options.add_free_dimension_override_by_name("unet_text_embeds_size", image_size + 256)
        sess_options.add_free_dimension_override_by_name("unet_time_ids_batch", batch_size * 2)
        sess_options.add_free_dimension_override_by_name("unet_time_ids_size", 6)

    if base_images is None:
        pipeline = ORTStableDiffusionXLPipeline.from_pretrained(
            model_dir, provider="DmlExecutionProvider", sess_options=sess_options
        )
    else:
        pipeline = ORTStableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_dir, provider="DmlExecutionProvider", sess_options=sess_options
        )

    if interactive:
        run_inference_gui(pipeline, prompt, num_images, batch_size, image_size, num_inference_steps, base_images)
    else:
        run_inference_loop(pipeline, prompt, num_images, batch_size, image_size, num_inference_steps, base_images)


def optimize(
    model_id: str,
    is_refiner_model: bool,
    unoptimized_model_dir: Path,
    optimized_model_dir: Path,
):
    from google.protobuf import __version__ as protobuf_version

    # protobuf 4.x aborts with OOM when optimizing unet
    if version.parse(protobuf_version) > version.parse("3.20.3"):
        print("This script requires protobuf 3.20.3. Please ensure your package version matches requirements.txt.")
        exit(1)

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
    pipeline = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

    model_info = dict()

    submodel_names = ["vae_encoder", "vae_decoder", "unet", "text_encoder_2"]

    if not is_refiner_model:
        submodel_names.append("text_encoder")

    for submodel_name in submodel_names:
        print(f"\nOptimizing {submodel_name}")

        olive_config = None
        with open(script_dir / f"config_{submodel_name}.json", "r") as fin:
            olive_config = json.load(fin)

        # TODO: Remove this once we figure out which nodes are causing the black screen
        if is_refiner_model and submodel_name == "vae_encoder":
            olive_config["passes"]["optimize"]["config"]["float16"] = False

        # TODO: Remove this once we figure out which nodes are causing the black screen
        if submodel_name == "vae_decoder":
            olive_config["passes"]["optimize"]["config"]["float16"] = False

        olive_config["input_model"]["config"]["model_path"] = model_id
        olive_run(olive_config)

        footprints_file_path = (
            Path(__file__).resolve().parent / "footprints" / f"{submodel_name}_gpu-dml_footprints.json"
        )
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

            unoptimized_olive_model = ONNXModel(**conversion_footprint["model_config"]["config"])
            optimized_olive_model = ONNXModel(**optimizer_footprint["model_config"]["config"])

            model_info[submodel_name] = {
                "unoptimized": {
                    "path": Path(unoptimized_olive_model.model_path),
                },
                "optimized": {
                    "path": Path(optimized_olive_model.model_path),
                },
            }

            print(f"Unoptimized Model : {model_info[submodel_name]['unoptimized']['path']}")
            print(f"Optimized Model   : {model_info[submodel_name]['optimized']['path']}")

    # Save the unoptimized models in a directory structure that the diffusers library can load and run.
    # This is optional, and the optimized models can be used directly in a custom pipeline if desired.
    print("\nCreating ONNX pipeline...")

    if getattr(pipeline, "feature_extractor", None) is not None:
        feature_extractor = pipeline.feature_extractor
    else:
        feature_extractor = None

    vae_encoder_session = OnnxRuntimeModel.load_model(
        model_info["vae_encoder"]["unoptimized"]["path"].parent / "model.onnx"
    )
    vae_decoder_session = OnnxRuntimeModel.load_model(
        model_info["vae_decoder"]["unoptimized"]["path"].parent / "model.onnx"
    )
    text_encoder_2_session = OnnxRuntimeModel.load_model(
        model_info["text_encoder_2"]["unoptimized"]["path"].parent / "model.onnx"
    )
    unet_session = OnnxRuntimeModel.load_model(model_info["unet"]["unoptimized"]["path"].parent / "model.onnx")

    if is_refiner_model:
        onnx_pipeline = ORTStableDiffusionXLImg2ImgPipeline(
            vae_encoder_session=vae_encoder_session,
            vae_decoder_session=vae_decoder_session,
            text_encoder_session=text_encoder_2_session,
            unet_session=unet_session,
            tokenizer=pipeline.tokenizer_2,
            scheduler=pipeline.scheduler,
            feature_extractor=feature_extractor,
            config=dict(pipeline.config),
        )
    else:
        text_encoder_session = OnnxRuntimeModel.load_model(
            model_info["text_encoder"]["unoptimized"]["path"].parent / "model.onnx"
        )

        onnx_pipeline = ORTStableDiffusionXLPipeline(
            vae_encoder_session=vae_encoder_session,
            vae_decoder_session=vae_decoder_session,
            text_encoder_session=text_encoder_session,
            unet_session=unet_session,
            text_encoder_2_session=text_encoder_2_session,
            tokenizer=pipeline.tokenizer,
            tokenizer_2=pipeline.tokenizer_2,
            scheduler=pipeline.scheduler,
            feature_extractor=feature_extractor,
            config=dict(pipeline.config),
        )

    print("Saving unoptimized models...")
    onnx_pipeline.save_pretrained(unoptimized_model_dir)

    # The refiner model pipeline expect text_encoder_2 and tokenizer_2, but since the ORT pipeline saves them as
    # text_encoder and tokenizer, we need to rename them
    if is_refiner_model:
        (unoptimized_model_dir / "text_encoder").rename(unoptimized_model_dir / "text_encoder_2")
        (unoptimized_model_dir / "tokenizer").rename(unoptimized_model_dir / "tokenizer_2")

    # Create a copy of the unoptimized model directory, then overwrite with optimized models from the olive cache.
    print("Copying optimized models...")
    shutil.copytree(unoptimized_model_dir, optimized_model_dir, ignore=shutil.ignore_patterns("weights.pb"))
    for submodel_name in submodel_names:
        src_path = model_info[submodel_name]["optimized"]["path"]
        dst_path = optimized_model_dir / submodel_name / "model.onnx"
        shutil.copyfile(src_path, dst_path)

        weights_src_path = src_path.parent / (src_path.name + ".data")
        if weights_src_path.is_file():
            weights_dst_path = dst_path.parent / (dst_path.name + ".data")
            shutil.copyfile(weights_src_path, weights_dst_path)

    print(f"The optimized pipeline is located here: {optimized_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-xl-base-1.0", type=str)
    parser.add_argument("--base_images", default=None, nargs="+")
    parser.add_argument("--interactive", action="store_true", help="Run with a GUI")
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--clean_cache", action="store_true", help="Deletes the Olive cache")
    parser.add_argument("--test_unoptimized", action="store_true", help="Use unoptimized model for inference")
    parser.add_argument(
        "--prompt",
        default=(
            "castle surrounded by water and nature, village, volumetric lighting, photorealistic, "
            "detailed and intricate, fantasy, epic cinematic shot, mountains, 8k ultra hd"
        ),
        type=str,
    )
    parser.add_argument("--num_images", default=1, type=int, help="Number of images to generate")
    parser.add_argument("--batch_size", default=1, type=int, help="Number of images to generate per batch")
    parser.add_argument("--num_inference_steps", default=50, type=int, help="Number of steps in diffusion process")
    parser.add_argument(
        "--static_dims",
        action="store_true",
        help="DEPRECATED (now enabled by default). Use --dynamic_dims to disable static_dims.",
    )
    parser.add_argument("--dynamic_dims", action="store_true", help="Disable static shape optimization")
    args = parser.parse_args()

    if args.static_dims:
        print(
            "WARNING: the --static_dims option is deprecated, and static shape optimization is enabled by default. "
            "Use --dynamic_dims to disable static shape optimization."
        )

    model_to_config = {
        "stabilityai/stable-diffusion-xl-base-1.0": {
            "image_size": 1024,
            "hidden_state_size": 2048,
            "time_ids_size": 6,
            "is_refiner_model": False,
        },
        "stabilityai/stable-diffusion-xl-refiner-1.0": {
            "image_size": 1024,
            "hidden_state_size": 1280,
            "time_ids_size": 5,
            "is_refiner_model": True,
        },
    }

    if args.model_id not in list(model_to_config.keys()):
        print(
            f"WARNING: {args.model_id} is not an officially supported model for this example and may not work as "
            + "expected."
        )

    if version.parse(ort.__version__) < version.parse("1.15.0"):
        print("This script requires onnxruntime-directml 1.15.0 or newer")
        exit(1)

    script_dir = Path(__file__).resolve().parent

    if args.clean_cache:
        shutil.rmtree(script_dir / "cache", ignore_errors=True)

    # Optimize the models
    unoptimized_model_dir = script_dir / "models" / "unoptimized" / args.model_id
    optimized_model_dir = script_dir / "models" / "optimized" / args.model_id

    model_config = model_to_config.get(args.model_id, {})
    config.image_size = model_config.get("image_size", 1024)
    config.hidden_state_size = model_config.get("hidden_state_size", 2048)
    config.time_ids_size = model_config.get("time_ids_size", 6)
    is_refiner_model = model_config.get("is_refiner_model", False)

    if is_refiner_model and not args.optimize and args.base_images is None:
        print("--base_images needs to be provided when executing a refiner model without --optimize")
        exit(1)

    if not is_refiner_model and args.base_images is not None:
        print("--base_images should only be provided for refiner models")
        exit(1)

    if args.optimize or not optimized_model_dir.exists():
        # TODO: clean up warning filter (mostly during conversion from torch to ONNX)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimize(args.model_id, is_refiner_model, unoptimized_model_dir, optimized_model_dir)

    # Run inference on the models
    if not args.optimize:
        unoptimized_model_dir = script_dir / "models" / "unoptimized" / args.model_id
        optimized_model_dir = script_dir / "models" / "optimized" / args.model_id

        model_dir = unoptimized_model_dir if args.test_unoptimized else optimized_model_dir
        use_static_dims = not args.dynamic_dims

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_inference(
                model_dir,
                args.prompt,
                args.num_images,
                args.batch_size,
                config.image_size,
                args.num_inference_steps,
                use_static_dims,
                args.interactive,
                args.base_images,
            )
