# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path
from typing import Dict

import config
import numpy as np
import torch
from diffusers import DiffusionPipeline
from packaging import version
from user_script import get_base_model_name

from olive.common.utils import set_tempdir
from olive.workflows import run as olive_run

# pylint: disable=redefined-outer-name
# ruff: noqa: TID252, T201


def save_image(result, batch_size, provider, num_images, images_saved, image_callback=None):
    passed_safety_checker = 0
    for image_index in range(batch_size):
        if result.nsfw_content_detected is None or not result.nsfw_content_detected[image_index]:
            passed_safety_checker += 1
            if images_saved < num_images:
                output_path = f"result_{images_saved}.png"
                result.images[image_index].save(output_path)
                if image_callback:
                    image_callback(images_saved, output_path)
                images_saved += 1
                print(f"Generated {output_path}")
    print(f"Inference Batch End ({passed_safety_checker}/{batch_size} images).")
    if provider == "openvino":
        print("WARNING: Safety checker is not supported by OpenVINO. It will be disabled.")
    else:
        print("Images passed the safety checker.")
    return images_saved


def run_inference_loop(
    pipeline,
    prompt,
    num_images,
    batch_size,
    image_size,
    num_inference_steps,
    guidance_scale,
    strength: float,
    provider: str,
    generator=None,
    image_callback=None,
    step_callback=None,
):
    images_saved = 0

    def update_steps(step, timestep, latents):
        if step_callback:
            step_callback((images_saved // batch_size) * num_inference_steps + step)

    while images_saved < num_images:
        print(f"\nInference Batch Start (batch size = {batch_size}).")

        kwargs = {"strength": strength} if provider == "openvino" else {}

        result = pipeline(
            [prompt] * batch_size,
            num_inference_steps=num_inference_steps,
            callback=update_steps if step_callback else None,
            height=image_size,
            width=image_size,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs,
        )

        images_saved = save_image(result, batch_size, provider, num_images, images_saved, image_callback)


def run_inference_gui(
    pipeline,
    prompt,
    num_images,
    batch_size,
    image_size,
    num_inference_steps,
    guidance_scale,
    strength,
    provider,
    generator,
):
    import threading
    import tkinter as tk
    import tkinter.ttk as ttk

    from PIL import Image, ImageTk

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
                guidance_scale,
                strength,
                provider,
                generator,
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


def update_config_with_provider(config: Dict, provider: str):
    if provider == "dml":
        # DirectML EP is the default, so no need to update config.
        return config
    elif provider == "cuda":
        from sd_utils.ort import update_cuda_config

        return update_cuda_config(config)
    elif provider == "openvino":
        from sd_utils.ov import update_ov_config

        return update_ov_config(config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def optimize(
    model_id: str,
    provider: str,
    unoptimized_model_dir: Path,
    optimized_model_dir: Path,
):
    from google.protobuf import __version__ as protobuf_version

    # protobuf 4.x aborts with OOM when optimizing unet
    if version.parse(protobuf_version) > version.parse("3.20.3"):
        print("This script requires protobuf 3.20.3. Please ensure your package version matches requirements.txt.")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent

    # Clean up previously optimized models, if any.
    shutil.rmtree(script_dir / "footprints", ignore_errors=True)
    shutil.rmtree(unoptimized_model_dir, ignore_errors=True)
    shutil.rmtree(optimized_model_dir, ignore_errors=True)

    # The model_id and base_model_id are identical when optimizing a standard stable diffusion model like
    # runwayml/stable-diffusion-v1-5. These variables are only different when optimizing a LoRA variant.
    base_model_id = get_base_model_name(model_id)

    # Load the entire PyTorch pipeline to ensure all models and their configurations are downloaded and cached.
    # This avoids an issue where the non-ONNX components (tokenizer, scheduler, and feature extractor) are not
    # automatically cached correctly if individual models are fetched one at a time.
    print("Download stable diffusion PyTorch pipeline...")
    pipeline = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32)
    config.vae_sample_size = pipeline.vae.config.sample_size
    config.cross_attention_dim = pipeline.unet.config.cross_attention_dim
    config.unet_sample_size = pipeline.unet.config.sample_size

    model_info = {}

    submodel_names = ["vae_encoder", "vae_decoder", "unet", "text_encoder"]

    has_safety_checker = getattr(pipeline, "safety_checker", None) is not None

    if has_safety_checker:
        if provider == "openvino":
            print("WARNING: Safety checker is not supported by OpenVINO. It will be disabled.")
        else:
            submodel_names.append("safety_checker")

    for submodel_name in submodel_names:
        print(f"\nOptimizing {submodel_name}")

        olive_config = None
        with (script_dir / f"config_{submodel_name}.json").open() as fin:
            olive_config = json.load(fin)
        olive_config = update_config_with_provider(olive_config, provider)

        if submodel_name in ("unet", "text_encoder"):
            olive_config["input_model"]["model_path"] = model_id
        else:
            # Only the unet & text encoder are affected by LoRA, so it's better to use the base model ID for
            # other models: the Olive cache is based on the JSON config, and two LoRA variants with the same
            # base model ID should be able to reuse previously optimized copies.
            olive_config["input_model"]["model_path"] = base_model_id

        run_res = olive_run(olive_config)

        if provider == "openvino":
            from sd_utils.ov import save_optimized_ov_submodel

            save_optimized_ov_submodel(run_res, submodel_name, optimized_model_dir, model_info)
        else:
            from sd_utils.ort import save_optimized_onnx_submodel

            save_optimized_onnx_submodel(submodel_name, provider, model_info)

    if provider == "openvino":
        from sd_utils.ov import save_ov_model_info

        save_ov_model_info(model_info, optimized_model_dir)
    else:
        from sd_utils.ort import save_onnx_pipeline

        save_onnx_pipeline(
            has_safety_checker, model_info, optimized_model_dir, unoptimized_model_dir, pipeline, submodel_names
        )

    return model_info


def parse_common_args(raw_args):
    parser = argparse.ArgumentParser("Common arguments")

    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5", type=str)
    parser.add_argument(
        "--provider", default="dml", type=str, choices=["dml", "cuda", "openvino"], help="Execution provider to use"
    )
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--clean_cache", action="store_true", help="Deletes the Olive cache")
    parser.add_argument("--test_unoptimized", action="store_true", help="Use unoptimized model for inference")
    parser.add_argument("--batch_size", default=1, type=int, help="Number of images to generate per batch")
    parser.add_argument(
        "--prompt",
        default=(
            "castle surrounded by water and nature, village, volumetric lighting, photorealistic, "
            "detailed and intricate, fantasy, epic cinematic shot, mountains, 8k ultra hd"
        ),
        type=str,
    )
    parser.add_argument(
        "--guidance_scale",
        default=7.5,
        type=float,
        help="Guidance scale as defined in Classifier-Free Diffusion Guidance",
    )
    parser.add_argument("--num_images", default=1, type=int, help="Number of images to generate")
    parser.add_argument("--num_inference_steps", default=50, type=int, help="Number of steps in diffusion process")
    parser.add_argument("--interactive", action="store_true", help="Run with a GUI")
    parser.add_argument("--tempdir", default=None, type=str, help="Root directory for tempfile directories and files")
    parser.add_argument(
        "--strength",
        default=1.0,
        type=float,
        help=(
            "Value between 0.0 and 1.0, that controls the amount of noise that is added to the input image. "
            "Values that approach 1.0 enable lots of variations but will also produce images "
            "that are not semantically consistent with the input."
        ),
    )
    parser.add_argument("--image_size", default=512, type=int, help="Width and height of the images to generate")
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="The seed to give to the generator to generate deterministic results.",
    )

    return parser.parse_known_args(raw_args)


def parse_ort_args(raw_args):
    parser = argparse.ArgumentParser("ONNX Runtime arguments")

    parser.add_argument(
        "--static_dims",
        action="store_true",
        help="DEPRECATED (now enabled by default). Use --dynamic_dims to disable static_dims.",
    )
    parser.add_argument("--dynamic_dims", action="store_true", help="Disable static shape optimization")

    return parser.parse_known_args(raw_args)


def parse_ov_args(raw_args):
    parser = argparse.ArgumentParser("OpenVINO arguments")

    parser.add_argument("--device", choices=["CPU", "GPU", "VPU"], default="CPU", type=str)
    parser.add_argument("--image_path", default=None, type=str)
    parser.add_argument("--img_to_img_example", action="store_true", help="Runs the image to image example")

    return parser.parse_known_args(raw_args)


def main(raw_args=None):
    common_args, extra_args = parse_common_args(raw_args)

    provider = common_args.provider
    model_id = common_args.model_id

    script_dir = Path(__file__).resolve().parent
    unoptimized_model_dir = script_dir / "models" / "unoptimized" / model_id
    optimized_dir_name = f"optimized-{provider}"
    optimized_model_dir = script_dir / "models" / optimized_dir_name / model_id

    if common_args.clean_cache:
        shutil.rmtree(script_dir / "cache", ignore_errors=True)

    guidance_scale = common_args.guidance_scale

    if model_id == "stabilityai/sd-turbo" and guidance_scale > 0:
        guidance_scale = 0.0
        print(f"WARNING: Classifier free guidance has been forcefully disabled since {model_id} doesn't support it.")

    ov_args, ort_args = None, None
    if provider == "openvino":
        ov_args, extra_args = parse_ov_args(extra_args)
    else:
        ort_args, extra_args = parse_ort_args(extra_args)

    if common_args.optimize or not optimized_model_dir.exists():
        set_tempdir(common_args.tempdir)

        # TODO(jstoecker): clean up warning filter (mostly during conversion from torch to ONNX)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if provider != "openvino":
                from sd_utils.ort import validate_args

                validate_args(ort_args, common_args.provider)
            optimize(common_args.model_id, common_args.provider, unoptimized_model_dir, optimized_model_dir)

    generator = None if common_args.seed is None else np.random.RandomState(seed=common_args.seed)

    if not common_args.optimize:
        model_dir = unoptimized_model_dir if common_args.test_unoptimized else optimized_model_dir
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if provider == "openvino":
                from sd_utils.ov import get_ov_pipeline

                pipeline = get_ov_pipeline(common_args, ov_args, optimized_model_dir)
            else:
                from sd_utils.ort import get_ort_pipeline

                pipeline = get_ort_pipeline(model_dir, common_args, ort_args, guidance_scale)
            if provider == "openvino" and (ov_args.image_path or ov_args.img_to_img_example):
                res = None
                if ov_args.image_path:
                    from sd_utils.ov import run_ov_image_inference

                    res = run_ov_image_inference(
                        pipeline,
                        ov_args.image_path,
                        common_args.prompt,
                        common_args.strength,
                        guidance_scale,
                        common_args.image_size,
                        common_args.num_inference_steps,
                        common_args,
                        generator=generator,
                    )
                if ov_args.img_to_img_example:
                    from sd_utils.ov import run_ov_img_to_img_example

                    res = run_ov_img_to_img_example(pipeline, guidance_scale, common_args)
                save_image(res, common_args.batch_size, "openvino", common_args.num_images, 0)
                sys.exit(0)

            if common_args.interactive:
                run_inference_gui(
                    pipeline,
                    common_args.prompt,
                    common_args.num_images,
                    common_args.batch_size,
                    common_args.image_size,
                    common_args.num_inference_steps,
                    guidance_scale,
                    common_args.strength,
                    provider=provider,
                    generator=generator,
                )
            else:
                run_inference_loop(
                    pipeline,
                    common_args.prompt,
                    common_args.num_images,
                    common_args.batch_size,
                    common_args.image_size,
                    common_args.num_inference_steps,
                    guidance_scale,
                    common_args.strength,
                    provider=provider,
                    generator=generator,
                )


if __name__ == "__main__":
    main()
