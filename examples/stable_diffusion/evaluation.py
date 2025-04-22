# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
import math
import os
import re
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
import requests
import torch
from datasets import load_dataset
from PIL import Image
from sd_utils.qdq import OnnxStableDiffusionPipelineWithSave
from torchmetrics.functional.multimodal import clip_score
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Clip score


def get_clip_score(prompt: str, path: Path, clip_score_fn):
    with torch.no_grad():
        image = Image.open(path / f"{prompt}.png")
        image = torch.tensor(np.array(image), dtype=torch.uint8).permute(2, 0, 1)
        score = clip_score_fn(image, prompt)
        return score.detach()


def get_clip_scores(prompts: list[str], path: Path, clip_score_fn):
    scores = []
    with open(path / "clip_scores.txt", "w") as f:
        f.write("| Prompt | Score |\n")
        for prompt in prompts:
            score = get_clip_score(prompt, path, clip_score_fn)
            scores.append(score)
            f.write(f"| {prompt} | {score} |\n")

        logger.info("CLIP Scores avg: %s", np.mean(np.array(scores)))
        f.write(f"| Avg | {np.mean(np.array(scores))} |\n")


# MSE score


def calc_error(image1, image2):
    image1 = Image.open(image1)
    image2 = Image.open(image2)
    image1 = np.array(image1, dtype=np.float32)
    image2 = np.array(image2, dtype=np.float32)
    return np.mean((image1 - image2) ** 2)


def get_mse_scores(prompts: list[str], unoptimized_path: Path, optimized_path: Path, train_num: int):
    train_error = []
    test_error = []
    with open(optimized_path / "mse_scores.txt", "w") as f:
        f.write("| Prompt | Error |\n")
        for i, prompt in enumerate(prompts):
            error = calc_error(unoptimized_path / f"{prompt}.png", optimized_path / f"{prompt}.png")
            f.write(f"| {prompt} | {error} |\n")
            if i < train_num:
                train_error.append(error)
            else:
                test_error.append(error)

        train_error = np.mean(np.array(train_error))
        test_error = np.mean(np.array(test_error))
        logger.info("Average train error %f", train_error)
        logger.info("Average test error %f", test_error)
        f.write(f"| Avg Train | {train_error} |\n")
        f.write(f"| Avg Test | {test_error} |\n")


# FID score


def get_fid_scores(prompts: list[str], path: Path, real_images):
    with torch.no_grad(), open(path / "fid_scores.txt", "w") as f:
        f.write("| Prompt | Score |\n")
        images = []
        for prompt in prompts:
            image = Image.open(path / f"{prompt}.png")
            image = torch.tensor(np.array(image), dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)
            images.append(image)
        images = torch.cat(images)

        fid = FrechetInceptionDistance()
        fid.update(real_images, real=True)
        fid.update(images, real=False)

        score = fid.compute()
        logger.info("FID: %f", score)
        f.write(f"| FID | {score} |\n")


# hpsv2 score


def get_hpsv2_scores(path: Path, generate_image: Callable[[str, str], None], prompt_style: str):
    if prompt_style is None:
        return
    import hpsv2

    # Get benchmark prompts (<style> = all, anime, concept-art, paintings, photo)
    all_prompts = hpsv2.benchmark_prompts(prompt_style)
    if prompt_style != "all":
        all_prompts = {prompt_style: all_prompts}
    for style, prompts in all_prompts.items():
        os.makedirs(path / style, exist_ok=True)
        for idx, prompt in enumerate(prompts):
            logger.info("Generating %s for %s [%d/%d]", prompt, style, idx + 1, len(prompts))
            output = path / style / f"{idx:05d}.jpg"
            generate_image(prompt, output)
    hpsv2.evaluate(path.as_posix(), hps_version="v2.1")


# prepare data


def sanitize_path(input_string):
    return re.sub(r"[^\w\-, ]", "", input_string.strip())


def download_file(url, save_path):
    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes

        # Write the content to the specified file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        logger.info("File successfully downloaded and saved to %s", save_path)
    except requests.exceptions.MissingSchema:
        logger.info("Error: Invalid URL. Please provide a valid URL.")
    except requests.exceptions.ConnectionError:
        logger.info("Error: Network issue. Please check your internet connection.")
    except requests.exceptions.HTTPError as http_err:
        logger.info("HTTP error occurred: %s", http_err)
    except Exception as e:
        logger.info("An error occurred: %s", e)


def get_real_images(train_data, num_data):
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "quantize_data_coco2017"
    with torch.no_grad():
        images = []
        for i, example in enumerate(train_data):
            if i >= num_data:
                break
            image_path = data_dir / example["file_name"]
            if not image_path.exists():
                download_file(example["coco_url"], image_path)
            image = Image.open(image_path).convert("RGB")
            image = torch.tensor(np.array(image), dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)
            # TODO(anyone): use resize?
            image = F.center_crop(image, (256, 256))
            images.append(image)
        return torch.cat(images)


def run_inference(pipeline, args, prompt: str, output_path: Path, pathIsFile: bool = False):
    output = output_path if pathIsFile else output_path / f"{prompt}.png"
    if output.exists():
        return
    generator = None if args.seed is None else np.random.RandomState(seed=args.seed)
    result = pipeline(
        [prompt],
        num_inference_steps=args.num_inference_steps,
        height=args.image_size,
        width=args.image_size,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    result.images[0].save(output)


def parse_args(raw_args):
    import argparse

    parser = argparse.ArgumentParser("Common arguments")
    parser.add_argument("--save_data", action="store_true")
    parser.add_argument("--model_id", default="CompVis/stable-diffusion-v1-4", type=str)
    parser.add_argument(
        "--guidance_scale",
        default=7.5,
        type=float,
        help="Guidance scale as defined in Classifier-Free Diffusion Guidance",
    )
    parser.add_argument("--num_inference_steps", default=50, type=int, help="Number of steps in diffusion process")
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
        help="The seed to give to the generator to generate deterministic results.",
    )
    parser.add_argument("--data_dir", default="quantize_data", type=str)
    parser.add_argument(
        "--sub_dir", default="optimized", type=str, help="Sub directory to save the data for optimized model test"
    )
    parser.add_argument("--num_data", default=10, type=int)
    parser.add_argument("--train_ratio", default=0.5, type=float)
    parser.add_argument("--image_size", default=512, type=int, help="Width and height of the images to generate")
    parser.add_argument("--hpsv2_style", default=None, type=str, help="Style for hpsv2 benchmark")
    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = parse_args(raw_args)
    real_images = None
    dataset = load_dataset("phiyodr/coco2017", streaming=True)
    train_data = dataset["train"]
    prompts = [sanitize_path(example["captions"][0]) for i, example in enumerate(train_data) if i < args.num_data]
    real_images = get_real_images(train_data, args.num_data)

    train_num = math.floor(len(prompts) * args.train_ratio)
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    script_dir = Path()  # Path(__file__).resolve().parent
    unoptimized_path: Path = script_dir / args.data_dir / "unoptimized"
    optimized_path: Path = script_dir / args.data_dir / args.sub_dir

    unoptimized_model_dir = script_dir / "models" / "unoptimized" / args.model_id
    optimized_dir_name = "optimized-cpu_qdq"
    optimized_model_dir = script_dir / "models" / optimized_dir_name / args.model_id

    model_dir = unoptimized_model_dir if args.save_data else optimized_model_dir
    pipeline = OnnxStableDiffusionPipelineWithSave.from_pretrained(model_dir, provider="CPUExecutionProvider")
    pipeline.save_data_dir = None

    if args.save_data:
        os.makedirs(unoptimized_path, exist_ok=True)
        for i, prompt in enumerate(prompts):
            logger.info(prompt)
            if i < train_num:
                pipeline.save_data_dir = script_dir / args.data_dir / "data" / prompt
                os.makedirs(pipeline.save_data_dir, exist_ok=True)
            else:
                pipeline.save_data_dir = None
            run_inference(pipeline, args, prompt, unoptimized_path)
        get_clip_scores(prompts, unoptimized_path, clip_score_fn)
        get_fid_scores(prompts, unoptimized_path, real_images)
        get_hpsv2_scores(
            unoptimized_path / "hpsv2", partial(run_inference, pipeline, args, pathIsFile=True), args.hpsv2_style
        )

    else:
        os.makedirs(optimized_path, exist_ok=True)
        for prompt in prompts:
            logger.info(prompt)
            run_inference(pipeline, args, prompt, optimized_path)

        get_clip_scores(prompts, optimized_path, clip_score_fn)
        get_fid_scores(prompts, optimized_path, real_images)
        get_mse_scores(prompts, unoptimized_path, optimized_path, train_num)
        get_hpsv2_scores(
            optimized_path / "hpsv2", partial(run_inference, pipeline, args, pathIsFile=True), args.hpsv2_style
        )


if __name__ == "__main__":
    main()
