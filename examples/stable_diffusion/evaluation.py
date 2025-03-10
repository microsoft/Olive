# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
from pathlib import Path
from PIL import Image
import logging
import sys
import math
import numpy as np
from sd_utils.qnn import QnnStableDiffusionPipeline
from datasets import load_dataset
from torchmetrics.functional.multimodal import clip_score
from functools import partial
from torchvision.transforms import functional as F
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import io

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


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
    parser.add_argument("--num_data", default=10, type=int)
    parser.add_argument("--dataset", default="phiyodr/coco2017", type=str)
    parser.add_argument("--train_ratio", default=0.5, type=float)
    return parser.parse_args(raw_args)


def calc_error(image1, image2):
    image1 = Image.open(image1)
    image2 = Image.open(image2)
    image1 = np.array(image1, dtype=np.float32)
    image2 = np.array(image2, dtype=np.float32)
    mse = np.mean((image1 - image2) ** 2)
    return mse


def run_inference(pipeline, args, prompt: str, output_path: Path):
    generator = None if args.seed is None else np.random.RandomState(seed=args.seed)
    result = pipeline(
        [prompt],
        num_inference_steps = args.num_inference_steps,
        height=512,
        width=512,
        guidance_scale=args.guidance_scale,
        generator=generator
    )
    result.images[0].save(output_path / f"{prompt}.png")


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

        logger.info("Scores avg: %s", np.mean(np.array(scores)))
        f.write(f"| Avg | {np.mean(np.array(scores))} |\n")


def get_mse_scores(prompts: list[str], unoptimized_path: Path, optimized_path: Path, train_num: int):
    train_error = []
    test_error = []
    with open(optimized_path / "mse_scores.txt", "w") as f:
        f.write("| Prompt | Error |\n")
        for i, prompt in enumerate(prompts):
            error = calc_error(unoptimized_path / f'{prompt}.png', optimized_path / f'{prompt}.png')
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

def get_fid_scores(prompts: list[str], path: Path, real_images):
    with torch.no_grad():
        with open(path / "fid_scores.txt", "w") as f:
            f.write("| Prompt | Score |\n")
            images = []
            for prompt in prompts:
                image = Image.open(path / f"{prompt}.png")
                image = torch.tensor(np.array(image), dtype=torch.uint8).permute(2, 0, 1)
                images.append(image)
            images = torch.cat(images)

            fid = FrechetInceptionDistance()
            fid.update(real_images, real=True)
            fid.update(images, real=False)

            score = fid.compute()
            logger.info("FID: %f", score)
            f.write(f"| FID | {score} |\n")


def get_real_images(train_data, num_data):
    with torch.no_grad():
        images = []
        for i, example in enumerate(train_data):
            if i >= num_data:
                break
            image = Image.open(io.BytesIO(example["image"])).convert("RGB")
            image = torch.tensor(np.array(image), dtype=torch.uint8).permute(2, 0, 1)
            images.append(image)
        return torch.cat(images)


def main(raw_args=None):
    args = parse_args(raw_args)
    real_images = None
    if args.dataset != "relaion2B-en-research-safe":
        dataset = load_dataset(args.dataset, streaming=True)
        train_data = dataset['train']
        prompts = [example["captions"][0] for i, example in enumerate(train_data) if i < args.num_data]
        real_images = get_real_images(train_data, args.num_data)
    else:
        # Some selected captions from https://huggingface.co/datasets/laion/relaion2B-en-research-safe
        prompts = [
            "Arroyo Hondo Preserve Wedding",
            "Budget-Friendly Thanksgiving Table Decor Ideas",
            "Herd of cows on alpine pasture among mountains in Alps, northern Italy. Stock Photo",
            "Hot Chocolate With Marshmallows, Warm Happiness To Soon Follow",
            "Lovely Anthodium N Roses Arrangement with Cute Teddy",
            "Everyone can join and learn how to cook delicious dishes with us.",
            "Image result for youth worker superhero",
            "Road improvements coming along in west Gulfport",
            "Butcher storefront and a companion work, Louis Hayet, Click for value",
            "folding electric bike"
        ][:args.num_data]
    
    train_num = math.floor(len(prompts) * args.train_ratio)
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    script_dir = Path(__file__).resolve().parent
    unoptimized_path = script_dir / args.data_dir / 'unoptimized'
    optimized_path = script_dir / args.data_dir / 'optimized'

    unoptimized_model_dir = script_dir / "models" / "unoptimized" / args.model_id
    optimized_dir_name = "optimized-qnn"
    optimized_model_dir = script_dir / "models" / optimized_dir_name / args.model_id

    model_dir = unoptimized_model_dir if args.save_data else optimized_model_dir
    pipeline = QnnStableDiffusionPipeline.from_pretrained(
        model_dir, provider="CPUExecutionProvider"
    )
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

    else:
        os.makedirs(optimized_path, exist_ok=True)
        for i, prompt in enumerate(prompts):
            logger.info(prompt)
            run_inference(pipeline, args, prompt, optimized_path)

        get_clip_scores(prompts, optimized_path, clip_score_fn)
        get_fid_scores(prompts, optimized_path, real_images)
        get_mse_scores(prompts, unoptimized_path, optimized_path, train_num)


if __name__ == "__main__":
    main()
