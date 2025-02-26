# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import shutil
from pathlib import Path
import subprocess
from PIL import Image
import logging
import sys
import math
import numpy as np


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
    return parser.parse_args(raw_args)


def calc_error(image1, image2):
    image1 = Image.open(image1)
    image2 = Image.open(image2)
    image1 = np.array(image1, dtype=np.float32)
    image2 = np.array(image2, dtype=np.float32)
    mse = np.mean((image1 - image2) ** 2)
    return mse


def main(raw_args=None):
    args = parse_args(raw_args)

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
    command_base = [
        "python", "stable_diffusion.py",
        "--model_id", args.model_id,
        "--provider", "qnn",
        "--num_inference_steps", str(args.num_inference_steps),
        "--seed", str(args.seed),
        "--guidance_scale", str(args.guidance_scale),
        "--data_dir", args.data_dir + "/data",
    ]
    train_num = math.floor(len(prompts) * 0.8)
    data_path = Path(args.data_dir)
    unoptimized_path = data_path / 'unoptimized'
    optimized_path = data_path / 'optimized'

    if args.save_data:
        os.makedirs(unoptimized_path, exist_ok=True)
        for i, prompt in enumerate(prompts):
            command = command_base + ["--test_unoptimized", "--prompt", prompt]
            if i < train_num:
                command.append("--save_data")
            logger.info(command)
            subprocess.run(command)
            shutil.move('result_0.png', unoptimized_path / f'{prompt}.png')
    else:
        os.makedirs(optimized_path, exist_ok=True)
        for i, prompt in enumerate(prompts):
            command = command_base + ["--prompt", prompt]
            logger.info(command)
            subprocess.run(command)
            shutil.move('result_0.png', optimized_path / f'{prompt}.png')

        train_error = []
        test_error = []
        for i, prompt in enumerate(prompts):
            error = calc_error(unoptimized_path / f'{prompt}.png', optimized_path / f'{prompt}.png')
            logger.info("| %s | %f |", prompt, error)
            if i < train_num:
                train_error.append(error)
            else:
                test_error.append(error)

        train_error = np.array(train_error)
        test_error = np.array(test_error)
        logger.info("Average train error %f", np.mean(train_error))
        logger.info("Average test error %f", np.mean(test_error))


if __name__ == "__main__":
    main()
