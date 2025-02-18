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


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def parse_args(raw_args):
    import argparse

    parser = argparse.ArgumentParser("Common arguments")
    parser.add_argument("--save_data", action="store_true")
    return parser.parse_args(raw_args)

def mse(image1, image2):
    image1 = Image.open(image1)
    image2 = Image.open(image2)

    import numpy as np

    image1 = np.array(image1, dtype=np.float32)
    image2 = np.array(image2, dtype=np.float32)
    return np.mean((image1 - image2) ** 2)

def main(raw_args=None):
    args = parse_args(raw_args)

    # Some selected captions from https://huggingface.co/datasets/laion/relaion2B-en-research-safe
    prompts = [
        "Arroyo Hondo Preserve Wedding",
        "Herd of cows on alpine pasture among mountains in Alps, northern Italy. Stock Photo",
        "Hot Chocolate With Marshmallows, Warm Happiness To Soon Follow",
        "Lovely Anthodium N Roses Arrangement with Cute Teddy",
        "Everyone can join and learn how to cook delicious dishes with us.",
        "Budget-Friendly Thanksgiving Table Decor Ideas",
        "Image result for youth worker superhero",
        "Road improvements coming along in west Gulfport",
        "Butcher storefront and a companion work, Louis Hayet, Click for value",
        "folding electric bike"
    ]
    command_base = [
        "python", "stable_diffusion.py",
        "--model_id", "stabilityai/stable-diffusion-2-1-base",
        "--provider", "qnn",
        "--num_inference_steps", "5",
        "--seed", "0",
    ]
    train_num = int(len(prompts) * 0.8)
    data_path = Path('quantize_data')
    unoptimized_path = data_path / 'unoptimized'
    optimized_path = data_path / 'optimized'

    if args.save_data:
        os.makedirs(unoptimized_path, exist_ok=True)
        for i, prompt in enumerate(prompts):
            command = command_base + ["--test_unoptimized", "--prompt", prompt]
            if i < train_num:
                command.append("--save_data")
            subprocess.run(command)
            shutil.move('result_0.png', unoptimized_path / f'{prompt}.png')
    else:
        os.makedirs(optimized_path, exist_ok=True)
        train_error = 0
        test_error = 0
        for i, prompt in enumerate(prompts):
            command = command_base + ["--prompt", prompt]
            subprocess.run(command)
            shutil.move('result_0.png', optimized_path / f'{prompt}.png')

            error = math.sqrt(mse(unoptimized_path / f'{prompt}.png', optimized_path / f'{prompt}.png'))
            logger.info("sqrt(mse) for %s: %f", prompt, error)
            if i < train_num:
                train_error += error
            else:
                test_error += error

        logger.info("Train error %f", train_error / train_num)
        logger.info("Test error %f", test_error / (len(prompts) - train_num))


if __name__ == "__main__":
    main()
