# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import os
import shutil
from pathlib import Path


def parse_args(raw_args):
    import argparse

    parser = argparse.ArgumentParser("Common arguments")
    parser.add_argument("--save_data", action="store_true")
    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = parse_args(raw_args)

    # Some selected captions from https://huggingface.co/datasets/laion/relaion2B-en-research-safe
    prompts = [
        "Herd of cows on alpine pasture among mountains in Alps, northern Italy. Stock Photo",
        "Hot Chocolate With Marshmallows, Warm Happiness To Soon Follow",
        "Arroyo Hondo Preserve Wedding",
        "Lovely Anthodium N Roses Arrangement with Cute Teddy"
    ]
    prompts.sort()
    data_path = Path('quantize_data')
    unoptimized_path = data_path / 'unoptimized'
    optimized_path = data_path / 'optimized'

    if args.save_data:
        import subprocess

        os.makedirs('quantize_data/', exist_ok=True)
        for prompt in prompts:
            command = [
                "python", "stable_diffusion.py",
                "--model_id", "stabilityai/stable-diffusion-2-1-base",
                "--provider", "qnn",
                "--save_data",
                "--num_inference_steps", "5",
                "--seed", "0",
                "--test_unoptimized",
                "--prompt", prompt
            ]

            # Run the command
            subprocess.run(command)
            shutil.move('result_0.png', unoptimized_path / f'{prompt}.png')


if __name__ == "__main__":
    main()
