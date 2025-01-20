# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import sys
from pathlib import Path
import torch
from diffusers import DiffusionPipeline
from packaging import version
from sd_utils import config
from user_script import get_base_model_name
from olive.workflows import run as olive_run


def optimize(
    model_id: str,
    step: str,
    submodel: str,
):
    from google.protobuf import __version__ as protobuf_version

    # protobuf 4.x aborts with OOM when optimizing unet
    if version.parse(protobuf_version) > version.parse("3.20.3"):
        print("This script requires protobuf 3.20.3. Please ensure your package version matches requirements.txt.")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent

    # Clean up previously optimized models, if any.
    #shutil.rmtree(script_dir / "footprints", ignore_errors=True)
    #shutil.rmtree(unoptimized_model_dir, ignore_errors=True)
    #shutil.rmtree(optimized_model_dir, ignore_errors=True)

    # The model_id and base_model_id are identical when optimizing a standard stable diffusion model like
    # CompVis/stable-diffusion-v1-4. These variables are only different when optimizing a LoRA variant.
    base_model_id = get_base_model_name(model_id)

    # Load the entire PyTorch pipeline to ensure all models and their configurations are downloaded and cached.
    # This avoids an issue where the non-ONNX components (tokenizer, scheduler, and feature extractor) are not
    # automatically cached correctly if individual models are fetched one at a time.
    print("Download stable diffusion PyTorch pipeline...")
    pipeline = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float32)

    config.vae_sample_size = pipeline.vae.config.sample_size
    config.cross_attention_dim = pipeline.unet.config.cross_attention_dim
    config.unet_sample_size = pipeline.unet.config.sample_size

    submodel_names = [submodel] # ["vae_encoder", "vae_decoder", "unet", "text_encoder"]

    for submodel_name in submodel_names:
        print(f"\nOptimizing {submodel_name}")

        olive_config = None
        name = f"config_{submodel_name}"
        if step:
            name += f".{step}"
        with (script_dir / (name + ".json")).open() as fin:
            olive_config = json.load(fin)

        run_res = olive_run(olive_config)


def parse_common_args(raw_args):
    parser = argparse.ArgumentParser("Common arguments")
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-2-1-base", type=str)
    parser.add_argument("--model", default="text_encoder", type=str, choices=["text_encoder", "unet", "vae_decoder"])
    parser.add_argument("--step", type=str, help="Runs different step like qnn")
    return parser.parse_known_args(raw_args)


def main(raw_args=None):
    common_args = parse_common_args(raw_args)
    optimize(common_args.model_id, common_args.step, common_args.model)


if __name__ == "__main__":
    main()
