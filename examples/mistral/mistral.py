# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
from pathlib import Path

from optimum.onnxruntime import ORTModelForCausalLM
from optimum.utils.save_utils import maybe_save_preprocessors
from transformers import AutoConfig, AutoTokenizer

from olive.workflows import run as olive_run

# ruff: noqa: T201, T203


def optimize(model_name: str, olive_config: dict):
    # Optimize the model with Olive
    print(f"Optimizing {model_name}")

    olive_config["input_model"]["config"]["model_path"] = model_name
    print(olive_run(olive_config))


def inference(model_id: str, optimized_model_dir: Path, execution_provider: str, prompt: str, max_length: int):
    # save any configs that might be needed for inference
    maybe_save_preprocessors(model_id, optimized_model_dir, trust_remote_code=True)
    AutoConfig.from_pretrained(model_id).save_pretrained(optimized_model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ORTModelForCausalLM.from_pretrained(
        optimized_model_dir, execution_provider=execution_provider, use_io_binding=False
    )

    device = "cuda" if execution_provider == "CUDAExecutionProvider" else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print(inputs)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        dest="model_id",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Model Id to load",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the Olive config file",
        choices=["mistral_int4_optimize.json", "mistral_fp16_optimize.json"],
    )
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--inference", action="store_true", help="Runs the inference step")

    args = parser.parse_args()

    with Path(args.config).open() as f:
        config = json.load(f)

    # get ep and accelerator
    ep = config["engine"]["execution_providers"][0]
    ep_header = ep.replace("ExecutionProvider", "").lower()
    accelerator = "gpu" if ep_header == "cuda" else "cpu"

    # output model dir
    script_dir = Path(__file__).resolve().parent
    optimized_model_dir = (
        script_dir
        / config["engine"]["output_dir"]
        / "-".join(config["pass_flows"][0])
        / f"{config['engine']['output_name']}_{accelerator}-{ep_header}_model"
    )

    if args.optimize:
        shutil.rmtree(optimized_model_dir, ignore_errors=True)

    if args.optimize or not optimized_model_dir.exists():
        optimize(args.model_id, config)

    if args.inference:
        prompt = "Is it normal to have a dark ring around the iris of my eye?"
        print(f"Running inference on prompt: {prompt}")
        output = inference(args.model_id, optimized_model_dir, ep, prompt, 50)
        print(f"Output: {output}")


if __name__ == "__main__":
    main()
