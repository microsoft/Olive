# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
from pathlib import Path
from typing import List

from onnxruntime import __version__ as OrtVersion
from packaging import version

from olive.workflows import run as olive_run

# ruff: noqa: T201


def get_args(raw_args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
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
    parser.add_argument(
        "--prompt",
        nargs="*",
        type=str,
        default=[
            "Is it normal to have a dark ring around the iris of my eye?",
            "Write a extremely long story starting with once upon a time",
        ],
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Max length for generation",
    )

    return parser.parse_args(raw_args)


def optimize(model_name: str, olive_config: dict):
    # Optimize the model with Olive
    print(f"Optimizing {model_name}")
    olive_config["input_model"]["model_path"] = model_name
    olive_run(olive_config)


def inference(model_id: str, optimized_model_dir: Path, execution_provider: str, prompt: List[str], max_length: int):
    import onnxruntime as ort
    from optimum.onnxruntime import ORTModelForCausalLM
    from optimum.utils.save_utils import maybe_save_preprocessors
    from transformers import AutoConfig, AutoTokenizer

    ort.set_default_logger_severity(3)
    # save any configs that might be needed for inference
    maybe_save_preprocessors(model_id, optimized_model_dir, trust_remote_code=True)
    AutoConfig.from_pretrained(model_id).save_pretrained(optimized_model_dir)

    # model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # there is no pad token for this tokenizer, so we set it to eos token
    tokenizer.pad_token = tokenizer.eos_token
    model = ORTModelForCausalLM.from_pretrained(
        optimized_model_dir, provider=execution_provider, use_io_binding=execution_provider == "CUDAExecutionProvider"
    )

    # generate
    device = "cuda" if execution_provider == "CUDAExecutionProvider" else "cpu"
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
    ).to(device)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main(raw_args=None):
    if version.parse(OrtVersion) < version.parse("1.17.0"):
        raise ValueError("This example requires ONNX Runtime 1.17.0 or later.")

    args = get_args(raw_args)

    with Path(args.config).open() as f:
        config = json.load(f)

    # get ep and accelerator
    ep = config["systems"]["local_system"]["accelerators"][0]["execution_providers"][0]
    ep_header = ep.replace("ExecutionProvider", "").lower()
    accelerator = "gpu" if ep_header == "cuda" else "cpu"

    # output model dir
    script_dir = Path(__file__).resolve().parent
    optimized_model_dir = (
        script_dir
        / config["output_dir"]
        / "-".join(config["pass_flows"][0])
        / f"{config['output_name']}_{accelerator}-{ep_header}_model"
    )

    if args.optimize:
        shutil.rmtree(optimized_model_dir, ignore_errors=True)

    if args.optimize or not optimized_model_dir.exists():
        optimize(args.model_id, config)

    if args.inference:
        output = inference(args.model_id, optimized_model_dir, ep, args.prompt, args.max_length)
        for prompt_i, output_i in zip(args.prompt, output):
            print(f"Prompt: {prompt_i}")
            print(f"Generation output: {output_i}")
            print("*" * 50)


if __name__ == "__main__":
    main()
