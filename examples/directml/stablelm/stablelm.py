# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
from pathlib import Path

import os
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForCausalLM
from instruct_pipeline import InstructionTextGenerationPipeline

import torch
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from olive.workflows import run as olive_run

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def run_inference(optimized_model_dir, max_new_tokens):
    print("Loading models into ORT session...")
    tokenizer = AutoTokenizer.from_pretrained(optimized_model_dir)
    model = ORTModelForCausalLM.from_pretrained(optimized_model_dir, provider="DmlExecutionProvider", use_cache=True, use_merged=True, use_io_binding=False, pad_token_id=tokenizer.eos_token_id)

    system_prompt = """# StableLM Tuned (Alpha version)
    - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
    - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
    - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
    - StableLM will refuse to participate in anything that could harm a human.
    """

    user_prompt = input("USER: ")

    while user_prompt != "":
        prompt = f"<|SYSTEM|>{system_prompt}<|USER|>{user_prompt}<|ASSISTANT|>"

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        tokens = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.7,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
            pad_token_id=tokenizer.eos_token_id,
        )
        answer = tokenizer.decode(tokens[0], skip_special_tokens=True)
        answer = answer.replace(system_prompt, "")
        answer = answer.replace(user_prompt, "")
        print(f"\nStableLM: {answer}\n")

        user_prompt = input("USER: ")


def optimize(model_name: str, optimized_model_dir: Path, optimize_provider: str):
    ort.set_default_logger_severity(4)
    script_dir = Path(__file__).resolve().parent

    model_info = dict()

    # Optimize the model with Olive
    print(f"\nOptimizing {model_name}")

    olive_config = None
    with open(script_dir / "config_stablelm.json", "r", encoding="utf8") as fin:
        olive_config = json.load(fin)
    if optimize_provider:
        olive_config["passes"]["optimize"]["config"]["target_provider"] = optimize_provider

    olive_run(olive_config)

    # TODO: rename the 0 prefix in the path when the hardware accelerator feature is implemented.
    footprints_file_path = Path(__file__).resolve().parent / "footprints/stablelm_0_footprints.json"
    with footprints_file_path.open("r") as footprint_file:
        footprints = json.load(footprint_file)
        conversion_footprint = None
        merger_footprint = None
        for _, footprint in footprints.items():
            if footprint["from_pass"] == "OptimumConversion":
                conversion_footprint = footprint
            elif footprint["from_pass"] == "OptimumMerging":
                merger_footprint = footprint

        assert conversion_footprint and merger_footprint

        unoptimized_config = conversion_footprint["model_config"]["config"]
        optimized_config = merger_footprint["model_config"]["config"]

        model_info = {
            "unoptimized": {
                "path": os.path.dirname(unoptimized_config["model_components"][0]["config"]["model_path"]),
            },
            "optimized": {
                "path": Path(optimized_config["model_path"]),
            },
        }

        print(f"Unoptimized Model : {model_info['unoptimized']['path']}")
        print(f"Optimized Model   : {model_info['optimized']['path']}")

    # Create a copy of the unoptimized model directory, then overwrite with optimized models from the olive cache.
    shutil.copytree(model_info['unoptimized']['path'], optimized_model_dir, ignore=shutil.ignore_patterns("*.onnx", "*.onnx_data"))

    merged_model_path = str(model_info["optimized"]["path"])
    merged_weights_path = merged_model_path + ".data"

    merged_model_name = os.path.basename(merged_model_path)
    merged_weights_name = merged_model_name + ".data"

    print(f"Copying the optimized model to {optimized_model_dir}")
    shutil.copyfile(merged_model_path, optimized_model_dir / merged_model_name)
    shutil.copyfile(merged_weights_path, optimized_model_dir / merged_weights_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--optimize_provider", type=str, default="directml", help="EP target for inference")
    parser.add_argument("--clean_cache", action="store_true", help="Deletes the Olive cache")
    parser.add_argument("--test_unoptimized", action="store_true", help="Use unoptimized model for inference")
    parser.add_argument("--model", default="stabilityai/stablelm-tuned-alpha-7b", type=str)
    parser.add_argument("--max_new_tokens", default=64, type=int, help="Maximum number of tokens that the model will generate")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    optimized_model_dir = script_dir / "models" / "optimized" / args.model

    if args.clean_cache:
        shutil.rmtree(script_dir / "cache", ignore_errors=True)

    if args.optimize:
        shutil.rmtree(script_dir / "footprints", ignore_errors=True)
        shutil.rmtree(optimized_model_dir, ignore_errors=True)

    if args.optimize or not optimized_model_dir.exists():
        optimize(args.model, optimized_model_dir, args.optimize_provider)

    if not args.optimize:
        run_inference(optimized_model_dir, args.max_new_tokens)
