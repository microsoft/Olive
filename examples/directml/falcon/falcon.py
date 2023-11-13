# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
from pathlib import Path

import onnxruntime as ort

from olive.workflows import run as olive_run


def optimize(model_name: str, optimized_model_des: Path):
    ort.set_default_logger_severity(4)
    cur_dir = Path(__file__).resolve().parent

    model_info = {}

    # Optimize the model with Olive
    print(f"\nOptimizing {model_name}")

    olive_config = None
    with (cur_dir / "config_falcon.json").open() as fin:
        olive_config = json.load(fin)

    olive_config["input_model"]["config"]["model_path"] = model_name
    olive_run(olive_config)

    footprints_file_path = Path(__file__).resolve().parent / "gpu-dml_footprints.json"
    with footprints_file_path.open("r") as footprint_file:
        footprints = json.load(footprint_file)
        conversion_footprint = None
        for footprint in footprints.values():
            if footprint["from_pass"] == "OnnxConversion":
                conversion_footprint = footprint
        assert conversion_footprint

        model_info = {
            "unoptimized": {
                "path": Path(conversion_footprint["model_config"]["config"]["model_path"]),
            },
        }
        print(f"Unoptimized Model : {model_info['unoptimized']['path']}")

    # Create a copy of the unoptimized model directory
    shutil.copytree(model_info["unoptimized"]["path"], optimized_model_des)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--model", default="tiiuae/falcon-7b", type=str)
    parser.add_argument("--inference", action="store_true", help="Runs the inference step")
    parser.add_argument("--max_length", default=200, type=int, help="max iterations to generate output")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    optimized_model_dir = script_dir / "models" / "optimized" / args.model

    if args.optimize:
        shutil.rmtree(script_dir / "footprints", ignore_errors=True)
        shutil.rmtree(optimized_model_dir, ignore_errors=True)

    if args.optimize or not optimized_model_dir.exists():
        optimize(args.model, optimized_model_dir)

    if args.inference:
        import numpy as np
        from transformers import AutoTokenizer
        from transformers import pipeline
        from transformers import AutoModelForCausalLM
        import torch
        
        model_path = optimized_model_dir / "model.onnx"
        session = ort.InferenceSession(model_path, providers=["DmlExecutionProvider"])
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        args.model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", offload_folder = script_dir / "offload", trust_remote_code=True)
        pipeline_ = pipeline(
            "text-generation",
            model=args.model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        sequences = pipeline_(
           "Girafatron is obsessed with giraffes, the most glorious animal on the face of this Earth. Giraftron believes all other animals are irrelevant when compared to the glorious majesty of the giraffe.\nDaniel: Hello, Girafatron!\nGirafatron:",
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")
