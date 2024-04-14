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

# ruff: noqa: T201


def optimize(model_name: str, optimized_model_des: Path):
    ort.set_default_logger_severity(4)
    cur_dir = Path(__file__).resolve().parent

    model_info = {}

    # Optimize the model with Olive
    print(f"\nOptimizing {model_name}")

    olive_config = None
    with (cur_dir / "config_phi.json").open() as fin:
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
    parser.add_argument("--model", default="microsoft/phi-1", type=str)
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

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        inputs = tokenizer(
            '''def print_prime(n):
               """
               Print all primes between 1 and n
               """''',
            return_tensors="pt",
            return_attention_mask=False,
        )

        model_path = optimized_model_dir / "model.onnx"
        session = ort.InferenceSession(model_path, providers=["DmlExecutionProvider"])

        inputs_numpy = inputs["input_ids"].numpy()
        for _ in range(args.max_length):
            outputs = session.run(["output"], {"input_ids": inputs_numpy})
            prediction = np.argmax(outputs, -1, keepdims=False)[0][0][-1]
            inputs_numpy = np.append(inputs_numpy, [[prediction]], axis=1)

        text = tokenizer.batch_decode(inputs_numpy)[0]
        print(text)
