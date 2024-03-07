# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
from pathlib import Path

import onnxruntime as ort
import torch
from transformers import AutoConfig, LlamaTokenizer

from olive.workflows import run as olive_run

# ruff: noqa: T201, T203


def optimize(model_name: str, optimized_model_des: Path, config: str):
    ort.set_default_logger_severity(4)
    cur_dir = Path(__file__).resolve().parent

    # Optimize the model with Olive
    print(f"\nOptimizing {model_name}")

    olive_config = None
    with (cur_dir / config).open() as fin:
        olive_config = json.load(fin)

    olive_config["input_model"]["config"]["model_path"] = model_name
    olive_run(olive_config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument(
        "--model-id",
        dest="model_id",
        type=str,
        default="mistralai/Mistral-7B-v0.1",
        help="Model Id to load",
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        default="mistral_optimize.json",
        help="Path to the Olive config file",
    )
    parser.add_argument("--inference", action="store_true", help="Runs the inference step")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    optimized_model_dir = script_dir / "models" / "convert-optimize-perf_tuning" / "mistral_gpu-cuda_model"

    if args.optimize:
        shutil.rmtree(optimized_model_dir, ignore_errors=True)

    if args.optimize or not optimized_model_dir.exists():
        optimize(args.model_id, optimized_model_dir, args.config)

    if args.inference:
        prompt = "Is it normal to have a dark ring around the iris of my eye?"

        tokenizer = LlamaTokenizer.from_pretrained(args.model_id)
        tokens = tokenizer(prompt, return_tensors="pt")
        tokenizer = None

        config = AutoConfig.from_pretrained(args.model_id)
        num_heads = config.num_key_value_heads
        head_size = config.hidden_size // config.num_attention_heads
        past_seq_len = 0

        position_ids = tokens.attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(tokens.attention_mask == 0, 1)

        onnx_inputs = {
            "input_ids": tokens.input_ids.numpy(),
            "attention_mask": tokens.attention_mask.numpy(),
            "position_ids": position_ids.numpy(),
        }
        for i in range(config.num_hidden_layers):
            onnx_inputs[f"past_key_values.{i}.key"] = torch.rand(
                1, num_heads // 1, past_seq_len, head_size, dtype=torch.float16
            ).numpy()
            onnx_inputs[f"past_key_values.{i}.value"] = torch.rand(
                1, num_heads // 1, past_seq_len, head_size, dtype=torch.float16
            ).numpy()

        model_path = optimized_model_dir / "model.onnx"

        session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
        session.run(None, onnx_inputs)[0]

        print("Inference test completed successfully!")


if __name__ == "__main__":
    main()
