#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import argparse
import os

from onnxruntime_genai.models.builder import create_model

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_path",
        required=True,
        help="Folder to load PyTorch model and associated files from",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Folder to save AWQ-quantized ONNX model and associated files in",
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Create ONNX model
    model_name = None
    input_folder = args.model_path
    output_folder = args.output_path
    precision = "fp32"  # int4 or fp32
    execution_provider = "cpu"
    cache_dir = os.path.join(".", "cache_dir")
    # NOTE export to onnx model
    create_model(model_name, input_folder, output_folder, precision, execution_provider, cache_dir)

if __name__ == "__main__":
    main()
