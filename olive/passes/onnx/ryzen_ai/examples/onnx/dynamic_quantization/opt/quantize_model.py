#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
from quark.onnx.quantization.config import (Config, get_default_config)
from quark.onnx import ModelQuantizer

def main(args: argparse.Namespace) -> None:
    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = args.input_model_path

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = args.output_model_path

    # Get quantization configuration
    quant_config = get_default_config(args.config)
    config = Config(global_quant_config=quant_config)
    print(f"The configuration for quantization is {config}")

    # Create an ONNX quantizer
    quantizer = ModelQuantizer(config)

    # Quantize the ONNX model
    quantizer.quantize_model(input_model_path, output_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_model_path", help="Specify the input model to be quantized", required=True)
    parser.add_argument("--output_model_path",
                        help="Specify the path to save the quantized model",
                        type=str,
                        default='',
                        required=False)
    parser.add_argument("--config", help="The configuration for quantization", type=str, default="INT8_TRANSFORMER_DEFAULT")

    args = parser.parse_args()

    main(args)
