#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import copy
import argparse
from transformers import GPT2Tokenizer
import os
import numpy as np
from quark.onnx.quantization.config import (Config, get_default_config)
from quark.onnx import ModelQuantizer
from data_preparation import get_calib_dataloader

class CalibrationDataReader:

    def __init__(self, dataloader):
        super().__init__()
        self.iterator = iter(dataloader)

    def get_next(self) -> dict:
        try:
            inputs = next(self.iterator)[0]
            input_dict = {}
            input_dict["input_ids"] = inputs.numpy().reshape(1, -1)
            input_dict["attention_mask"] = np.ones_like(inputs.numpy().reshape(
                1, -1))
            return input_dict
        except StopIteration:
            return None

def main(args: argparse.Namespace) -> None:
    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = args.input_model_path

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = args.output_model_path

    tokenizer_class = GPT2Tokenizer
    tokenizer = tokenizer_class.from_pretrained(
        os.path.dirname(args.input_model_path),
        do_lower_case=False,
        cache_dir=None,
    )

    # `dr` (Data Reader) is an instance of DataReader, which is a utility class that
    # reads the calibration dataset and prepares it for the quantization process.
    calib_dataloader = get_calib_dataloader(dataset_name='pileval',
                                            tokenizer=tokenizer,
                                            batch_size=1,
                                            seqlen=512,
                                            device=args.device
                                            )
    calib_dataloader = CalibrationDataReader(calib_dataloader)
    # Get quantization configuration
    quant_config = get_default_config(args.config)
    config_copy = copy.deepcopy(quant_config)
    config_copy.extra_options['OpTypesToExcludeOutputQuantization'] = ['MatMul', 'Gemm']
    config_copy.include_sq = args.include_sq
    if args.include_sq:
        config_copy.extra_options['SmoothAlpha'] = 0.8
    config = Config(global_quant_config=config_copy)
    print(f"The configuration for quantization is {config}")

    # Create an ONNX quantizer
    quantizer = ModelQuantizer(config)

    # Quantize the ONNX model
    quantizer.quantize_model(input_model_path, output_model_path, calib_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_model_path", help="Specify the input model to be quantized", required=True)
    parser.add_argument("--output_model_path",
                        help="Specify the path to save the quantized model",
                        type=str,
                        default='',
                        required=False)
    parser.add_argument("--include_sq", action='store_true', help="Optimize the models using SmoothQuant")
    parser.add_argument("--num_calib_data", help="Number of samples for calibration", type=int, default=1000)
    parser.add_argument("--batch_size", help="Batch size for calibration", type=int, default=1)
    parser.add_argument("--workers", help="Number of worker threads used during calib data loading.", type=int, default=1)
    parser.add_argument("--device",
                        help="The device type of executive provider, it can be set to 'cpu', 'rocm' or 'cuda'",
                        type=str,
                        default="cpu")
    parser.add_argument("--config", help="The configuration for quantization", type=str, default="INT8_TRANSFORMER_DEFAULT")

    args = parser.parse_args()

    main(args)
