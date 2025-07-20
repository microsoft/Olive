#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
sys.path.append("..")
import argparse

import onnxruntime
from utils.onnx_validate import load_loader

from quark.onnx.quantization.config import (Config, get_default_config)
from quark.onnx import ModelQuantizer


class CalibrationDataReader:

    def __init__(self, dataloader):
        super().__init__()
        self.iterator = iter(dataloader)

    def get_next(self) -> dict:
        try:
            return {"input": next(self.iterator)[0].numpy()}
        except Exception:
            return None


def main(args: argparse.Namespace) -> None:
    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = args.input_model_path

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = args.output_model_path

    # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
    calibration_dataset_path = args.calibration_dataset_path

    # `dr` (Data Reader) is an instance of ResNet50DataReader, which is a utility class that
    # reads the calibration dataset and prepares it for the quantization process.
    data_loader = load_loader(args.model_name, calibration_dataset_path, args.batch_size, args.workers)
    dr = CalibrationDataReader(data_loader)

    # Get quantization configuration
    quant_config = get_default_config(args.config)
    if quant_config.include_fast_ft:
        # A larger 'DataSize' brings better accuracy but requires more running time and memory consumption.
        # On the premise of meeting the accuracy requirements, the data size can be appropriately reduced.
        quant_config.extra_options['FastFinetune']['DataSize'] = 100
        # 'LearningRate' is a parameter that has a significant impact on accuracy. Users need to make some
        # attempts to determine the best choice.
        quant_config.extra_options['FastFinetune']['LearningRate'] = 1e-6
        # Use GPU to accelerate the fast finetuning process.
        if args.device != 'cpu':
            quant_config.extra_options['FastFinetune']['OptimDevice'] = 'cuda:0'
            if 'ROCMExecutionProvider' in onnxruntime.get_available_providers():
                quant_config.extra_options['FastFinetune']['InferDevice'] = 'cuda:0'
            elif 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
                quant_config.extra_options['FastFinetune']['InferDevice'] = 'cuda:0'
    config = Config(global_quant_config=quant_config)
    print(f"The configuration for quantization is {config}")

    # Create an ONNX quantizer
    quantizer = ModelQuantizer(config)

    # Quantize the ONNX model
    quantizer.quantize_model(input_model_path, output_model_path, dr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", help="Specify the input model name to be quantized", required=True)
    parser.add_argument("--input_model_path", help="Specify the input model to be quantized", required=True)
    parser.add_argument("--output_model_path",
                        help="Specify the path to save the quantized model",
                        type=str,
                        default='',
                        required=False)
    parser.add_argument("--calibration_dataset_path",
                        help="The path of the dataset for calibration",
                        type=str,
                        default='',
                        required=False)
    parser.add_argument("--num_calib_data", help="Number of samples for calibration", type=int, default=1000)
    parser.add_argument("--batch_size", help="Batch size for calibration", type=int, default=1)
    parser.add_argument("--workers", help="Number of worker threads used during calib data loading.", type=int, default=1)
    parser.add_argument("--device",
                        help="The device type of executive provider, it can be set to 'cpu', 'rocm' or 'cuda'",
                        type=str,
                        default="cpu")
    parser.add_argument("--config", help="The configuration for quantization", type=str, default="BFP16")

    args = parser.parse_args()

    main(args)
