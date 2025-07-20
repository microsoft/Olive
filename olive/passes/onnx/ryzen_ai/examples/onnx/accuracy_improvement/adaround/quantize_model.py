#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
sys.path.append("..")
import copy
import argparse

from onnxruntime.quantization.calibrate import CalibrationMethod

from quark.onnx.quantization.config import (Config, get_default_config)
from quark.onnx import ModelQuantizer
from utils.onnx_validate import load_loader

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
    if calibration_dataset_path == '':
        dr = None
    else:
        data_loader = load_loader(args.model_name, calibration_dataset_path, args.batch_size, args.workers)
        dr = CalibrationDataReader(data_loader)

    # Get quantization configuration
    quant_config = get_default_config(args.config)
    config_copy = copy.deepcopy(quant_config)
    config_copy.calibrate_method = CalibrationMethod.MinMax
    config = Config(global_quant_config=config_copy)
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
    parser.add_argument("--config", help="The configuration for quantization", type=str, default="S8S8_AAWS_ADAROUND")

    args = parser.parse_args()

    main(args)
