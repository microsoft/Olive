#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import copy
import argparse
import os
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization.calibrate import CalibrationMethod
from quark.onnx.quantization.config import (Config, get_default_config)
from onnxruntime.quantization.quant_utils import QuantType
from quark.onnx import auto_search

class AutoSearchConfig_Default:
    search_space: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.Entropy],
        "activation_type": [QuantType.QInt8],
        "weight_type": [QuantType.QInt8],
        "include_cle": [False],
        "include_fast_ft": [True],
        "extra_options": {
            "CalibMovingAverage": [True,],
            "CalibMovingAverageConstant": [0.01],
            'FastFinetune': {
                'DataSize': [5,],
                'NumIterations': [10, 50],
                'OptimAlgorithm': ['adaround'],
                'LearningRate': [0.1, 0.01,]
                }
            }
    }

    search_metric: str = "L2"
    search_algo: str = "grid_search"  # candidates: "grid_search", "random"
    search_evaluator = None
    search_metric_tolerance: float = 1.00
    search_cache_dir: str = "./"
    search_output_dir: str = "./"
    search_log_path: str = "./auto_search.log"

    search_stop_condition: dict[str, any] = {
        "find_n_candidates": 2,
        "iteration_limit": 10000,
        "time_limit": 1000000.0,  # unit: second
    }


class CalibrationDataReader:

    def __init__(self, dataloader):
        super().__init__()
        self.iterator = iter(dataloader)

    def get_next(self) -> dict:
        try:
            return {"input": next(self.iterator)[0].numpy()}
        except Exception:
            return None


class ImageDataReader(CalibrationDataReader):

    def __init__(self, calibration_image_folder: str, model_path: str, data_size: int = 100, batch_size: int = 1):
        self.enum_data = None
        # Use inference session to get input shape.
        session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'])
        self.nhwc_data_list = []
        self.all_files = os.listdir(calibration_image_folder)
        self.all_files = [item for item in self.all_files if item.endswith(".npy")]
        if data_size > len(self.all_files):
            data_size = len(self.all_files)
        for i in range(data_size):
            one_item_path = os.path.join(calibration_image_folder, f"sample_{i}.npy")
            one_item = np.load(one_item_path)
            self.nhwc_data_list.append(one_item)

        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{
                self.input_name: nhwc_data
            } for nhwc_data in self.nhwc_data_list])
        return next(self.enum_data, None)

    def get_item(self, idx):
        if idx < self.datasize:
            temp_data = self.nhwc_data_list[idx]
        else:
            pass
        return {self.input_name: temp_data}

    def __getitem__(self, idx):
        return {self.input_name: self.nhwc_data_list[idx]}

    def __len__(self,):
        return self.datasize

    def rewind(self):
        self.enum_data = None

    def reset(self):
        self.enum_data = None


def main(args: argparse.Namespace) -> None:
    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = args.input_model_path

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = args.output_model_path

    # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
    calibration_dataset_path = args.calibration_dataset_path

    # get auto search config
    if args.auto_search_config == "default_auto_search":
        auto_search_config = AutoSearchConfig_Default()
    else:
        auto_search_config = args.default_auto_search

    # Get quantization configuration
    quant_config = get_default_config(args.config)
    config_copy = copy.deepcopy(quant_config)
    config_copy.calibrate_method = CalibrationMethod.MinMax
    config = Config(global_quant_config=config_copy)
    print(f"The configuration for quantization is {config}")

    # Quantize the ONNX model
    # quantizer.quantize_model(input_model_path, output_model_path, dr)
    dr = ImageDataReader(calibration_image_folder="/group/ossdphi_algo_scratch_13/penglu12/yolov3/yolov3_code/cali_dataset",
                         model_path=input_model_path, data_size=10, batch_size=1)

    # Create auto search instance
    auto_search_ins = auto_search.AutoSearch(
        config = config,
        auto_search_config=auto_search_config,
        model_input=input_model_path,
        model_output=output_model_path,
        calibration_data_reader=dr,
    )

    # Excute the auto search process
    auto_search_ins.search_model()


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
    parser.add_argument("--auto_search_config", help="The configuration for auto search quantizaiton setting", type=str, default="default_auto_search")

    args = parser.parse_args()

    main(args)
