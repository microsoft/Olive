
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#


import os
import copy
import argparse
import cv2
import onnx
import numpy as np
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantType
from quark.onnx.quantization.config import (Config, get_default_config)
from quark.onnx.quant_utils import (PowerOfTwoMethod, ExtendedQuantType, ExtendedQuantFormat)
from quark.onnx import auto_search


class AutoSearchConfig_Default:
    # for s8s8 & s16s8 aaws/asws
    search_space: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
        "activation_type": [QuantType.QInt8, QuantType.QInt16,],
        "weight_type": [QuantType.QInt8,],
        "include_cle": [False],
        "include_fast_ft": [False],
        "extra_options": {
            'ActivationSymmetric': [True, False],
            'WeightSymmetric': [True],
            "CalibMovingAverage": [False, True],
            "CalibMovingAverageConstant": [0.01],
        }
    }

    # for s8s8 aaws/asws
    search_space_s8s8: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
        "activation_type": [QuantType.QInt8,],
        "weight_type": [QuantType.QInt8,],
        "include_cle": [False],
        "include_fast_ft": [False],
        "extra_options": {
            'ActivationSymmetric': [True, False],
            'WeightSymmetric': [True],
            "CalibMovingAverage": [False, True],
            "CalibMovingAverageConstant": [0.01],
            'AlignSlice': [False],
            'FoldRelu': [True],
            'AlignConcat': [True],
        }
    }

    search_space_s8s8_advanced: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
        "activation_type": [QuantType.QInt8],
        "weight_type": [QuantType.QInt8,],
        "include_cle": [False, True],
        "include_fast_ft": [False, True],
        "extra_options": {
            'ActivationSymmetric': [True, False],
            'WeightSymmetric': [True],
            "CalibMovingAverage": [False, True,],
            "CalibMovingAverageConstant": [0.01],
            'AlignSlice': [False],
            'FoldRelu': [True],
            'AlignConcat': [True],
            'FastFinetune': {
                'DataSize': [200,],
                'NumIterations': [1000],
                'OptimAlgorithm': ['adaround'],
                'LearningRate': [0.1],
                'OptimDevice': ['cuda:0'],
                'InferDevice': ['cuda:0'],
                'EarlyStop': [False],
                }
            }
    }

    search_space_s8s8_advanced2: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
        "activation_type": [QuantType.QInt8, QuantType.QInt16,],
        "weight_type": [QuantType.QInt8,],
        "include_cle": [False, True],
        "include_fast_ft": [False, True],
        "extra_options": {
            'ActivationSymmetric': [True, False],
            'WeightSymmetric': [True],
            "CalibMovingAverage": [False, True,],
            "CalibMovingAverageConstant": [0.01],
            'AlignSlice': [False],
            'FoldRelu': [True],
            'AlignConcat': [True],
            'FastFinetune': {
                'DataSize': [200,],
                'NumIterations': [5000],
                'OptimAlgorithm': ['adaquant'],
                'LearningRate': [1e-5,],
                'OptimDevice': ['cuda:0'],
                'InferDevice': ['cuda:0'],
                'EarlyStop': [False],
                }
            }
    }

    # for s16s8 aaws/asws
    search_space_s16s8: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
        "activation_type": [QuantType.QInt16,],
        "weight_type": [QuantType.QInt8,],
        "include_cle": [False],
        "include_fast_ft": [False],
        "extra_options": {
            'ActivationSymmetric': [True, False],
            'WeightSymmetric': [True],
            "CalibMovingAverage": [False, True],
            "CalibMovingAverageConstant": [0.01],
            'AlignSlice': [False],
            'FoldRelu': [True],
            'AlignConcat': [True],
            'AlignEltwiseQuantType': [True]
        }
    }

    search_space_s16s8_advanced: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
        "activation_type": [QuantType.QInt16,],
        "weight_type": [QuantType.QInt8,],
        "include_cle": [False, True],
        "include_fast_ft": [False, True],
        "extra_options": {
            'ActivationSymmetric': [True, False],
            'WeightSymmetric': [True],
            "CalibMovingAverage": [False, True,],
            "CalibMovingAverageConstant": [0.01],
            'AlignSlice': [False],
            'FoldRelu': [True],
            'AlignConcat': [True],
            'AlignEltwiseQuantType': [True],
            'FastFinetune': {
                'DataSize': [200,],
                'NumIterations': [1000],
                'OptimAlgorithm': ['adaround'],
                'LearningRate': [0.1],
                'OptimDevice': ['cuda:0'],
                'InferDevice': ['cuda:0'],
                'EarlyStop': [False],
                }
            }
    }

    search_space_s16s8_advanced2: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax, CalibrationMethod.Percentile],
        "activation_type": [QuantType.QInt16,],
        "weight_type": [QuantType.QInt8,],
        "include_cle": [False, True],
        "include_fast_ft": [False, True],
        "extra_options": {
            'ActivationSymmetric': [True, False],
            'WeightSymmetric': [True],
            "CalibMovingAverage": [False, True,],
            "CalibMovingAverageConstant": [0.01],
            'AlignSlice': [False],
            'FoldRelu': [True],
            'AlignConcat': [True],
            'AlignEltwiseQuantType': [True],
            'FastFinetune': {
                'DataSize': [200,],
                'NumIterations': [5000],
                'OptimAlgorithm': ['adaquant'],
                'LearningRate': [1e-5,],
                'OptimDevice': ['cuda:0'],
                'InferDevice': ['cuda:0'],
                'EarlyStop': [False],
                }
            }
    }

    # for XINT8
    search_space_XINT8: dict[str, any] = {
        "calibrate_method": [PowerOfTwoMethod.MinMSE],
        "activation_type": [QuantType.QUInt8,],
        "weight_type": [QuantType.QInt8,],
        "enable_npu_cnn": [True],
        "include_cle": [False],
        "include_fast_ft": [False],
        "extra_options": {
            'ActivationSymmetric': [True,],
        }
    }

    search_space_XINT8_advanced: dict[str, any] = {
        "calibrate_method": [PowerOfTwoMethod.MinMSE],
        "activation_type": [QuantType.QUInt8,],
        "weight_type": [QuantType.QInt8,],
        "enable_npu_cnn": [True],
        "include_cle": [False, True],
        "include_fast_ft": [True],
        "extra_options": {
            'ActivationSymmetric': [True,],
            'WeightSymmetric': [True],
            "CalibMovingAverage": [False, True,],
            "CalibMovingAverageConstant": [0.01],
            'FastFinetune': {
                'DataSize': [200,],
                'NumIterations': [1000],
                'OptimAlgorithm': ['adaround'],
                'LearningRate': [0.1,],
                'OptimDevice': ['cuda:0'],
                'InferDevice': ['cuda:0'],
                'EarlyStop': [False],
                }
        }
    }

    search_space_XINT8_advanced2: dict[str, any] = {
        "calibrate_method": [PowerOfTwoMethod.MinMSE],
        "activation_type": [QuantType.QUInt8,],
        "weight_type": [QuantType.QInt8,],
        "enable_npu_cnn": [True],
        "include_cle": [False, True],
        "include_fast_ft": [True],
        "extra_options": {
            'ActivationSymmetric': [True,],
            'WeightSymmetric': [True],
            "CalibMovingAverage": [False, True,],
            "CalibMovingAverageConstant": [0.01],
            'FastFinetune': {
                'DataSize': [200,],
                'NumIterations': [5000],
                'OptimAlgorithm': ['adaquant'],
                'LearningRate': [1e-5,],
                'OptimDevice': ['cuda:0'],
                'InferDevice': ['cuda:0'],
                'EarlyStop': [False],
                }
        }
    }

    # for BF16
    search_space_bf16: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax],
        "activation_type": [ExtendedQuantType.QBFloat16],
        "weight_type": [ExtendedQuantType.QBFloat16],
        "quant_format": [ExtendedQuantFormat.QDQ],
        "include_cle": [False],
        "include_fast_ft": [False],
    }

    search_space_bf16_advanced: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax],
        "activation_type": [ExtendedQuantType.QBFloat16],
        "weight_type": [ExtendedQuantType.QBFloat16],
        "quant_format": [ExtendedQuantFormat.QDQ],
        "include_cle": [False],
        "include_fast_ft": [True],
        "extra_options": {
            'FastFinetune': {
                'DataSize': [1000],
                'FixedSeed': [1705472343],
                'BatchSize': [2],
                'NumIterations': [1000],
                'LearningRate': [0.00001],
                'OptimAlgorithm': ['adaquant'],
                'OptimDevice': ['cuda:0'],
                'InferDevice': ['cuda:0'],
                'EarlyStop': [False],
                }
            }
    }

    #  for BFP16
    search_space_bfp16: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax],
        "activation_type": [ExtendedQuantType.QBFP],
        "weight_type": [ExtendedQuantType.QBFP],
        "quant_format": [ExtendedQuantFormat.QDQ],
        "include_cle": [False],
        "include_fast_ft": [False],
        "extra_options": {
            'BFPAttributes': [{
                'bfp_method': "to_bfp",
                'axis': 1,
                'bit_width': 16,
                'block_size': 8,
                'rounding_mode': 2,
            }]
            }
    }

    search_space_bfp16_advanced: dict[str, any] = {
        "calibrate_method": [CalibrationMethod.MinMax],
        "activation_type": [ExtendedQuantType.QBFP],
        "weight_type": [ExtendedQuantType.QBFP],
        "quant_format": [ExtendedQuantFormat.QDQ],
        "include_cle": [False],
        "include_fast_ft": [True],
        "extra_options": {
            'BFPAttributes': [{
                'bfp_method': "to_bfp",
                'axis': 1,
                'bit_width': 16,
                'block_size': 8,
                'rounding_mode': 2,
            }],
            'FastFinetune': {
                'DataSize': [1000],
                'FixedSeed': [1705472343],
                'BatchSize': [2],
                'NumIterations': [1000],
                'LearningRate': [0.00001],
                'OptimAlgorithm': ['adaquant'],
                'OptimDevice': ['cuda:0'],
                'InferDevice': ['cuda:0'],
                'EarlyStop': [False],
            }
            }
    }


    search_metric: str = "L2"
    search_algo: str = "grid_search"  # candidates: "grid_search", "random"
    search_evaluator = None
    search_metric_tolerance: float = .60001
    search_cache_dir: str = "./"
    search_output_dir: str = "./"
    search_log_path: str = "./auto_search.log"

    search_stop_condition: dict[str, any] = {
        "find_n_candidates": 1,
        "iteration_limit": 10000,
        "time_limit": 1000000.0,  # unit: second
    }

def get_model_input_name(input_model_path: str) -> str:
    model = onnx.load(input_model_path)
    model_input_name = model.graph.input[0].name
    return model_input_name

class ImageDataReader(CalibrationDataReader):

    def __init__(self, calibration_image_folder: str, input_name: str):
        self.enum_data = None

        self.input_name = input_name

        self.data_list = self._preprocess_images(
                calibration_image_folder)

    def _preprocess_images(self, image_folder: str):
        data_list = []
        img_names = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]
        for name in img_names:
            input_image = cv2.imread(os.path.join(image_folder, name))
            # Resize the input image. Because the size of Resnet50 is 224.
            input_image = cv2.resize(input_image, (224, 224))
            input_data = np.array(input_image).astype(np.float32)
            # Customer Pre-Process
            input_data = input_data.transpose(2, 0, 1)
            input_size = input_data.shape
            if input_size[1] > input_size[2]:
                input_data = input_data.transpose(0, 2, 1)
            input_data = np.expand_dims(input_data, axis=0)
            input_data = input_data / 255.0
            data_list.append(input_data)

        return data_list

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter([{self.input_name: data} for data in self.data_list])
        return next(self.enum_data, None)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def rewind(self):
        self.enum_data = None

    def reset(self):
        self.enum_data = None


def main(args: argparse.Namespace) -> None:
    quant_config = get_default_config(args.config)
    config_copy = copy.deepcopy(quant_config)
    config_copy.use_external_data_format = args.save_as_external_data
    if args.exclude_nodes:
        exclude_nodes = args.exclude_nodes.split(";")
        exclude_nodes = [node_name.strip() for node_name in exclude_nodes]
        config_copy.nodes_to_exclude = exclude_nodes

    model_input_name = get_model_input_name(args.input_model_path)
    calib_datareader = ImageDataReader(args.calib_data_path, model_input_name)
    quant_config = Config(global_quant_config=config_copy)

    # get auto search config
    if args.auto_search_config == "default_auto_search":
        auto_search_config = AutoSearchConfig_Default()
    else:
        auto_search_config = args.default_auto_search

    # Create auto search instance
    auto_search_ins = auto_search.AutoSearch(
        config = quant_config,
        auto_search_config=auto_search_config,
        model_input=args.input_model_path,
        model_output=args.output_model_path,
        calibration_data_reader=calib_datareader,
    )

    # build search space
    # fixed point
    space1 = auto_search_ins.build_all_configs(auto_search_config.search_space_XINT8)
    space2 = auto_search_ins.build_all_configs(auto_search_config.search_space_s8s8)
    space3 = auto_search_ins.build_all_configs(auto_search_config.search_space_s16s8)
    space4 = auto_search_ins.build_all_configs(auto_search_config.search_space_XINT8_advanced)
    space5 = auto_search_ins.build_all_configs(auto_search_config.search_space_XINT8_advanced2)
    space6 = auto_search_ins.build_all_configs(auto_search_config.search_space_s8s8_advanced)
    space7 = auto_search_ins.build_all_configs(auto_search_config.search_space_s8s8_advanced2)
    space8 = auto_search_ins.build_all_configs(auto_search_config.search_space_s16s8_advanced)
    space9 = auto_search_ins.build_all_configs(auto_search_config.search_space_s16s8_advanced2)
    # bf16 and bfp16
    space10 = auto_search_ins.build_all_configs(auto_search_config.search_space_bf16)
    space11 = auto_search_ins.build_all_configs(auto_search_config.search_space_bfp16)
    space12 = auto_search_ins.build_all_configs(auto_search_config.search_space_bf16_advanced)
    space13 = auto_search_ins.build_all_configs(auto_search_config.search_space_bfp16_advanced)
    auto_search_ins.all_configs = space1 + space2 + space3 + space4 + space5 + space6 + space7 + space8 + space9 + space10 + space11 + space12 + space13

    # Excute the auto search process
    auto_search_ins.search_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_model_path", help="Specify the input model to be quantized", required=True)
    parser.add_argument("--output_model_path",
                        help="Specify the path to save the quantized model",
                        type=str,
                        default='',
                        required=False)
    parser.add_argument("--calib_data_path", help="Specify the calibration data path for quantization", required=True)
    parser.add_argument("--device",
                        help="The device type of executive provider, it can be set to 'cpu', 'rocm' or 'cuda'",
                        type=str,
                        default="cpu")
    parser.add_argument("--config", help="The configuration for quantization", type=str, default="S8S8_AAWS")
    parser.add_argument("--auto_search_config", help="The configuration for auto search quantizaiton setting", type=str, default="default_auto_search")
    parser.add_argument("--exclude_nodes", help="The names of excluding nodes", type=str, default='', required=False)
    parser.add_argument('--save_as_external_data', action='store_true')

    args = parser.parse_args()

    main(args)
