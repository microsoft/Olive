#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import cv2
import onnx
import copy
import numpy as np
from argparse import ArgumentParser, Namespace
from quark.onnx.quantization.config.config import Config
from quark.onnx.quantization.config.custom_config import get_default_config
from onnxruntime.quantization import CalibrationDataReader
from quark.onnx import ModelQuantizer

DEFAULT_ADAROUND_PARAMS = {
    'DataSize': 1000,
    'FixedSeed': 1705472343,
    'BatchSize': 2,
    'NumIterations': 1000,
    'LearningRate': 0.1,
    'OptimAlgorithm': 'adaround',
    'OptimDevice': 'cpu',
    'InferDevice': 'cpu',
    'EarlyStop': True,
}

DEFAULT_ADAQUANT_PARAMS = {
    'DataSize': 1000,
    'FixedSeed': 1705472343,
    'BatchSize': 2,
    'NumIterations': 1000,
    'LearningRate': 0.00001,
    'OptimAlgorithm': 'adaquant',
    'OptimDevice': 'cpu',
    'InferDevice': 'cpu',
    'EarlyStop': True,
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

    def rewind(self):
        self.enum_data = None

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input_model_path", help="Specify the input model to be quantized", required=True)
    parser.add_argument("--calib_data_path", help="Specify the calibration data path for quantization", required=True)
    parser.add_argument("--output_model_path",
                        help="Specify the path to save the quantized model",
                        type=str,
                        default='quantized.onnx',
                        required=False)
    parser.add_argument("--config", help="The configuration for quantization", type=str, default="XINT8", required=False)
    parser.add_argument('--cle', action='store_true')
    parser.add_argument('--adaround', action='store_true')
    parser.add_argument('--adaquant', action='store_true')
    parser.add_argument("--learning_rate", help="The learing_rate for fastfinetune", type=float, default=0.1, required=False)
    parser.add_argument("--num_iters", help="The number of iterations for fastfinetune", type=int, default=1000, required=False)
    parser.add_argument("--exclude_nodes", help="The names of excluding nodes", type=str, default='', required=False)
    parser.add_argument('--save_as_external_data', action='store_true')
    args, _ = parser.parse_known_args()
    return args

def main(args: Namespace) -> None:
    quant_config = get_default_config(args.config)
    config_copy = copy.deepcopy(quant_config)
    config_copy.use_external_data_format = args.save_as_external_data
    if args.exclude_nodes:
        exclude_nodes = args.exclude_nodes.split(";")
        exclude_nodes = [node_name.strip() for node_name in exclude_nodes]
        config_copy.nodes_to_exclude = exclude_nodes
    if args.cle:
        config_copy.include_cle = True
    if args.adaround or args.adaquant:
        config_copy.include_fast_ft = True
        if args.adaround:
            config_copy.extra_options['FastFinetune'] = DEFAULT_ADAROUND_PARAMS
        if args.adaquant:
            config_copy.extra_options['FastFinetune'] = DEFAULT_ADAQUANT_PARAMS
        if args.learning_rate:
            config_copy.extra_options['FastFinetune']['LearningRate'] = args.learning_rate
        if args.num_iters:
            config_copy.extra_options['FastFinetune']['NumIterations'] = args.num_iters

    model_input_name = get_model_input_name(args.input_model_path)
    calib_datareader = ImageDataReader(args.calib_data_path, model_input_name)
    quant_config = Config(global_quant_config=config_copy)
    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(args.input_model_path, args.output_model_path, calib_datareader)

if __name__ == '__main__':
    args = parse_args()
    main(args)
