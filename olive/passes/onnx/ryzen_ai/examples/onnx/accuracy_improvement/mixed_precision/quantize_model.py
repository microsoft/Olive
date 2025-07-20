#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
sys.path.append("..")
import copy
import argparse
import numpy
import torch
from typing import List, Any, Union

from onnxruntime.quantization.calibrate import CalibrationMethod
from timm.models import create_model
from timm.data import create_loader, resolve_data_config, create_dataset

from quark.onnx.quantization.config import (Config, get_default_config)
from quark.onnx import ModelQuantizer
from utils.onnx_validate import load_loader


def post_process_top1(output: torch.tensor) -> float:
    _, preds_top1 = torch.max(output, 1)
    return preds_top1

def getAccuracy_top1(preds: Union[torch.tensor, list], targets: Union[torch.tensor, list]) -> float:
    assert len(preds) == len(targets)
    assert len(preds) > 0
    count = 0
    for i in range(len(preds)):
        pred = preds[i]
        target = targets[i]
        if pred == target:
            count += 1
    return count / len(preds)

def top1_acc(results: List[Union[torch.tensor, List[Any]]]) -> float:
    """
    Calculate the top1 accuracy of the model.
    :param results: the result of the model
    :return: the top1 accuracy
    """
    timm_model_name = model_name
    calib_data_path = calibration_dataset_path

    timm_model = create_model(
        timm_model_name,
        pretrained=False,
    )

    data_config = resolve_data_config(model=timm_model, use_test_size=True)

    loader = create_loader(create_dataset('', calib_data_path),
                           input_size=data_config['input_size'],
                           batch_size=20,
                           use_prefetcher=False,
                           interpolation=data_config['interpolation'],
                           mean=data_config['mean'],
                           std=data_config['std'],
                           num_workers=2,
                           crop_pct=data_config['crop_pct'])
    target = []
    for _, labels in loader:
        target.extend(labels.data.tolist())
    outputs_top1 = post_process_top1(torch.tensor(numpy.squeeze(numpy.array(results))))
    top1_acc = getAccuracy_top1(outputs_top1, target)
    return round(top1_acc, 2)

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
    # `model_name` is the name of the original, unquantized ONNX model.
    global model_name
    model_name = args.model_name

    # `input_model_path` is the path to the original, unquantized ONNX model.
    input_model_path = args.input_model_path

    # `output_model_path` is the path where the quantized model will be saved.
    output_model_path = args.output_model_path

    # `calibration_dataset_path` is the path to the dataset used for calibration during quantization.
    global calibration_dataset_path
    calibration_dataset_path = args.calibration_dataset_path

    # `dr` (Data Reader) is an instance of ResNet50DataReader, which is a utility class that
    # reads the calibration dataset and prepares it for the quantization process.
    data_loader = load_loader(model_name, calibration_dataset_path, args.batch_size, args.workers)
    dr = CalibrationDataReader(data_loader)

    # Get quantization configuration
    quant_config = get_default_config(args.config)
    config_copy = copy.deepcopy(quant_config)
    config_copy.calibrate_method = CalibrationMethod.MinMax
    if args.config == "S16S16_MIXED_S8S8":
        config_copy.extra_options['AutoMixprecision']['Top1AccTarget'] = 0.02
        config_copy.extra_options['AutoMixprecision']['EvaluateFunction'] = top1_acc
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
    parser.add_argument("--include_mixed_precision", action='store_true', help="Optimize the models using mixed_precision")
    parser.add_argument("--batch_size", help="Batch size for calibration", type=int, default=1)
    parser.add_argument("--workers", help="Number of worker threads used during calib data loading.", type=int, default=1)
    parser.add_argument("--device",
                        help="The device type of executive provider, it can be set to 'cpu', 'rocm' or 'cuda'",
                        type=str,
                        default="cpu")
    parser.add_argument("--config", help="The configuration for quantization", type=str, default="S16S16_MIXED_S8S8")

    args = parser.parse_args()

    main(args)
