#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import argparse
import torch
import onnx
import numpy as np
import onnxruntime as ort
from typing import Tuple
from onnxsim import simplify
from evalution import Trainer
from argparse import Namespace
from quark.onnx import ModelQuantizer
from quark.onnx.operators.custom_ops import get_library_path
from quark.onnx.quantization.config import Config, get_default_config
from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.training.dataloaders import coco2017_val_yolo_nas, coco2017_val_yolox
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.models import YoloXPostPredictionCallback

class ImageDataReader():

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

def export_onnx_model(model_name: str) -> Tuple[str, str]:
    if model_name == "yolo_nas_s":
        model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
    elif model_name == "yolo_nas_m":
        model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")
    elif model_name == "yolo_nas_l":
        model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")
    elif model_name == "yolox_t":
        model = models.get(Models.YOLOX_T, pretrained_weights="coco")
    elif model_name == "yolox_s":
        model = models.get(Models.YOLOX_S, pretrained_weights="coco")
    elif model_name == "yolox_n":
        model = models.get(Models.YOLOX_N, pretrained_weights="coco")
    elif model_name == "yolox_m":
        model = models.get(Models.YOLOX_M, pretrained_weights="coco")
    elif model_name == "yolox_l":
        model = models.get(Models.YOLOX_L, pretrained_weights="coco")
    elif model_name == "yolox_x":
        model = models.get(Models.YOLOX_X, pretrained_weights="coco")

    model = model.eval()
    model.prep_model_for_conversion(input_size=[1, 3, 640, 640])

    dummy_input = torch.rand(1, 3, 640, 640)
    os.makedirs("models", exist_ok=True)

    input_model_path = "models/" + model_name + ".onnx"
    quantize_model_path = "models/" + model_name + "_quantized.onnx"

    torch.onnx.export(model,
                      dummy_input,
                      input_model_path,
                      export_params=True,
                      do_constant_folding=True,
                      opset_version=18,
                      input_names=['input.1'],
                      )

    model = onnx.load(input_model_path)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save_model(model_simp, input_model_path)
    print(f"ONNX model has been exported successfully at {input_model_path}.")
    return input_model_path, quantize_model_path

def yolo_nas_infer(model_name, onnx_path, data_dir, calib_path="", calib_num=128, use_gpu=False):
    if "yolo_nas" in model_name:
        valid_dataloader = coco2017_val_yolo_nas(dataloader_params={"batch_size": 1}, dataset_params={"data_dir": data_dir})
    elif "yolox_" in model_name:
        valid_dataloader = coco2017_val_yolox(dataloader_params={"batch_size": 1}, dataset_params={"data_dir": data_dir})

    # Set graph optimization level
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if use_gpu:
        providers = ['CUDAExecutionProvider']
        sess_options.register_custom_ops_library(get_library_path("CUDA"))
    else:
        providers = ['CPUExecutionProvider']
        sess_options.register_custom_ops_library(get_library_path("CPU"))

    model = models.get(model_name, pretrained_weights="coco",)
    if onnx_path is not None:
        onnx_session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

    trainer = Trainer("yolo_nas_experiment")

    if "yolo_nas" in model_name:
        metric1 = DetectionMetrics_050(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=80,
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(
                        score_threshold=0.25,
                        nms_top_k=1000,
                        max_predictions=300,
                        nms_threshold=0.7,
                    )
                )
    elif "yolox_" in model_name:
        metric1 = DetectionMetrics(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=80,
                    normalize_targets=True,
                    post_prediction_callback=YoloXPostPredictionCallback(
                        conf = 0.001,
                        iou = 0.6,
                        classes = 80,
                        max_predictions = 300,
                        with_confidence = True,
                        class_agnostic_nms = False,
                        multi_label_per_box = True,
                    )
                )

    if calib_path != "":
        if not os.path.exists(calib_path):
            os.mkdir(calib_path)
        else:
            print(f"calib_path:{calib_path} has existed!")

        if onnx_path is None:
            result = trainer.test(model.eval(), test_loader=valid_dataloader, test_metrics_list=[metric1], calib_path=calib_path, calib_num=calib_num)
        else:
            result = trainer.test(model.eval(), ort_sess=onnx_session, test_loader=valid_dataloader, test_metrics_list=[metric1], calib_path=calib_path, calib_num=calib_num)

        return None

    if onnx_path is None:
        result = trainer.test(model.eval(), test_loader=valid_dataloader, test_metrics_list=[metric1], )
    else:
        result = trainer.test(model.eval(), ort_sess=onnx_session, test_loader=valid_dataloader, test_metrics_list=[metric1], )
    print(result)

    if "yolo_nas" in model_name:
        return result["mAP@0.50"]
    elif "yolox_" in model_name:
        return result["mAP@0.50:0.95"]

def quantize_model(model_name, calibration_data_path: str, input_model_path: str, quantized_model_path: str, config_name: str, use_gpu: bool = False) -> None:
    # reads the calibration data and prepares it for the quantization process.
    dr = ImageDataReader(calibration_data_path, input_model_path, data_size=32, batch_size=1)

    # Get quantization configuration
    quant_config = get_default_config(config_name)
    if model_name == "yolox_s":
        quant_config.subgraphs_to_exclude = [(['/_head/_modules_list.14/Transpose', '/_head/_modules_list.14/Transpose_1', '/_head/_modules_list.14/Transpose_2'], ['/_head/_modules_list.14/Concat_9'])]
    if "ADAROUND" in config_name or "ADAQUANT" in config_name:
        quant_config.extra_options["FastFinetune"]["DataSize"] = 20
        if use_gpu:
            quant_config.extra_options["FastFinetune"]["OptimDevice"] = "cuda:0"
    config = Config(global_quant_config=quant_config)
    print(f"The configuration for quantization is {config}")

    # Create an ONNX quantizer
    quantizer = ModelQuantizer(config)

    # Quantize the ONNX model
    quantizer.quantize_model(input_model_path, quantized_model_path, dr)
    print(f"The quantizated model has been saved at {quantized_model_path}")

def main(args: argparse.Namespace) -> None:
    input_model_path, quantized_model_path = export_onnx_model(args.model_name)
    calibration_data_path = args.calib_data_path
    evaluation_data_path = args.eval_data_path
    # prepare calibration data items
    yolo_nas_infer(args.model_name, input_model_path, evaluation_data_path, calib_path=calibration_data_path, use_gpu=False)
    quantize_model(args.model_name, calibration_data_path, input_model_path, quantized_model_path, args.config, args.gpu)
    mAP_val = yolo_nas_infer(args.model_name, quantized_model_path, evaluation_data_path, calib_path='', use_gpu=args.gpu)
    print(f"mAP value:{mAP_val}")

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", help="Specify the input model name to be quantized", required=True)
    parser.add_argument("--data_path",
                        help="The path of the .tar.gz dataset for calibration and evaluation",
                        type=str,
                        default='',
                        required=False)
    parser.add_argument("--calib_data_path",
                        help="The path of the folder for calibration",
                        type=str,
                        default='',
                        required=False)
    parser.add_argument("--eval_data_path",
                        help="The path of the folder for evaluation",
                        type=str,
                        default='',
                        required=False)
    parser.add_argument("--config", help="The configuration for quantization", type=str, default="A8W8", required=False)
    parser.add_argument('--gpu', action='store_true', default=False, help='Whether use onnxruntime-gpu to infer.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
