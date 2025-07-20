#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import argparse
import torch
from typing import Tuple
from argparse import Namespace
import onnxruntime as ort
from evalution import Trainer
from super_gradients.training import models
from quark.onnx.operators.custom_ops import get_library_path
from super_gradients.common.object_names import Models
from super_gradients.training.dataloaders import coco2017_val_yolo_nas, coco2017_val_yolox
from super_gradients.training.metrics import DetectionMetrics_050, DetectionMetrics
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.models import YoloXPostPredictionCallback

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

    torch.onnx.export(model,
                      dummy_input,
                      input_model_path,
                      export_params=True,
                      do_constant_folding=True,
                      opset_version=18,
                      input_names=['input.1'],
                      )
    print(f"ONNX model has been exported successfully at {input_model_path}.")
    return input_model_path

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

def main(args: argparse.Namespace) -> None:
    input_model_path = export_onnx_model(args.model_name)
    mAP_val = yolo_nas_infer(args.model_name, input_model_path, data_dir=args.eval_data_path, use_gpu=args.gpu)
    print(f"{mAP_val}")

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", help="Specify the input model name to be quantized", required=True)
    parser.add_argument("--eval_data_path",
                        help="The path of the folder for evaluation",
                        type=str,
                        default='',
                        required=False)
    parser.add_argument('--gpu', action='store_true', default=False, help='Whether use onnxruntime-gpu to infer.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
