#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
from super_gradients.training import models
from super_gradients.common.object_names import Models
from evalution import Trainer
from super_gradients.training.dataloaders import coco2017_val_yolo_nas
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
import onnxruntime as ort
import argparse

# Need your own coco2017 val dataset path
data_dir = r"./coco/"

def yolo_nas_infer(onnx_path, data_dir=data_dir, calib_path="", calib_num=128):
    valid_dataloader = coco2017_val_yolo_nas(dataloader_params={"batch_size": 1}, dataset_params={"data_dir": data_dir})

    model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")

    onnx_session = ort.InferenceSession(onnx_path)

    trainer = Trainer("yolo_nas_experiment")

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

    return result["mAP@0.50"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_model_path",
                        help="Specify the input model to be quantized",
                        default="yolo_nas_s.onnx",
                        required=True)
    parser.add_argument("--data_dir",
                        help="Specify the input model to be quantized",
                        required=True)
    parser.add_argument("--calib_path",
                        default="",
                        help="Specify the input model to be quantized",
                        required=False)
    parser.add_argument("--calib_num",
                        type=int,
                        default=128,
                        help="Specify the input model to be quantized",
                        required=False)
    args = parser.parse_args()

    yolo_nas_infer(args.input_model_path, args.data_dir, args.calib_path, args.calib_num)
