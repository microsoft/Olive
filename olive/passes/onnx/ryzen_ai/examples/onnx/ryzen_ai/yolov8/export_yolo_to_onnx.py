#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from argparse import ArgumentParser, Namespace
from ultralytics import YOLO

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input_model_path", help="Specify the pytorch model path", required=True)
    parser.add_argument("--output_model_path", help="Specify the onnx model path", required=True)
    args, _ = parser.parse_known_args()
    return args

def export_yolov8m_to_onnx(input_model_path, output_model_path):
    model = YOLO(input_model_path)
    onnx_file_path = output_model_path
    model.export(format='onnx', imgsz=640)
    print(f"Model has been exported to {output_model_path}")

if __name__ == "__main__":
    args = parse_args()
    export_yolov8m_to_onnx(args.input_model_path, args.output_model_path)
