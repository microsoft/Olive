# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

path = hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt", local_dir= "./models/")
model = YOLO(path)
model.export(format='onnx')
