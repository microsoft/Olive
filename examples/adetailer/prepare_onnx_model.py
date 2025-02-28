# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


def download(model_name: str):
    models_dir = Path("./models", model_name.split("_")[0])
    models_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download("Bingsu/adetailer", f"{model_name}.pt", local_dir=f"./{models_dir}/")
    yolo_model = YOLO(f"{models_dir}/{model_name}.pt")
    torch_model = yolo_model.model
    torch.save(torch_model, f"{models_dir}/{model_name}_pytorch.pt")
    yolo_model.export(format="onnx")


download("face_yolov9c")
download("hand_yolov9c")
download("person_yolov8m-seg")
download("deepfashion2_yolov8s-seg")
