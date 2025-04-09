# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# ruff: noqa: T201

import json
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Download Tiny-ImageNet-200 dataset
# You can download it from: http://cs231n.stanford.edu/tiny-imagenet-200.zip
# Extract the contents and update dataset_path accordingly
dataset_path = Path("path_to_tiny_imagenet")
val_images_path = dataset_path / "val" / "images"
val_labels_path = dataset_path / "val" / "val_annotations.txt"

val_img_to_idx = {}
with open(val_labels_path) as f:
    for line in f.readlines():
        parts = line.strip().split("\t")
        val_img_to_idx[parts[0]] = parts[1]

val_idx_to_name = {}
val_words_path = os.path.join(dataset_path, "words.txt")
with open(val_words_path) as f:
    for line in f.readlines():
        parts = line.strip().split("\t")
        if len(parts) == 2:
            val_idx_to_name[parts[0]] = parts[1]


class TinyImageNetDataset(Dataset):
    def __init__(self, img_dir, img_to_idx, transform, limit):
        self.img_dir = img_dir
        self.img_to_idx = img_to_idx
        self.transform = transform
        self.image_filenames = sorted(img_to_idx.keys())[:limit]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.img_to_idx[img_name]
        if self.transform:
            image = self.transform(image)
        return image, label, img_name


val_images_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

val_dataset = TinyImageNetDataset(val_images_path, val_img_to_idx, val_images_transform, 20)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

options = onnxruntime.SessionOptions()
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
vit_session = onnxruntime.InferenceSession(
    "path_to_model.onnx",
    sess_options=options,
    providers=["QNNExecutionProvider"],
    provider_options=[{"backend_path": "QnnHtp.dll"}],
)


def evaluate_onnx_model(session, dataloader):
    total_time = 0
    correct_top1, correct_top5, total = 0, 0, 0
    with open("vit_id2label.json") as file:
        vit_id2label = json.load(file)

    for i, (image, label, img_name) in enumerate(dataloader):
        images_np = image.numpy().astype(np.float32)

        start_time = time.time()
        result = session.run(None, {"input": images_np})
        end_time = time.time()
        total_time += end_time - start_time

        logits = result[0]
        top1_pred = np.argmax(logits, axis=-1).item()
        top5_preds = np.argsort(logits, axis=-1)[0, -5:][::-1]
        ground_truth = val_idx_to_name[label[0]]
        pred_label = vit_id2label["id2label"][str(top1_pred)]

        print(f"Image {i + 1}: {img_name[0]}")
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Top-1 Prediction: {pred_label}")
        print(f"  Top-5 Predictions: {[vit_id2label['id2label'][str(pred)] for pred in top5_preds]}\n")

        correct_top1 += pred_label == ground_truth
        correct_top5 += ground_truth in [vit_id2label["id2label"][str(pred)] for pred in top5_preds]
        total += 1

    print(f"Average Inference Time per Image: {total_time / total:.4f} seconds")
    print(f"Top-1 Accuracy: {correct_top1 / total:.4f}")
    print(f"Top-5 Accuracy: {correct_top5 / total:.4f}")


evaluate_onnx_model(vit_session, val_dataloader)
