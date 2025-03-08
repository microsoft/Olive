import json
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from olive.data.registry import Registry


def resize_image_and_bbox(image, bbox: torch.Tensor, target_size=(800, 800)):
    """Resize the image and convert bbox.

    Resize the image to target_size,
    while converting the bounding box from (x, y, w, h) to a normalized (cx, cy, w, h) format.
    """
    original_w, original_h = image.size
    target_w, target_h = target_size
    scale_w = target_w / original_w
    scale_h = target_h / original_h

    image = image.resize(target_size, Image.Resampling.LANCZOS)

    x, y, w, h = bbox
    x_new = x * scale_w
    y_new = y * scale_h
    w_new = w * scale_w
    h_new = h * scale_h
    cx = (x_new + w_new / 2) / target_w
    cy = (y_new + h_new / 2) / target_h
    w_new /= target_w
    h_new /= target_h
    bbox = torch.tensor([cx, cy, w_new, h_new])

    return image, bbox


transform = transforms.Compose(
    [transforms.Resize(800), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
)


class TableBankDataset(Dataset):
    def __init__(self, data_dir, annotation_file):
        """Initialize the TableBank dataset.

        :param data_dir: Directory containing table images.
        :param annotation_file: Path to the TableBank JSON annotation file.
        """
        self.data_dir = data_dir

        with open(annotation_file, encoding="utf-8") as f:
            data = json.load(f)

        self.images = {img["id"]: img["file_name"] for img in data["images"]}
        self.annotations = data["annotations"]
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann["image_id"]
        image_file = self.images[image_id]
        image_path = os.path.join(self.data_dir, image_file)

        image = Image.open(image_path).convert("RGB")
        image, bbox = resize_image_and_bbox(image, ann["bbox"])

        return {"input": self.transform(image)}, bbox


@Registry.register_dataset()
def dataset_load(data_dir, **kwargs):
    data_dir = Path(data_dir)
    return TableBankDataset(
        data_dir=data_dir / "tablebank_latex_val_small_images",
        annotation_file=data_dir / "tablebank_latex_val_small.json",
    )


@Registry.register_post_process()
def dataset_post_process(outputs):
    m = outputs["logits"].softmax(-1).max(-1)
    pred_labels = m.indices.detach().cpu().numpy()[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]

    # Select only label=0 (table)
    valid_indices = np.where(pred_labels == 0)[0]
    if len(valid_indices) == 0:
        return torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    return pred_bboxes[valid_indices]


def cxcywh_to_xyxy(boxes):
    # Convert (cx, cy, w, h) to (x1, y1, x2, y2).
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def calculate(preds, targets, iou_threshold=0.5):
    """Compute mean of maximum IoU and F1-score.

    The TableBank datasets consists of images containing multiple tables, while the annotation only marks one table.
    The model outputs preds with multiple tables.
    Therefore, calculate mean of maximum IoU and F1-score.
    """
    preds_xyxy = cxcywh_to_xyxy(preds)
    targets_xyxy = cxcywh_to_xyxy(targets[:, None, :])

    inter_x1 = torch.maximum(preds_xyxy[..., 0], targets_xyxy[..., 0])
    inter_y1 = torch.maximum(preds_xyxy[..., 1], targets_xyxy[..., 1])
    inter_x2 = torch.minimum(preds_xyxy[..., 2], targets_xyxy[..., 2])
    inter_y2 = torch.minimum(preds_xyxy[..., 3], targets_xyxy[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    preds_area = (preds_xyxy[..., 2] - preds_xyxy[..., 0]) * (preds_xyxy[..., 3] - preds_xyxy[..., 1])
    targets_area = (targets_xyxy[..., 2] - targets_xyxy[..., 0]) * (targets_xyxy[..., 3] - targets_xyxy[..., 1])

    union_area = preds_area + targets_area - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)
    max_iou, _ = iou.max(dim=1)
    mean_iou = max_iou.mean().item()

    tp = (max_iou >= iou_threshold).sum().item()
    fp = (max_iou < iou_threshold).sum().item()
    fn = targets.shape[0] - tp

    precision = tp / (tp + fp + 1e-6)  # Avoid division by zero
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    return mean_iou, f1_score


def evaluate(outputs, targets):
    mean_iou, f1_score = calculate(outputs.preds, targets)
    return {"mean_of_max_iou": mean_iou, "f1_score": f1_score}
