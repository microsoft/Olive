from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from random import Random

import torch
from transformers import AutoImageProcessor

from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry


@lru_cache(maxsize=1)
def get_imagenet_label_map():
    file_path = Path(__file__).parent / "imagenet_class_index.json"
    if not file_path.exists():
        import requests

        imagenet_class_index_url = (
            "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/imagenet_class_index.json"
        )
        try:
            response = requests.get(imagenet_class_index_url, timeout=10)
        except requests.exceptions.Timeout as e:
            raise RuntimeError(f"Request to {imagenet_class_index_url} timed out after 10 seconds") from e
        response.raise_for_status()
        content = response.json()
    else:
        import json

        with open(file_path) as f:
            content = json.loads(f.read())

    # Convert {0: ["n01440764", "tench"], ...} to {synset: index}
    return {v[0]: int(k) for k, v in content.items()}


@Registry.register_pre_process()
def image_pre_process(
    dataset,
    model_name,
    input_col,
    label_col,
    max_samples=None,
    shuffle=False,
    seed=42,
    **kwargs,
):
    if max_samples is not None:
        max_samples = min(max_samples, len(dataset))
        dataset = dataset.select(
            Random(seed).sample(range(len(dataset)), max_samples) if shuffle else range(max_samples)
        )

    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
    processed_dataset = (
        dataset.align_labels_with_mapping(get_imagenet_label_map(), label_col)
        .map(
            lambda example: {
                **processor([img.convert("RGB") for img in example[input_col]]),
            },
            batched=True,
            remove_columns=[input_col],
        )
        .with_format("torch", output_all_columns=True)
    )

    return BaseDataset(processed_dataset, label_col=label_col)


@Registry.register_post_process()
def image_post_process(output):
    """Post-processing for image classification output."""
    match output:
        case dict() | OrderedDict():
            return output["logits"].argmax(dim=-1)
        case torch.Tensor():
            return output.argmax(dim=-1)
    raise ValueError(f"Unsupported output type: {type(output)}")


def compute_topk_accuracy(logits, labels, k: int = 1):
    topk_preds = torch.argsort(logits, axis=-1)[:, -k:]
    labels = labels.reshape(-1, 1)
    correct = (topk_preds == labels).any(axis=1)
    return {f"top{k}_accuracy": correct.mean()}


def eval_accu_and_f1(outputs, targets, average="macro") -> dict[str, float]:
    from evaluate import load as load_metric

    accu = load_metric("accuracy")
    f1 = load_metric("f1")

    accu_results = accu.compute(
        predictions=outputs.preds,
        references=targets,
    )
    f1_results = f1.compute(
        predictions=outputs.preds,
        references=targets,
        labels=torch.unique(targets),
        average=average,
    )

    return {**(accu_results or {}), **(f1_results or {})}
