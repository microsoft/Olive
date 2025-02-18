# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger
from pathlib import Path

import requests
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from olive.data.registry import Registry

logger = getLogger(__name__)

class ImagenetDataset(Dataset):
    def __init__(self, data, model_name):
        self.images = data["images"]
        self.labels = data["labels"]

        imagenet_classes = data["imagenet_classes"]
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.text_inputs = self.processor(text=imagenet_classes.tolist(), return_tensors="np", padding="max_length", truncation=True)

    def __len__(self):
        return min(len(self.images), len(self.labels))

    def __getitem__(self, idx):
        model_input = {
            'input_ids':   torch.tensor(self.text_inputs['input_ids'], dtype=torch.int64),
            'pixel_values':  torch.tensor(self.processor(images=self.images[idx], return_tensors="np")['pixel_values']),
            'attention_mask': torch.tensor(self.text_inputs['attention_mask'], dtype=torch.int64),
        }
        return model_input, torch.tensor([self.labels[idx]], dtype=torch.int32)


@Registry.register_post_process()
def imagenet_post_fun(output):
    return output['logits_per_image'].argmax(axis=1)


@Registry.register_pre_process()
def dataset_pre_process(output_data, **kwargs):
    cache_key = kwargs.get("cache_key")
    size = kwargs.get("size", 256)
    model_name = kwargs.get("model_name")
    cache_file = None
    if cache_key:
        cache_file = Path(f"./cache/data/{cache_key}_{size}.npz")
        if cache_file.exists():
            with np.load(Path(cache_file)) as data:
                return ImagenetDataset(data, model_name)


    labels = []
    images = []
    for i, sample in enumerate(output_data):
        if i >= size:
            break
        image = sample["image"]
        label = sample["label"]
        image = image.convert("RGB").resize((244,244))
        images.append(image)
        labels.append(label)

    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url)
    response.raise_for_status()
    imagenet_classes = response.text.splitlines()

    result_data = {"images":np.array(images), "labels":np.array(labels), "imagenet_classes": np.array(imagenet_classes)}

    if cache_file:
        cache_file.parent.resolve().mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, **result_data)

    return ImagenetDataset(result_data, model_name)
