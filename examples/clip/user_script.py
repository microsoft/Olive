# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import CLIPProcessor

from olive.data.registry import Registry


class CLIPDataset(Dataset):
    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        dataset_name="nlphuji/flickr30k",
        start=0,
        end=100,
        image_size=(224, 224),
    ):
        assert 0 <= start < end
        self.start = start
        self.end = end
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.processor = CLIPProcessor.from_pretrained(self.model_name)  # max_length = 77 for input_ids
        self.length = self.end - self.start
        self.image_size = image_size
        dataset = load_dataset(self.dataset_name, split=f"test[{self.start}:{self.end}]")
        text_inputs = self.processor(
            text=[" ".join(item["caption"]) for item in dataset],
            return_tensors="np",
            padding="max_length",
            truncation=True,
        )
        image_inputs = [
            self.processor(images=item["image"].resize(self.image_size), return_tensors="np") for item in dataset
        ]
        self.model_inputs = [
            {
                "input_ids": text_inputs["input_ids"].astype(np.int64),
                "pixel_values": image_inputs[idx]["pixel_values"],
                "attention_mask": text_inputs["attention_mask"].astype(np.int64),
            }
            for idx in range(self.length)
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.model_inputs[idx], torch.Tensor([idx]).to(torch.int32)


@Registry.register_dataset()
def clip_dataset(**kwargs):
    return CLIPDataset(**kwargs)


@Registry.register_post_process()
def clip_post_process(output):
    return output["logits_per_image"].argmax(axis=-1)
