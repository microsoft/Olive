from functools import lru_cache
from random import Random
from typing import Dict, OrderedDict

import numpy as np
import torch
from torchvision import transforms

from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry


@lru_cache(maxsize=1)
def get_imagenet_label_map():
    import requests

    imagenet_class_index_url = (
        "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/imagenet_class_index.json"
    )
    response = requests.get(imagenet_class_index_url)
    response.raise_for_status()  # Ensure the request was successful

    # Convert {0: ["n01440764", "tench"], ...} to {synset: index}
    return {v[0]: int(k) for k, v in response.json().items()}


def preprocess_image(image):
    # Convert to rgb if
    # 1. black and white image (all 3 channels the same)
    # 2. with alpha channel
    if len(np.shape(image)) == 2 or np.shape(image)[-1] != 3:
        image = image.convert(mode="RGB")

    transformations = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transformations(image).numpy().astype(np.float32)


@Registry.register_pre_process()
def image_pre_process(
    dataset,
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

    label_names = dataset.features[label_col].names
    label_map = get_imagenet_label_map()
    tensor_ds = dataset.map(
        lambda example: {
            "pixel_values": preprocess_image(example[input_col]),
            "class": label_map[label_names[example[label_col]]],
        },
        batched=False,
        remove_columns=dataset.column_names,
    )
    tensor_ds.set_format("torch", output_all_columns=True)

    return BaseDataset(tensor_ds, label_col="class")


@Registry.register_post_process()
def image_post_process(output):
    if isinstance(output, (Dict, OrderedDict)):
        return output["logits"].argmax(dim=-1)
    elif isinstance(output, torch.Tensor):
        return output.argmax(dim=-1)

    raise ValueError(f"Unsupported output type: {type(output)}")
